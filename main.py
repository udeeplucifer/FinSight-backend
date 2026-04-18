import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time

import requests
import yfinance as yf
from dotenv import load_dotenv
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai

# ----------- INIT ----------- #
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
http_session = requests.Session()
http_session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
gemini_disabled_reason = None

# Use an OS-native temp folder so yfinance works on Windows and Linux.
YF_CACHE_DIR = Path(tempfile.gettempdir()) / "yf_cache"
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YF_CACHE_DIR))

# ----------- CACHE ----------- #
stock_cache = {}
profile_cache = {}
CACHE_TTL = 60  # seconds


def normalize_float(value):
    try:
        if value is None:
            return None
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def display_metric(value, *, percentage=False):
    raw = extract_raw_value(value)
    try:
        if raw is None:
            return "N/A"
        num = float(raw)
    except (TypeError, ValueError):
        return "N/A"
    if percentage:
        # Some providers return dividend yield as a ratio (0.0044),
        # others as an already scaled percent (0.44 or 44). Normalize
        # to a human-readable percentage without double-scaling.
        if num <= 1:
            return round(num * 100, 2)
        return round(num, 2)
    return round(num, 2)


def extract_raw_value(value):
    if isinstance(value, dict):
        return value.get("raw", value.get("fmt"))
    return value


def extract_json_object(text: str):
    if not text:
        return None

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def fetch_json(url, params=None):
    response = http_session.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def merge_missing_fields(target, source):
    if not source:
        return target

    for field in [
        "price",
        "market_cap",
        "name",
        "sector",
        "pe_ratio",
        "week_high",
        "week_low",
        "beta",
        "one_year_return",
    ]:
        value = source.get(field)
        if target.get(field) in (None, "", "N/A") and value not in (None, "", "N/A"):
            target[field] = value

    return target


def read_first(mapping, *keys):
    for key in keys:
        try:
            value = mapping.get(key)
        except Exception:
            value = None
        if value is not None:
            return value
    return None


def derive_one_year_return_from_chart(quote, price):
    price_val = normalize_float(price)
    if price_val in (None, 0):
        return None

    closes = [normalize_float(val) for val in quote.get("close", []) if val is not None]
    closes = [val for val in closes if val is not None]
    if not closes:
        return None

    start_price = closes[0]
    if start_price in (None, 0):
        return None

    return round(((price_val - start_price) / start_price) * 100, 2)


def fetch_yahoo_search_profile(ticker: str):
    now = time.time()
    cached = profile_cache.get(ticker)
    if cached and now - cached[1] < CACHE_TTL:
        return cached[0]

    try:
        payload = fetch_json(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={
                "q": ticker,
                "quotesCount": 10,
                "newsCount": 0,
                "enableFuzzyQuery": "false",
            },
        )
        quotes = payload.get("quotes", [])
        exact = next(
            (
                item
                for item in quotes
                if str(item.get("symbol", "")).upper() == ticker.upper()
            ),
            None,
        )
        best = exact or (quotes[0] if quotes else None)
        if not best:
            return None

        profile = {
            "name": best.get("longname")
            or best.get("shortname")
            or best.get("symbol")
            or ticker,
            "sector": best.get("sectorDisp") or best.get("sector"),
        }
        profile_cache[ticker] = (profile, now)
        return profile
    except Exception:
        logger.warning("Yahoo search profile unavailable for %s", ticker)
        return None


def fetch_fast_info_profile(ticker: str):
    try:
        fast = yf.Ticker(ticker).fast_info
        return {
            "price": read_first(fast, "last_price", "lastPrice"),
            "market_cap": read_first(fast, "market_cap", "marketCap"),
            "week_high": read_first(fast, "year_high", "yearHigh"),
            "week_low": read_first(fast, "year_low", "yearLow"),
            "one_year_return": normalize_float(read_first(fast, "year_change", "yearChange")),
        }
    except Exception:
        logger.warning("Fast info enrichment unavailable for %s", ticker)
        return None


def fetch_info_profile(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        if not info:
            return None
        return {
            "market_cap": info.get("marketCap"),
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector"),
            "pe_ratio": info.get("trailingPE"),
            "week_high": info.get("fiftyTwoWeekHigh"),
            "week_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "one_year_return": normalize_float(info.get("52WeekChange")),
        }
    except Exception:
        logger.warning("Ticker info enrichment unavailable for %s", ticker)
        return None


def fetch_stock_from_yahoo_api(ticker: str):
    chart_data = fetch_json(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        params={
            "range": "1y",
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,splits",
        },
    )
    result = chart_data.get("chart", {}).get("result", [])
    if not result:
        return None

    history = result[0]
    meta = history.get("meta", {})
    quote_list = history.get("indicators", {}).get("quote", [])
    quote = quote_list[0] if quote_list else {}

    closes = [value for value in quote.get("close", []) if value is not None]
    highs = [value for value in quote.get("high", []) if value is not None]
    lows = [value for value in quote.get("low", []) if value is not None]

    price = (
        meta.get("regularMarketPrice")
        or meta.get("previousClose")
        or meta.get("chartPreviousClose")
        or (closes[-1] if closes else None)
    )
    if price is None:
        return None

    data = {
        "price": price,
        "market_cap": None,
        "name": meta.get("longName")
        or meta.get("shortName")
        or meta.get("symbol")
        or ticker,
        "sector": None,
        "pe_ratio": None,
        "week_high": max(highs) if highs else None,
        "week_low": min(lows) if lows else None,
        "beta": None,
        "one_year_return": derive_one_year_return_from_chart(quote, price),
    }

    try:
        summary_data = fetch_json(
            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}",
            params={
                "modules": ",".join(
                    [
                        "price",
                        "summaryDetail",
                        "defaultKeyStatistics",
                        "assetProfile",
                    ]
                )
            },
        )
        summary_result = summary_data.get("quoteSummary", {}).get("result", []) or [{}]
        modules = summary_result[0]
        price_module = modules.get("price", {})
        summary_detail = modules.get("summaryDetail", {})
        stats_module = modules.get("defaultKeyStatistics", {})
        profile_module = modules.get("assetProfile", {})

        data.update(
            {
                "market_cap": extract_raw_value(price_module.get("marketCap")),
                "name": price_module.get("longName")
                or price_module.get("shortName")
                or data["name"],
                "sector": profile_module.get("sector"),
                "pe_ratio": extract_raw_value(summary_detail.get("trailingPE")),
                "week_high": extract_raw_value(summary_detail.get("fiftyTwoWeekHigh"))
                or data["week_high"],
                "week_low": extract_raw_value(summary_detail.get("fiftyTwoWeekLow"))
                or data["week_low"],
                "beta": extract_raw_value(stats_module.get("beta")),
            }
        )
    except Exception:
        logger.warning("Quote summary unavailable for %s; using chart-only data", ticker)

    merge_missing_fields(data, fetch_yahoo_search_profile(ticker))
    merge_missing_fields(data, fetch_fast_info_profile(ticker))
    merge_missing_fields(data, fetch_info_profile(ticker))

    return data


def fetch_stock_from_yfinance(ticker: str):
    # Let yfinance manage its own HTTP session. Recent versions reject
    # manually injected requests.Session objects.
    stock = yf.Ticker(ticker)

    # Try fast_info first (lightweight)
    fast = stock.fast_info

    data = {
        "price": read_first(fast, "last_price", "lastPrice"),
        "market_cap": read_first(fast, "market_cap", "marketCap"),
        "name": ticker,
        "sector": None,
        "pe_ratio": None,
        "week_high": read_first(fast, "year_high", "yearHigh"),
        "week_low": read_first(fast, "year_low", "yearLow"),
        "beta": None,
        "one_year_return": normalize_float(read_first(fast, "year_change", "yearChange")),
    }

    # Fallback to info only if needed.
    try:
        info = stock.info
        data.update(
            {
                "name": info.get("longName") or info.get("shortName") or ticker,
                "sector": info.get("sector"),
                "pe_ratio": info.get("trailingPE"),
                "week_high": info.get("fiftyTwoWeekHigh"),
                "week_low": info.get("fiftyTwoWeekLow"),
                "beta": info.get("beta"),
                "one_year_return": normalize_float(info.get("52WeekChange")),
            }
        )
    except Exception:
        logger.exception("Unable to load extended stock info for %s", ticker)

    if data["price"] is None:
        return None

    return data


def get_cached_stock(ticker):
    now = time.time()

    if ticker in stock_cache:
        data, timestamp = stock_cache[ticker]
        if now - timestamp < CACHE_TTL:
            return data

    try:
        data = fetch_stock_from_yahoo_api(ticker)
        if data is None:
            logger.warning("Yahoo API quote fallback triggered for %s", ticker)
            data = fetch_stock_from_yfinance(ticker)

        if data is None or data.get("price") is None:
            logger.warning("No price returned for ticker %s", ticker)
            return None

        stock_cache[ticker] = (data, now)
        time.sleep(1)  # rate limit protection
        return data
    except Exception:
        logger.exception("Direct Yahoo API fetch failed for %s", ticker)

        try:
            data = fetch_stock_from_yfinance(ticker)
        except Exception:
            logger.exception("Stock fetch error for %s", ticker)
            return None

        if data is None or data.get("price") is None:
            logger.warning("No price returned for ticker %s", ticker)
            return None

        stock_cache[ticker] = (data, now)
        time.sleep(1)  # rate limit protection
        return data


def get_stock_history_data(ticker: str, range_key: str = "1Y"):
    range_map = {
        "1M": ("1mo", "1d"),
        "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1d"),
        "3Y": ("3y", "1wk"),
        "5Y": ("5y", "1wk"),
    }

    period, interval = range_map.get(range_key.upper(), ("1y", "1d"))

    try:
        chart_data = fetch_json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={
                "range": period,
                "interval": interval,
                "includePrePost": "false",
                "events": "div,splits",
            },
        )
        result = chart_data.get("chart", {}).get("result", [])
        if not result:
            logger.warning("No historical data returned for %s (%s)", ticker, range_key)
            return {"prices": [], "volumes": []}

        history = result[0]
        timestamps = history.get("timestamp", [])
        quote_list = history.get("indicators", {}).get("quote", [])
        if not timestamps or not quote_list:
            logger.warning("No historical data returned for %s (%s)", ticker, range_key)
            return {"prices": [], "volumes": []}

        quote = quote_list[0]
        opens = quote.get("open", [])
        closes = quote.get("close", [])
        volumes_raw = quote.get("volume", [])

        prices = []
        volumes = []
        for idx, timestamp in enumerate(timestamps):
            close_val = normalize_float(closes[idx] if idx < len(closes) else None)
            open_val = normalize_float(opens[idx] if idx < len(opens) else None)
            volume_val = volumes_raw[idx] if idx < len(volumes_raw) else 0

            if close_val is None:
                continue

            date_str = time.strftime("%Y-%m-%d", time.localtime(timestamp))
            prices.append({"time": date_str, "value": close_val})
            volumes.append(
                {
                    "time": date_str,
                    "value": int(volume_val) if volume_val is not None else 0,
                    "color": (
                        "rgba(0,229,160,0.3)"
                        if open_val is None or close_val >= open_val
                        else "rgba(255,77,109,0.3)"
                    ),
                }
            )

        return {"prices": prices, "volumes": volumes}
    except Exception:
        logger.exception("Direct Yahoo history lookup failed for %s", ticker)

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval, auto_adjust=False)

        if hist.empty:
            logger.warning("No historical data returned from yfinance for %s (%s)", ticker, range_key)
            return {"prices": [], "volumes": []}

        prices = []
        volumes = []
        for date, row in hist.iterrows():
            close_val = normalize_float(row.get("Close"))
            open_val = normalize_float(row.get("Open"))
            volume_val = row.get("Volume")

            if close_val is None:
                continue

            date_str = date.strftime("%Y-%m-%d")
            prices.append({"time": date_str, "value": close_val})
            volumes.append(
                {
                    "time": date_str,
                    "value": int(volume_val) if volume_val is not None else 0,
                    "color": (
                        "rgba(0,229,160,0.3)"
                        if open_val is None or close_val >= open_val
                        else "rgba(255,77,109,0.3)"
                    ),
                }
            )

        return {"prices": prices, "volumes": volumes}
    except Exception:
        logger.exception("History lookup failed for %s", ticker)
        return {"prices": [], "volumes": []}


def build_compare_stock_payload(ticker: str):
    data = get_cached_stock(ticker.upper())
    if not data:
        return None

    return {
        "ticker": ticker.upper(),
        "name": data.get("name") or ticker.upper(),
        "price": normalize_float(data.get("price")) or 0,
        "market_cap": format_market_cap(data.get("market_cap")),
        "sector": data.get("sector") or "N/A",
        "pe_ratio": display_metric(data.get("pe_ratio")),
        "week_high": display_metric(data.get("week_high")),
        "week_low": display_metric(data.get("week_low")),
        "beta": display_metric(data.get("beta")),
        "one_year_return": display_metric(data.get("one_year_return"), percentage=True),
    }


# ----------- HELPERS ----------- #
def format_market_cap(val):
    if val is None:
        return "N/A"
    if val >= 1_000_000_000_000:
        return f"${val / 1_000_000_000_000:.2f}T"
    if val >= 1_000_000_000:
        return f"${val / 1_000_000_000:.2f}B"
    return f"${val / 1_000_000:.2f}M"


def get_rule_based_verdict(name, price, sector, pe_ratio, week_high, week_low):
    score = 0
    reasons = []

    price_val = normalize_float(price)
    high_val = normalize_float(week_high)
    low_val = normalize_float(week_low)
    pe_val = normalize_float(pe_ratio)

    if (
        price_val is not None
        and high_val is not None
        and low_val is not None
        and high_val > low_val
    ):
        range_position = (price_val - low_val) / (high_val - low_val)
        if range_position <= 0.35:
            score += 1
            reasons.append("trading in the lower part of its 52-week range")
        elif range_position >= 0.8:
            score -= 1
            reasons.append("trading close to its 52-week high")
        else:
            reasons.append("trading near the middle of its 52-week range")

    if pe_val is not None:
        if pe_val < 20:
            score += 1
            reasons.append("valuation looks reasonable on earnings")
        elif pe_val > 35:
            score -= 1
            reasons.append("valuation looks relatively expensive")

    if score >= 2:
        verdict = "BUY"
        confidence = 76
    elif score <= -1:
        verdict = "SELL"
        confidence = 72
    else:
        verdict = "HOLD"
        confidence = 68

    summary_bits = []
    if reasons:
        summary_bits.append(f"{name} is {', '.join(reasons[:2])}.")
    if sector and sector != "N/A":
        summary_bits.append(f"It is in the {sector} sector.")
    summary_bits.append("This is a rule-based fallback, not live investment advice.")

    return {
        "verdict": verdict,
        "confidence": confidence,
        "summary": " ".join(summary_bits),
    }


def get_ai_verdict(name, price, sector, pe_ratio, week_high, week_low):
    global gemini_disabled_reason

    if gemini_disabled_reason or not os.getenv("GEMINI_API_KEY"):
        return get_rule_based_verdict(name, price, sector, pe_ratio, week_high, week_low)

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"""
You are a calm stock research assistant for a retail-investing dashboard.
Use the data provided below and respond with practical, balanced language.
Do not mention missing fields unless they materially affect the conclusion.

Stock: {name}
Current price: {price}
Sector: {sector}
PE ratio: {pe_ratio}
52-week high: {week_high}
52-week low: {week_low}

Respond with a single JSON object only.
Use exactly this schema:
{{"verdict":"BUY|SELL|HOLD","confidence":75,"summary":"short reason"}}
""",
        )
        response_text = getattr(response, "text", "") or ""
        parsed = extract_json_object(response_text)
        if not parsed:
            raise ValueError(f"Could not parse AI JSON: {response_text}")

        verdict = str(parsed.get("verdict", "HOLD")).upper()
        if verdict not in {"BUY", "SELL", "HOLD"}:
            verdict = "HOLD"

        confidence = parsed.get("confidence", 70)
        try:
            confidence = max(0, min(100, int(float(confidence))))
        except (TypeError, ValueError):
            confidence = 70

        summary = str(parsed.get("summary", "")).strip() or "Basic analysis unavailable."

        return {
            "verdict": verdict,
            "confidence": confidence,
            "summary": summary,
        }
    except Exception as exc:
        error_message = str(exc)
        if (
            "quota" in error_message.lower()
            or "billing" in error_message.lower()
            or "429" in error_message.lower()
        ):
            gemini_disabled_reason = "insufficient_quota"
            logger.warning(
                "Gemini quota unavailable; using rule-based verdict fallback"
            )
            return get_rule_based_verdict(
                name, price, sector, pe_ratio, week_high, week_low
            )
        logger.exception("Gemini verdict generation failed for %s", name)
        return get_rule_based_verdict(name, price, sector, pe_ratio, week_high, week_low)


def is_relevant_news(article, company_name, ticker):
    title = (article.get("title") or "").lower()
    description = (article.get("description") or "").lower()
    haystack = f"{title} {description}"

    ticker_token = ticker.lower()
    name_tokens = [
        token.lower()
        for token in re.split(r"[^a-zA-Z0-9]+", company_name or "")
        if len(token) >= 4 and token.lower() not in {"inc", "corp", "group", "ltd", "plc"}
    ]

    if ticker_token and ticker_token in haystack:
        return True
    return any(token in haystack for token in name_tokens[:3])


def is_finance_context(article, ticker):
    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
    finance_terms = {
        "stock",
        "stocks",
        "share",
        "shares",
        "market",
        "markets",
        "earnings",
        "revenue",
        "profit",
        "invest",
        "investing",
        "investor",
        "nasdaq",
        "nyse",
        "valuation",
        "price target",
        ticker.lower(),
    }
    return any(term in text for term in finance_terms)


def is_clean_english_headline(title):
    if not title:
        return False

    lowered = title.lower()
    blocked_terms = {
        "job",
        "hiring",
        "engineer",
        "developer",
        "career",
        "vacancy",
        "recruit",
    }
    if any(term in lowered for term in blocked_terms):
        return False

    meaningful_chars = [ch for ch in title if ch.isalpha()]
    if not meaningful_chars:
        return True

    ascii_chars = [ch for ch in meaningful_chars if ord(ch) < 128]
    return (len(ascii_chars) / len(meaningful_chars)) >= 0.85


def score_news_article(article, company_name, ticker):
    title = article.get("title") or ""
    description = article.get("description") or ""
    haystack = f"{title} {description}".lower()

    score = 0
    if ticker.lower() in haystack:
        score += 5

    company_name_lower = (company_name or "").lower()
    if company_name_lower and company_name_lower in haystack:
        score += 6

    core_terms = [
        token.lower()
        for token in re.split(r"[^a-zA-Z0-9]+", company_name or "")
        if len(token) >= 4 and token.lower() not in {"inc", "corp", "group", "ltd", "plc"}
    ]
    score += sum(2 for token in core_terms[:2] if token in haystack)

    if is_finance_context(article, ticker):
        score += 3

    return score


def get_news(company_name, ticker):
    try:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key or not company_name:
            return []

        query = f'"{company_name}" OR {ticker}'
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={query}&sortBy=publishedAt&pageSize=15&language=en&apiKey={api_key}"
        )
        res = requests.get(url, timeout=10).json()

        filtered_articles = [
            a
            for a in res.get("articles", [])
            if a.get("title")
            and a.get("url")
            and "consent.yahoo.com" not in a["url"]
            and is_clean_english_headline(a["title"])
            and is_relevant_news(a, company_name, ticker)
            and is_finance_context(a, ticker)
        ]

        ranked_articles = sorted(
            filtered_articles,
            key=lambda article: score_news_article(article, company_name, ticker),
            reverse=True,
        )

        return [
            {
                "title": a["title"],
                "url": a["url"],
            }
            for a in ranked_articles[:5]
        ]
    except Exception:
        logger.exception("News lookup failed for %s", ticker)
        return []


def build_stock_response(ticker: str, data, ai, news):
    return {
        "ticker": ticker.upper(),
        "name": data.get("name") or ticker.upper(),
        "price": normalize_float(data.get("price")) or 0,
        "market_cap": format_market_cap(data.get("market_cap")),
        "sector": data.get("sector") or "N/A",
        "pe_ratio": display_metric(data.get("pe_ratio")),
        "week_high": display_metric(data.get("week_high")),
        "week_low": display_metric(data.get("week_low")),
        "beta": display_metric(data.get("beta")),
        "one_year_return": display_metric(data.get("one_year_return"), percentage=True),
        "verdict": ai.get("verdict"),
        "confidence": ai.get("confidence"),
        "summary": ai.get("summary"),
        "news": news,
    }


# ----------- ROUTES ----------- #
@app.get("/")
def root():
    return {"status": "FinSight AI Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stock/{ticker}")
def get_stock(ticker: str):
    data = get_cached_stock(ticker.upper())

    if not data:
        return {"error": "Stock not found"}

    ai = get_ai_verdict(
        data.get("name"),
        data.get("price"),
        data.get("sector"),
        data.get("pe_ratio"),
        data.get("week_high"),
        data.get("week_low"),
    )

    news = get_news(data.get("name"), ticker.upper())

    return build_stock_response(ticker, data, ai, news)


@app.get("/stock/{ticker}/history")
def get_stock_history(ticker: str, range: str = "1Y"):
    return get_stock_history_data(ticker.upper(), range)


@app.get("/compare/{t1}/{t2}")
def compare(t1: str, t2: str):
    s1 = build_compare_stock_payload(t1)
    s2 = build_compare_stock_payload(t2)

    return {
        "stock1": s1,
        "stock2": s2,
        "chart1": get_stock_history_data(t1.upper(), "1Y").get("prices", []),
        "chart2": get_stock_history_data(t2.upper(), "1Y").get("prices", []),
    }


@app.post("/chat")
async def chat(payload: dict = Body(...)):
    question = payload.get("question", "")

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=question,
        )
        answer = (getattr(response, "text", "") or "").strip()
        return {"answer": answer or "Error processing request"}
    except Exception:
        logger.exception("Chat request failed")
        return {"answer": "Error processing request"}

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import os
import requests
import json
import anthropic
from dotenv import load_dotenv
import time

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Fix 429 rate limit
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

def format_market_cap(val):
    if not val:
        return "N/A"
    if val >= 1_000_000_000_000:
        return f"${val/1_000_000_000_000:.2f}T"
    if val >= 1_000_000_000:
        return f"${val/1_000_000_000:.2f}B"
    return f"${val/1_000_000:.2f}M"

def get_ai_verdict(name, price, sector, pe_ratio, week_high, week_low):
    try:
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"""You are an expert stock market analyst. Analyze this stock and give a verdict.

Stock: {name}
Current Price: ${price}
Sector: {sector}
P/E Ratio: {pe_ratio}
52 Week High: ${week_high}
52 Week Low: ${week_low}

Respond in exactly this JSON format with no extra text:
{{"verdict": "BUY" or "SELL" or "HOLD", "confidence": number between 60-95, "summary": "2-3 sentence plain English explanation for beginners"}}"""
            }]
        )
        result = json.loads(message.content[0].text)
        return result
    except Exception as e:
        print(f"❌ Claude Verdict Error: {e}")
        return {
            "verdict": "HOLD",
            "confidence": 70,
            "summary": f"{name} is currently trading at ${price}. Always do your own research before investing."
        }

def get_news(ticker, company_name):
    try:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            return []
        url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
        response = requests.get(url)
        data = response.json()
        articles = data.get("articles", [])
        return [
            {
                "title": a["title"],
                "url": a["url"],
                "source": a["source"]["name"],
                "publishedAt": a["publishedAt"]
            }
            for a in articles if a.get("title") and "[Removed]" not in a["title"]
        ][:5]
    except Exception as e:
        print(f"❌ News Error: {e}")
        return []

@app.get("/")
def root():
    return {"status": "FinSight AI Backend Running! 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stock/{ticker}")
def get_stock(ticker: str):
    try:
        stock = yf.Ticker(ticker, session=session)
        try:
            info = stock.info
        except:
            time.sleep(3)
            info = stock.info

        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        market_cap = format_market_cap(info.get("marketCap"))
        sector = info.get("sector", "N/A")
        name = info.get("longName") or info.get("shortName") or ticker
        pe_ratio = info.get("trailingPE")
        week_high = info.get("fiftyTwoWeekHigh")
        week_low = info.get("fiftyTwoWeekLow")
        beta = info.get("beta")
        dividend_yield = info.get("dividendYield")

        ai_result = get_ai_verdict(name, price, sector, pe_ratio, week_high, week_low)
        news = get_news(ticker, name)

        return {
            "ticker": ticker.upper(),
            "name": name,
            "price": round(price, 2) if price else 0,
            "market_cap": market_cap,
            "sector": sector,
            "pe_ratio": round(pe_ratio, 2) if pe_ratio else "N/A",
            "week_high": week_high,
            "week_low": week_low,
            "beta": round(beta, 2) if beta else "N/A",
            "dividend_yield": round(dividend_yield * 100, 2) if dividend_yield else "N/A",
            "verdict": ai_result["verdict"],
            "confidence": ai_result["confidence"],
            "summary": ai_result["summary"],
            "news": news
        }
    except Exception as e:
        print(f"❌ Stock Error: {e}")
        return {"error": str(e)}

@app.get("/stock/{ticker}/history")
def get_stock_history(ticker: str, range: str = "1Y"):
    try:
        range_map = {
            "1M": ("1mo", "1d"),
            "3M": ("3mo", "1d"),
            "6M": ("6mo", "1d"),
            "1Y": ("1y", "1d"),
            "3Y": ("3y", "1wk"),
            "5Y": ("5y", "1wk"),
        }
        period, interval = range_map.get(range, ("1y", "1d"))
        stock = yf.Ticker(ticker, session=session)
        hist = stock.history(period=period, interval=interval)

        prices = []
        volumes = []
        for date, row in hist.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            prices.append({"time": date_str, "value": round(row["Close"], 2)})
            volumes.append({
                "time": date_str,
                "value": int(row["Volume"]),
                "color": "rgba(0,229,160,0.3)" if row["Close"] >= row["Open"] else "rgba(255,77,109,0.3)"
            })

        return {"prices": prices, "volumes": volumes}
    except Exception as e:
        print(f"❌ History Error: {e}")
        return {"error": str(e)}

@app.get("/compare/{ticker1}/{ticker2}")
def compare_stocks(ticker1: str, ticker2: str):
    try:
        stock1 = yf.Ticker(ticker1, session=session)
        stock2 = yf.Ticker(ticker2, session=session)

        try:
            info1 = stock1.info
        except:
            time.sleep(3)
            info1 = stock1.info

        try:
            info2 = stock2.info
        except:
            time.sleep(3)
            info2 = stock2.info

        def get_info(info, ticker):
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            return {
                "ticker": ticker.upper(),
                "name": info.get("longName") or info.get("shortName") or ticker,
                "price": round(price, 2),
                "market_cap": format_market_cap(info.get("marketCap")),
                "sector": info.get("sector", "N/A"),
                "pe_ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else "N/A",
                "week_high": info.get("fiftyTwoWeekHigh"),
                "week_low": info.get("fiftyTwoWeekLow"),
                "beta": round(info.get("beta"), 2) if info.get("beta") else "N/A",
                "dividend_yield": round(info.get("dividendYield") * 100, 2) if info.get("dividendYield") else "N/A",
            }

        s1 = get_info(info1, ticker1)
        s2 = get_info(info2, ticker2)

        hist1 = stock1.history(period="1y", interval="1d")
        hist2 = stock2.history(period="1y", interval="1d")

        def normalize(hist):
            if hist.empty:
                return []
            base = hist["Close"].iloc[0]
            return [
                {"time": date.strftime("%Y-%m-%d"), "value": round((row["Close"] / base) * 100, 2)}
                for date, row in hist.iterrows()
            ]

        return {
            "stock1": s1,
            "stock2": s2,
            "chart1": normalize(hist1),
            "chart2": normalize(hist2),
        }
    except Exception as e:
        print(f"❌ Compare Error: {e}")
        return {"error": str(e)}

@app.post("/chat")
async def chat(payload: dict = Body(...)):
    try:
        question = payload.get("question", "") or payload.get("message", "")
        context = payload.get("context", "")

        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"""You are FinSight AI — a friendly stock market assistant helping ALL types of investors.

Current stock context: {context}

Rules:
- Explain in simple plain English
- For beginners avoid jargon use analogies
- Keep response to 2-3 sentences max
- Add disclaimer for investment advice
- Use emojis to keep it friendly
- Never recommend specific buy/sell without disclaimer

User question: {question}"""
            }]
        )

        return {"answer": message.content[0].text}

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return {"answer": "🤖 Great question! Ask me about any stock, market trends, or investing concepts!"}
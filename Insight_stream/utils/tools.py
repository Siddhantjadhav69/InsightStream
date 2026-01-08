import os
import json
import yfinance as yf
from langchain.tools import tool
from langchain_community.tools.duckduckgo_search import DuckDuckGoSearchRun
from datetime import datetime, timedelta

# --- 1. WEB RESEARCH TOOL ---
@tool
def search_latest_finance_news(query: str) -> str:
    """
    Searches for the latest financial news, market sentiment, and 
    macroeconomic trends. Use this for qualitative research.
    """
    search = DuckDuckGoSearchRun()
    # We append '2026' to ensure we get current data
    current_year = datetime.now().year
    full_query = f"{query} {current_year}"
    return search.run(full_query)

# --- 2. STOCK DATA TOOL ---
@tool
def fetch_stock_financials(ticker: str) -> str:
    """
    Retrieves real-time stock price, 52-week highs/lows, market cap, 
    and P/E ratio. Use this for quantitative analysis.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extracting key metrics for the Analyst
        metrics = {
            "symbol": ticker,
            "current_price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "dividend_yield": info.get("dividendYield"),
            "revenue_growth": info.get("revenueGrowth"),
            "summary": info.get("longBusinessSummary")[:300] + "..."
        }
        return json.dumps(metrics, indent=2)
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"

# --- 3. HISTORICAL TRENDS TOOL ---
@tool
def get_price_history(ticker: str) -> str:
    """
    Fetches the percentage change in stock price over the last 1, 3, and 6 months.
    Useful for spotting momentum.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            return "No historical data found."
            
        current = hist['Close'].iloc[-1]
        m1 = hist['Close'].iloc[-21] # ~1 month ago
        m3 = hist['Close'].iloc[-63] # ~3 months ago
        
        perf = {
            "1_month_change": f"{((current - m1)/m1)*100:.2f}%",
            "3_month_change": f"{((current - m3)/m3)*100:.2f}%",
            "6_month_perf": f"{((current - hist['Close'].iloc[0])/hist['Close'].iloc[0])*100:.2f}%"
        }
        return json.dumps(perf)
    except Exception as e:
        return f"Historical data error: {str(e)}"

# --- 4. TOOL LIST FOR AGENT EXECUTOR ---
def get_all_tools():
    """Returns a list of all tools to be used by the Agent."""
    return [
        search_latest_finance_news, 
        fetch_stock_financials, 
        get_price_history
    ]
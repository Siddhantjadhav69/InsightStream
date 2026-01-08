import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import textwrap
import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools.duckduckgo_search import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# --- 1. TOOL DEFINITIONS ---
@tool
def get_stock_performance(symbol: str):
    """Fetches real-time stock price and percentage change for the last 30 days."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1mo")
    if hist.empty:
        return "Could not find data for that symbol."
    start_price = hist['Close'].iloc[0]
    end_price = hist['Close'].iloc[-1]
    pct_change = ((end_price - start_price) / start_price) * 100
    return f"The stock {symbol} moved from {start_price:.2f} to {end_price:.2f} ({pct_change:.2f}%) in 30 days."

@tool
def get_company_news(company_name: str):
    """Searches for recent news and sentiment regarding a specific company."""
    search = DuckDuckGoSearchRun()
    return search.run(f"recent financial news and investor sentiment for {company_name} 2026")

# --- 2. AGENT CORE ---
def initialize_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [get_stock_performance, get_company_news]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Financial Analyst. Use tools to provide data-driven insights. "
                   "Be concise, professional, and always cite your reasoning."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. UI & DASHBOARD ---
st.set_page_config(page_title="InsightStream AI", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

def generate_pdf(report_text):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 750, "InsightStream Financial Intelligence Report")
    c.setFont("Helvetica", 12)

    left_margin = 72
    y = 720
    line_height = 14
    max_width = 90  # approx characters per line before wrapping

    # Split into paragraphs to preserve spacing
    paragraphs = report_text.split('\n\n') if report_text else [""]
    for para in paragraphs:
        wrapped_lines = textwrap.wrap(para, width=max_width)
        if not wrapped_lines:
            y -= line_height
        for line in wrapped_lines:
            if y < 72:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = 750
            c.drawString(left_margin, y, line)
            y -= line_height
        # add spacing between paragraphs
        y -= line_height

    c.save()
    buf.seek(0)
    return buf

@st.cache_data(ttl=300)
def get_stock_data(symbol: str, period: str = "6mo"):
    try:
        data = yf.download(symbol, period=period)
        return data
    except Exception as e:
        logging.exception("Error fetching stock data for %s", symbol)
        return None

def main():
    st.sidebar.title("📈 InsightStream")
    st.sidebar.info("Multi-Agent Financial Research System")
    
    tab1, tab2, tab3 = st.tabs(["Market Analysis", "Visualizer", "Reports"])
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- TAB 1: RESEARCH AGENT ---
    with tab1:
        st.subheader("AI Analyst Terminal")
        user_input = st.text_input("Enter a company or ticker to research (e.g., 'Analyze Nvidia's current outlook'):")
        
        if st.button("Run Research"):
            if not user_input or user_input.strip() == "":
                st.warning("Please enter a company or ticker to research.")
            else:
                with st.spinner("Agents are gathering data..."):
                    try:
                        agent_executor = initialize_agent()
                        result = agent_executor.invoke({
                            "input": user_input,
                            "chat_history": st.session_state.chat_history
                        })
                        # Extract output safely from different agent return shapes
                        if isinstance(result, dict):
                            output = result.get("output") or result.get("response") or str(result)
                        else:
                            output = str(result)

                        st.markdown("### Analyst Findings")
                        st.write(output)
                        st.session_state.last_report = output
                        st.session_state.chat_history.append(output)
                    except Exception as e:
                        logging.exception("Agent invocation failed")
                        st.error(f"Agent failed to produce results: {e}")

    # --- TAB 2: DATA VISUALS ---
    with tab2:
        st.subheader("Real-Time Charting")
        symbol = st.text_input("Ticker Symbol:", value="NVDA").upper()
        if symbol:
            data = get_stock_data(symbol, period="6mo")
            if data is None or data.empty:
                st.warning(f"No price data found for {symbol}.")
            else:
                fig = go.Figure(data=[go.Candlestick(x=data.index,
                                open=data['Open'], high=data['High'],
                                low=data['Low'], close=data['Close'])])
                fig.update_layout(title=f"{symbol} Share Price (6 Months)", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True) 

    # --- TAB 3: REPORT EXPORT ---
    with tab3:
        st.subheader("Generate Formal Report")
        if "last_report" in st.session_state:
            st.text_area("Review Report Content", st.session_state.last_report, height=300)
            pdf_data = generate_pdf(st.session_state.last_report)
            st.download_button("Download PDF Report", pdf_data, "Report.pdf", "application/pdf")
        else:
            st.warning("Run research in Tab 1 first.")

if __name__ == "__main__":
    main()
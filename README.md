# 📈 InsightStream: Multi-Agent Financial Research System

**InsightStream** is an advanced AI-powered research platform that utilizes an autonomous multi-agent architecture to perform deep financial analysis. Unlike standard RAG bots, InsightStream combines **real-time web tools**, **quantitative valuation models**, and **qualitative sentiment analysis** to generate professional-grade investment reports.

---

## 🚀 Key Features

* **Multi-Agent Orchestration:** Uses specialized agents (Researcher & Analyst) to handle complex, multi-step tasks.
* **Real-Time Intelligence:** Integrated with YFinance and DuckDuckGo for up-to-the-minute market data.
* **Deterministic Valuation:** Implements a Discounted Cash Flow (DCF) model to calculate stock fair value—avoiding LLM math hallucinations.
* **Structured Outputs:** Leverages Pydantic for consistent, schema-validated JSON data handling.
* **PDF Report Generation:** One-click export of synthesized findings into a formal PDF document.

---

## 🛠️ Technical Stack

* **Orchestration:** LangChain (Agentic Workflow)
* **LLM:** OpenAI GPT-4o (with Function Calling)
* **Frontend:** Streamlit (Real-time Streaming UI)
* **Data Sources:** YFinance API & DuckDuckGo Search
* **Visualization:** Plotly (Interactive Candlestick Charts)

---

## 🏗️ System Architecture



The system operates in a three-stage pipeline:
1.  **Research Phase:** The Research Agent fetches news sentiment and momentum data.
2.  **Analysis Phase:** The Analyst Agent pulls hard financials and runs the DCF valuation.
3.  **Synthesis Phase:** Both outputs are combined into a final verdict (Buy/Hold/Sell) with clear rationale.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/InsightStream.git](https://github.com/yourusername/InsightStream.git)
   cd InsightStream

2. **Set up a Virtual Environment:**

  python -m venv venv
  source venv/bin/activate  # Windows: venv\Scripts\activate 
3. **Install Dependencies:**

  pip install -r requirements.txt
  Configure Environment Variables: Create a .env file in the root directory:

4. **Code snippet**

  OPENAI_API_KEY=your_openai_key_here

5. **Run the Application:**

  streamlit run app.py

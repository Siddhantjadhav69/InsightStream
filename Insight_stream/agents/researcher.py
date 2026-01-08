import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_community.tools.duckduckgo_search import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- 1. DATA MODELS FOR STRUCTURED OUTPUT ---
# This ensures the agent returns a clean JSON-like object, not just a wall of text.
class ResearchFinding(BaseModel):
    summary: str = Field(description="A 3-sentence summary of the company's current status.")
    key_drivers: List[str] = Field(description="List of factors driving the stock price.")
    risks: List[str] = Field(description="Potential risks or negative news.")
    sentiment_score: float = Field(description="A score from 0 (very bearish) to 1 (very bullish).")

# --- 2. THE RESEARCHER CLASS ---
class FinancialResearcher:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.search_tool = DuckDuckGoSearchRun()
        self.parser = PydanticOutputParser(pydantic_object=ResearchFinding)

    def gather_raw_data(self, company_name: str, ticker: str) -> str:
        """Executes multiple searches to gather a broad context."""
        queries = [
            f"{company_name} ({ticker}) financial performance 2025 2026",
            f"{company_name} latest news and investor sentiment",
            f"{ticker} stock price forecast and analyst ratings"
        ]
        
        aggregated_results = ""
        for q in queries:
            aggregated_results += f"\n--- Results for {q} ---\n"
            aggregated_results += self.search_tool.run(q)
            
        return aggregated_results

    def analyze_and_structure(self, company_name: str, raw_data: str) -> ResearchFinding:
        """Processes raw text into a structured ResearchFinding object."""
        
        prompt = PromptTemplate(
            template="""You are a Professional Financial Researcher. 
            Analyze the following raw search data for {company} and extract key insights.
            
            {format_instructions}
            
            RAW DATA:
            {data}
            """,
            input_variables=["company", "data"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        chain = prompt | self.llm | self.parser
        return chain.invoke({"company": company_name, "data": raw_data})

    def run_full_research(self, company_name: str, ticker: str):
        """Main entry point for the researcher agent."""
        print(f"DEBUG: Starting research on {company_name}...")
        raw_info = self.gather_raw_data(company_name, ticker)
        structured_insight = self.analyze_and_structure(company_name, raw_info)
        return structured_insight

# --- 3. STANDALONE TESTING ---
if __name__ == "__main__":
    # This allows you to test the researcher without running the whole Streamlit app
    researcher = FinancialResearcher()
    result = researcher.run_full_research("NVIDIA", "NVDA")
    print(result.json(indent=2))
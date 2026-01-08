import pandas as pd
import numpy as np
from typing import Dict, Any
from pydantic import BaseModel, Field

# --- 1. DATA MODELS ---
class ValuationReport(BaseModel):
    ticker: str
    intrinsic_value: float = Field(description="The calculated fair price of the stock.")
    current_price: float
    upside_downside: float = Field(description="Percentage difference between fair value and current price.")
    recommendation: str = Field(description="BUY, HOLD, or SELL recommendation.")
    rationale: str = Field(description="Brief explanation of the valuation logic.")

# --- 2. THE ANALYST CLASS ---
class FinancialAnalyst:
    def __init__(self, tax_rate: float = 0.21, discount_rate: float = 0.10):
        self.tax_rate = tax_rate
        self.discount_rate = discount_rate # WACC (Weighted Average Cost of Capital)

    def calculate_dcf(self, ticker: str, financial_data: Dict[str, Any]) -> float:
        """
        A simplified 5-year Discounted Cash Flow (DCF) model.
        financial_data should contain: free_cash_flow, growth_rate, terminal_growth
        """
        fcf = financial_data.get('free_cash_flow', 0)
        growth = financial_data.get('growth_rate', 0.05)
        terminal_growth = financial_data.get('terminal_growth', 0.02)
        
        # Project 5 years of future cash flows
        projections = []
        for i in range(1, 6):
            projected_fcf = fcf * ((1 + growth) ** i)
            discounted_fcf = projected_fcf / ((1 + self.discount_rate) ** i)
            projections.append(discounted_fcf)
            
        # Calculate Terminal Value (Gordon Growth Model)
        final_fcf = projections[-1] * (1 + terminal_growth)
        terminal_value = final_fcf / (self.discount_rate - terminal_growth)
        discounted_tv = terminal_value / ((1 + self.discount_rate) ** 5)
        
        intrinsic_value = sum(projections) + discounted_tv
        return intrinsic_value

    def generate_final_verdict(self, ticker: str, current_price: float, research_insight: Any) -> ValuationReport:
        """
        Combines mathematical DCF with sentiment from the Researcher.
        """
        # Mocking financial inputs (in a real app, these come from yfinance)
        mock_fin_data = {
            "free_cash_flow": 1000000000, # $1B
            "growth_rate": 0.12,         # 12%
            "terminal_growth": 0.02      # 2%
        }
        
        # 1. Math: Calculate fair value
        fair_value = self.calculate_dcf(ticker, mock_fin_data) / 10000000 # Normalized to share price
        
        # 2. Logic: Compare current vs fair
        upside = ((fair_value - current_price) / current_price) * 100
        
        # 3. Decision Engine: Factor in Researcher Sentiment
        # If Researcher score is high (>0.7), we are more aggressive
        sentiment_mod = research_insight.sentiment_score
        
        if upside > 15 and sentiment_mod > 0.6:
            rec = "STRONG BUY"
        elif upside > 5:
            rec = "BUY"
        elif upside < -10:
            rec = "SELL"
        else:
            rec = "HOLD"

        return ValuationReport(
            ticker=ticker,
            intrinsic_value=round(fair_value, 2),
            current_price=current_price,
            upside_downside=round(upside, 2),
            recommendation=rec,
            rationale=f"Fair value of ${fair_value:.2f} based on DCF projections. "
                      f"Sentiment modifier ({sentiment_mod:.2f}) applied to final rating."
        )

# --- 3. STANDALONE TESTING ---
if __name__ == "__main__":
    from agents.researcher import ResearchFinding # Assuming file exists
    
    # Mock research input
    mock_insight = ResearchFinding(
        summary="Positive growth in AI sectors.",
        key_drivers=["Strong cloud sales"],
        risks=["Regulatory hurdles"],
        sentiment_score=0.85
    )
    
    analyst = FinancialAnalyst()
    verdict = analyst.generate_final_verdict("NVDA", 120.0, mock_insight)
    print(verdict.json(indent=2))
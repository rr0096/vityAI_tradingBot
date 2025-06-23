from __future__ import annotations

from typing import Any, Dict, List, Tuple, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

# Re-export output models from the agent modules if they already exist
from agents.technical_v_enhanced_fixed import (
    EnhancedTechnicalAnalysisOutput as TechnicalAnalysisOutput
)
from agents.visual_analyst_enhanced import (
    EnhancedVisualChartAnalysisOutput as VisualAnalysisOutput
)
from agents.QABBAValidatorAgent import (
    QABBAAnalysisOutput
)
from agents.risk import RiskAssessment


class SentimentOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    overall_sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(...)
    positive_texts_count: int = Field(..., ge=0)
    negative_texts_count: int = Field(..., ge=0)
    neutral_texts_count: int = Field(..., ge=0)
    reasoning: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    fear_greed_value_used: int = Field(..., ge=0, le=100)
    fear_greed_influence_factor: float = Field(..., ge=0.0, le=1.0)
    avg_data_quality_score: float = Field(..., ge=0.0, le=1.0)
    total_texts_analyzed_by_llm: int = Field(..., ge=0)
    total_texts_fetched_initially: int = Field(..., ge=0)
    top_keywords_found: List[str] = Field(default_factory=list)
    sentiment_trend_short_term: str = "INSUFFICIENT_DATA"


class FinalDecisionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    final_decision: Literal["BUY", "SELL", "HOLD"] = Field(...)
    combined_reasoning: str
    confidence_in_decision: Literal["HIGH", "MEDIUM", "LOW"] = Field(...)
    key_conflicting_signals: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper utilities for tests
# ---------------------------------------------------------------------------

def create_mock_outputs_for_testing() -> Dict[str, BaseModel]:
    """Return a dictionary with mock outputs for all agents."""
    from backtest import (
        get_mock_sentiment_output,
        get_mock_technical_analysis_output,
        get_mock_visual_analysis_output,
        get_mock_qabba_output,
    )

    tech_metrics = {
        "last_price": 100.0,
        "rsi": 50.0,
        "macd_line": 0.0,
        "signal_line": 0.0,
        "atr": 1.0,
        "adx": 20.0,
    }

    return {
        "sentiment": get_mock_sentiment_output(),
        "technical": get_mock_technical_analysis_output(tech_metrics),
        "visual": get_mock_visual_analysis_output(),
        "qabba": get_mock_qabba_output(),
    }


def convert_to_decision_inputs(
    sentiment: SentimentOutput,
    technical: TechnicalAnalysisOutput,
    visual: VisualAnalysisOutput,
    qabba: QABBAAnalysisOutput,
) -> Tuple[SentimentOutput, TechnicalAnalysisOutput, VisualAnalysisOutput, QABBAAnalysisOutput]:
    """Prepare agent outputs for the decision agent.

    Currently this function simply returns the inputs unchanged, but having a
    dedicated conversion step allows future adaptations without changing the
    tests or callers.
    """
    return sentiment, technical, visual, qabba


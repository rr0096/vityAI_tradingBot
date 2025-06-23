# test_enhanced_agents.py
import asyncio
import logging
from agents.sentiment_enhanced import EnhancedSentimentAnalyst
from agents.technical_v_enhanced_fixed import EnhancedTechnicalAnalyst
from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent
from agents.QABBAValidatorAgent import EnhancedQABBAAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agents():
    """Test cada agente individualmente"""
    
    print("\n🧪 TEST 1: Sentiment Agent")
    try:
        sentiment_agent = EnhancedSentimentAnalyst()
        sentiment_agent.refresh()
        result = sentiment_agent.run()
        print(f"✅ Sentiment: {result.overall_sentiment} (Confidence: {result.confidence_score})")
    except Exception as e:
        print(f"❌ Error en Sentiment: {e}")
    
    print("\n🧪 TEST 2: Technical Agent")
    try:
        technical_agent = EnhancedTechnicalAnalyst()
        # Datos de prueba
        metrics = {
            "last_price": 100.0,
            "rsi": 45.0,
            "macd_line": 0.5,
            "signal_line": 0.3,
            "atr": 2.0,
            "adx": 25.0
        }
        sequences = {}
        
        result = technical_agent.run(
            current_tech_metrics=metrics,
            indicator_sequences=sequences,
            sentiment_label="NEUTRAL"
        )
        print(f"✅ Technical: {result.signal} (Confidence: {result.confidence_level})")
    except Exception as e:
        print(f"❌ Error en Technical: {e}")
    
    print("\n🧪 TEST 3: QABBA Agent")
    try:
        qabba_agent = EnhancedQABBAAgent()
        result = qabba_agent.get_qabba_analysis(
            tech_metrics=metrics,
            price_data_sequence=[98, 99, 100, 101, 100]
        )
        print(f"✅ QABBA: {result.qabba_signal} (Confidence: {result.qabba_confidence})")
    except Exception as e:
        print(f"❌ Error en QABBA: {e}")

if __name__ == "__main__":
    asyncio.run(test_agents())

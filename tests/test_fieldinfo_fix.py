#!/usr/bin/env python3
"""
Test rápido para verificar que el error FieldInfo se haya corregido
"""

def test_sentiment_agent_fieldinfo_fix():
    """Test que el error FieldInfo esté corregido"""
    try:
        print("🧪 Testing FieldInfo fix...")
        
        from agents.sentiment_enhanced import EnhancedSentimentAnalyst
        agent = EnhancedSentimentAnalyst()
        print("✅ Agent created successfully")
        
        # Test the attributes that were causing problems
        print(f"✅ max_texts_per_source: {agent.max_texts_per_source}")
        print(f"✅ min_text_quality_threshold: {agent.min_text_quality_threshold}")
        
        # Test the comparison that was failing
        test_count = 5
        comparison_result = test_count >= agent.max_texts_per_source
        print(f"✅ Comparison test: {test_count} >= {agent.max_texts_per_source} = {comparison_result}")
        
        print("🎉 FieldInfo error has been fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sentiment_agent_fieldinfo_fix()
    if success:
        print("\n✅ SENTIMENT AGENT FIELDINFO ERROR FIXED!")
    else:
        print("\n❌ Still have issues to resolve")

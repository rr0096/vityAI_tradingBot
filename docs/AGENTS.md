# Fenix Trading Bot - Agent System

## 🤖 Agent Roles

Fenix Trading Bot uses a multi-agent system, where each agent is an expert in a specific domain. The agents collaborate to analyze the market, assess risk, and make trading decisions.

### Main Agents

- **VisualAnalystAgent**: Analyzes chart images (e.g., TradingView screenshots) to detect visual patterns, candlestick formations, and trends that may not be obvious from raw data.
- **TechnicalAnalystAgent**: Processes market data to compute and interpret technical indicators (RSI, MACD, moving averages, etc.).
- **SentimentAgent**: Uses LLMs to analyze news headlines, articles, and social media to determine market sentiment (bullish, bearish, neutral).
- **RiskManagementAgent**: Evaluates the risk of each proposed trade, considering volatility, drawdown, and account exposure. Can veto trades that exceed risk limits.
- **ConsensusAgent**: Aggregates all agent outputs and makes the final decision (Buy, Sell, Hold) using a weighted consensus.

## 🧩 Agent Workflow

1. **Data Collection**: Tools and scrapers gather market, news, and social data.
2. **Analysis**: Each agent processes the data relevant to its specialty.
3. **Risk Assessment**: The Risk Manager checks if the trade is safe.
4. **Consensus**: The Consensus Agent decides the final action.
5. **Execution**: The system executes the trade and logs the outcome.

## 🛡️ Risk Management

- Circuit breakers: Stop trading after excessive losses or drawdown.
- Dynamic position sizing: Adjusts trade size based on volatility and account balance.
- Daily loss and trade limits.

---

**See [ARCHITECTURE.md](ARCHITECTURE.md) for a full system diagram.**

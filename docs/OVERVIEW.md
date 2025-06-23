# Fenix Trading Bot - Project Overview

## 🚀 What is Fenix Trading Bot?

Fenix Trading Bot is an advanced, production-ready cryptocurrency trading system powered by a crew of collaborative AI agents. It is designed to operate autonomously, analyze the market from multiple perspectives, and execute trades with robust risk management and real-time monitoring.

- **Multi-Agent Architecture:** Each agent specializes in a domain (sentiment, technical, visual, risk, consensus).
- **LLM Integration:** Uses local LLMs (Ollama) for deep analysis and decision-making.
- **Real-Time Monitoring:** Web dashboard, metrics, and alerting system.
- **Backtesting & Paper Trading:** Simulate and validate strategies safely.
- **Modular & Extensible:** Easy to add new agents, tools, or strategies.

---

## 🧠 How Does It Work?

1. **Data Collection:**
   - Market data (prices, volume, indicators)
   - News and articles
   - Social sentiment (Twitter, Reddit)
   - On-chain analytics (Fear & Greed Index)

2. **Agent Analysis:**
   - **Visual Analyst:** Interprets chart images for patterns and trends.
   - **Technical Analyst:** Calculates and analyzes technical indicators.
   - **Sentiment Analyst:** Uses LLMs to evaluate news and social sentiment.

3. **Risk & Consensus:**
   - **Risk Manager:** Evaluates risk, can veto unsafe trades.
   - **Consensus Agent:** Aggregates all agent outputs to make the final decision (Buy, Sell, Hold).

4. **Execution & Monitoring:**
   - Executes trades (live, paper, or backtest mode)
   - Logs all actions and results for transparency and learning
   - Real-time dashboard and alerting

---

## 📦 Project Structure

```
fenixtradingbot/
├── agents/         # AI agents (sentiment, technical, visual, risk, decision)
├── config/         # Configuration and model management
├── models/         # Shared Pydantic models
├── tools/          # Data scrapers and utilities
├── monitoring/     # Metrics and alerting system
├── memory/         # Trade memory and state
├── tests/          # Unit and integration tests
├── docs/           # Documentation
```

---

## 🛠️ Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-user/fenixtradingbot.git
   cd fenixtradingbot
   ```
2. **Set up your environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Configure API keys and settings:**
   - Copy `.env.example` to `.env` and fill in your API keys.
   - Edit `config/config.yaml` for trading parameters.
4. **Run in paper trading mode:**
   ```bash
   python run_paper_trading.py
   ```
5. **Access the dashboard:**
   - Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📚 More Documentation

- [docs/AGENTS.md](docs/AGENTS.md) — Agent roles and logic
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture
- [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) — Usage and configuration
- [docs/MONITORING.md](docs/MONITORING.md) — Monitoring and alerting
- [docs/INSTALL_GUIDE.md](docs/INSTALL_GUIDE.md) — Installation and setup

---

**For questions or contributions, open an issue or pull request!**

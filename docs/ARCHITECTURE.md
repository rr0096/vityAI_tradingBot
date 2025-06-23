# Fenix Trading Bot - System Architecture

## 🏗️ High-Level Architecture

```
+-------------------+
|  Data Collection  |
+-------------------+
          |
          v
+-------------------+
|   Agent Analysis  | <---+-------------------+
+-------------------+     |  Market Data      |
          |               |  News & Social    |
          v               |  Chart Images     |
+-------------------+     +-------------------+
|  Risk Management  |
+-------------------+
          |
          v
+-------------------+
|   Consensus Agent |
+-------------------+
          |
          v
+-------------------+
|   Trade Execution |
+-------------------+
          |
          v
+-------------------+
|   Monitoring &    |
|   Logging         |
+-------------------+
```

## 🔄 Data Flow

1. **Data Collection**: Scrapers and APIs gather market, news, and social data.
2. **Agent Analysis**: Each agent processes the data and produces a signal.
3. **Risk Management**: The risk manager checks if the trade is safe.
4. **Consensus**: The consensus agent makes the final decision.
5. **Execution**: The trade is executed (live, paper, or backtest).
6. **Monitoring**: All actions are logged and visualized in the dashboard.

## ⚙️ Technologies Used

- **Python 3.11+**
- **CrewAI** (multi-agent orchestration)
- **Ollama** (local LLMs)
- **Binance API** (live trading)
- **Pandas, Pandas-TA** (data analysis)
- **Matplotlib, mplfinance** (visualization)
- **Selenium, BeautifulSoup** (scraping)
- **FastAPI, websockets** (dashboard & monitoring)

---

**For more details, see [AGENTS.md](AGENTS.md) and [USAGE_GUIDE.md](USAGE_GUIDE.md).**

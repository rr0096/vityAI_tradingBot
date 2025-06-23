# Fenix Trading Bot - Usage Guide

## 🚦 Modes of Operation

- **Live Trading:** Trades with real funds on Binance (use with caution).
- **Paper Trading:** Simulates trades in a safe environment (recommended for testing).
- **Backtesting:** Runs strategies on historical data to evaluate performance.

## 🛠️ Configuration

- Edit `config/config.yaml` to set trading pairs, risk parameters, and monitoring options.
- Add your API keys to `.env` (see `.env.example`).

## 🚀 Running the Bot

### Paper Trading (Recommended for Testing)
```bash
python run_paper_trading.py
```

### Live Trading (Real Funds)
```bash
python live_trading.py
```

### Backtesting
```bash
python backtest.py --strategy <strategy_name> --symbol BTC/USDT
```

## 📊 Monitoring

- Access the dashboard at [http://localhost:5000](http://localhost:5000) for real-time metrics, alerts, and trade logs.

## 🧪 Testing

- Run all tests:
```bash
pytest
```

## 📝 Tips

- Always start with paper trading to validate your configuration.
- Never share your API keys.
- Monitor the bot closely, especially during initial deployment.

---

**For more, see [OVERVIEW.md](OVERVIEW.md) and [MONITORING.md](MONITORING.md).**

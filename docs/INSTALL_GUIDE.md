# Fenix Trading Bot - Installation Guide

## 🛠️ Requirements

- Python 3.11+
- Git
- (Recommended) Virtual environment: `venv` or `conda`
- [Ollama](https://ollama.com/) installed for local LLMs

## 📦 Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-user/fenixtradingbot.git
   cd fenixtradingbot
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install Ollama and models:**
   - Download and install Ollama from [https://ollama.com/](https://ollama.com/)
   - Pull recommended models (see `docs/ROADMAP_PHASE_2.md` for details)

5. **Configure API keys:**
   - Copy `.env.example` to `.env` and fill in your API keys (Binance, OpenAI, etc.)
   - Edit `config/config.yaml` for trading pairs, risk, and monitoring settings

6. **Run a quick test:**
   ```bash
   python run_paper_trading.py --duration 5
   ```

---

**For troubleshooting, see [MONITORING.md](MONITORING.md) and [USAGE_GUIDE.md](USAGE_GUIDE.md).**

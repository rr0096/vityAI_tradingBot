import os
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, FUTURE_ORDER_TYPE_MARKET
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
LEVERAGE = int(os.getenv("BINANCE_LEVERAGE", "5"))
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Initialize Binance Client for Futures Testnet
client = Client(API_KEY, API_SECRET, testnet=True)
# Set leverage
client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)


def execute_trade(side: str, quantity: float):
    """
    Execute a market order on Binance Futures Testnet.

    side: 'buy', 'sell', 'short', 'cover'
    quantity: amount in base asset
    """
    try:
        order_side = SIDE_BUY if side == 'buy' else SIDE_SELL
        # For open long or open short, in Binance Futures a SELL order on open position means short
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=order_side,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity
        )
        logger.info(f"Executed {side} order: {order}")
        return order
    except Exception as e:
        logger.error(f"Error executing {side}: {e}")
        return None


if __name__ == '__main__':
    # Example usage: execute a long and a short
    qty = 0.001  # example quantity
    execute_trade('buy', qty)    # open long
    execute_trade('sell', qty)   # open short (market sell creates a short in futures)
    execute_trade('sell', qty)   # cover long or exit long
    execute_trade('buy', qty)    # cover short or exit short

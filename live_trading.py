# live_trading.py
from __future__ import annotations

import sys
import os
# Add project root to sys.path to allow imports from sibling directories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Deque as TypingDeque, Literal, Tuple
import math
from collections import deque
from datetime import datetime, timezone, timedelta

import websockets
from dotenv import load_dotenv
from binance import Client, enums as binance_enums
from binance.exceptions import BinanceAPIException, BinanceOrderException
import requests
from fenix_banner import print_fenix_banner, print_farewell_message

# Importar la configuración cargada
from config.config_loader import APP_CONFIG # Importa la instancia APP_CONFIG

from memory.trade_memory import TradeMemory

# Importar sistema de monitoreo
try:
    from monitoring.metrics_collector import metrics_collector, TradeMetrics
    from monitoring.alerts import alert_manager
    from monitoring.dashboard import dashboard
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

from tools.technical_tools import (
    add_kline, get_current_indicators, get_indicator_sequences,
    close_buf, high_buf, low_buf, vol_buf,
    # MAXLEN y MIN_CANDLES_FOR_RELIABLE_CALC ahora se pueden obtener de APP_CONFIG si se desea
    # o mantenerlos en technical_tools.py si son intrínsecos a esa lógica.
    # Por ahora, los dejaremos en technical_tools.py y usaremos los de APP_CONFIG aquí.
)

trade_memory = TradeMemory()

# --- Logging Configuration ---
log_file_path = Path("logs/fenix_bot_live.log")
log_file_path.parent.mkdir(parents=True, exist_ok=True)
fenix_logger = logging.getLogger("FenixBotLive")
fenix_logger.setLevel(logging.INFO)
fenix_logger.propagate = False

file_handler = logging.FileHandler(log_file_path)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

if not fenix_logger.handlers:
    fenix_logger.addHandler(file_handler)
    fenix_logger.addHandler(stream_handler)

load_dotenv() # .env todavía puede ser útil para secretos no puestos en YAML

# --- Agent and Tool Imports ---
from agents.sentiment_enhanced import EnhancedSentimentAnalyst, SentimentOutput
from agents.technical_v_enhanced_fixed import EnhancedTechnicalAnalyst, EnhancedTechnicalAnalysisOutput
from agents.visual_analyst_enhanced import EnhancedVisualAnalystAgent, EnhancedVisualChartAnalysisOutput
from agents.decision import EnhancedDecisionAgent, FinalDecisionOutput
from agents.risk import AdvancedRiskManager, RiskAssessment, OrderDetails, PortfolioState, RiskParameters
from agents.QABBAValidatorAgent import EnhancedQABBAAgent, QABBAAnalysisOutput
from agents.trade_guardian import TradeGuardianAgent

# Configurar sistema de monitoreo si está disponible
if MONITORING_AVAILABLE:
    try:
        # Configurar canales de alerta
        from monitoring.alerts import FileLogChannel
        alert_manager.add_channel(FileLogChannel("logs/alerts.jsonl"))
        
        # Iniciar monitoreo del sistema
        metrics_collector.start_system_monitoring(interval_seconds=30.0)
        
        # Iniciar dashboard (opcional)
        try:
            dashboard.start()
        except Exception as e:
            fenix_logger.warning(f"No se pudo iniciar el dashboard: {e}")
        
        fenix_logger.info("Sistema de monitoreo inicializado correctamente")
    except Exception as e:
        fenix_logger.error(f"Error inicializando sistema de monitoreo: {e}")
        MONITORING_AVAILABLE = False

# --- Bot Configuration (ahora desde APP_CONFIG) ---
SYMBOL = APP_CONFIG.trading.symbol
TIMEFRAME = APP_CONFIG.trading.timeframe
WS_URL = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@kline_{TIMEFRAME}"

SENTIMENT_REFRESH_COOLDOWN_SECONDS = APP_CONFIG.trading.sentiment_refresh_cooldown_seconds
TRADE_COOLDOWN_AFTER_CLOSE_SECONDS = APP_CONFIG.trading.trade_cooldown_after_close_seconds
MIN_CANDLES_FOR_BOT_START = APP_CONFIG.trading.min_candles_for_bot_start

# Actualizar MAXLEN y MIN_CANDLES_FOR_RELIABLE_CALC en technical_tools si es necesario
# Esto es un poco más complejo si technical_tools.py es un módulo independiente.
# Una forma es que technical_tools.py también importe APP_CONFIG.
# O, al inicializar, pasar estos valores a technical_tools si tiene funciones de inicialización.
# Por simplicidad, asumiremos que los valores en technical_tools.py son los base,
# y aquí podemos usar los de APP_CONFIG para la lógica de este archivo.
TT_MAXLEN_CONFIG = APP_CONFIG.technical_tools.maxlen_buffer
TT_MIN_CANDLES_CONFIG = APP_CONFIG.technical_tools.min_candles_for_reliable_calc


TRADE_LOG_PERFORMANCE_PATH = Path("logs/fenix_tradelog_performance_live.jsonl")
TRADE_LOG_PERFORMANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
PORTFOLIO_STATE_FILE = Path("logs/portfolio_state.json")
PORTFOLIO_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

# API Keys desde APP_CONFIG (que a su vez los carga desde ENV si no están en YAML)
API_KEY = APP_CONFIG.binance.api_key if APP_CONFIG.binance.api_key else None
API_SECRET = APP_CONFIG.binance.api_secret if APP_CONFIG.binance.api_secret else None
USE_TESTNET = APP_CONFIG.trading.use_testnet

if not API_KEY or not API_SECRET:
    fenix_logger.critical("CRITICAL: BINANCE_API_KEY and BINANCE_API_SECRET must be configured in .env file or config.yaml.")
    sys.exit(1)

binance_client: Optional[Client] = None

# Symbol Precision & Limits (se mantienen globales, se obtienen de Binance)
TICK_SIZE: float = 0.01
STEP_SIZE: float = 0.001
MIN_NOTIONAL: float = 5.0
PRICE_PRECISION: int = 2
QTY_PRECISION: int = 3

# --- Agent Initialization (se mantienen globales por ahora) ---
sentiment_agent: Optional[EnhancedSentimentAnalyst] = None
technical_agent: Optional[EnhancedTechnicalAnalyst] = None
visual_agent: Optional[EnhancedVisualAnalystAgent] = None
qabba_agent: Optional[EnhancedQABBAAgent] = None
decision_agent: Optional[EnhancedDecisionAgent] = None
risk_manager: Optional[AdvancedRiskManager] = None

# --- Global State Variables ---
active_position: Optional[Dict[str, Any]] = None
last_trade_closed_time: float = 0.0
last_sentiment_refresh_time: float = 0.0
is_bot_paused_due_to_api_issues: bool = False

def initialize_binance_client():
    global binance_client
    try:
        binance_client = Client(API_KEY, API_SECRET, testnet=USE_TESTNET)
        fenix_logger.info(f"Binance Client initialized. Using Testnet: {USE_TESTNET}")
        binance_client.futures_ping()
        fenix_logger.info("Successfully pinged Binance Futures API.")
        return True
    except BinanceAPIException as bae:
        fenix_logger.critical(f"Binance API Exception during client initialization or ping: {bae}", exc_info=True)
    except requests.exceptions.RequestException as req_err:
        fenix_logger.critical(f"Network error during Binance client initialization or ping: {req_err}", exc_info=True)
    except Exception as e:
        fenix_logger.critical(f"Unexpected error initializing Binance client: {e}", exc_info=True)
    return False

def get_symbol_info_from_binance(symbol_to_check: str) -> bool:
    global TICK_SIZE, STEP_SIZE, MIN_NOTIONAL, PRICE_PRECISION, QTY_PRECISION, binance_client
    if not binance_client:
        fenix_logger.error("Binance client not initialized. Cannot fetch symbol info.")
        return False
    try:
        fenix_logger.info(f"Fetching symbol information for {symbol_to_check} from Binance Futures...")
        exchange_info = binance_client.futures_exchange_info()
        symbol_data = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol_to_check), None)

        if symbol_data:
            api_price_precision = int(symbol_data['pricePrecision'])
            api_qty_precision = int(symbol_data['quantityPrecision'])

            for f_filter in symbol_data['filters']:
                if f_filter['filterType'] == 'PRICE_FILTER':
                    TICK_SIZE = float(f_filter['tickSize'])
                    if TICK_SIZE > 1e-9:
                        if '.' in str(TICK_SIZE).rstrip('0'):
                            PRICE_PRECISION = len(str(TICK_SIZE).split('.')[1].rstrip('0'))
                        else:
                            PRICE_PRECISION = 0
                    else:
                        PRICE_PRECISION = api_price_precision
                        fenix_logger.warning(f"TickSize is zero or invalid ({TICK_SIZE}), using API pricePrecision: {api_price_precision}")
                elif f_filter['filterType'] == 'LOT_SIZE':
                    STEP_SIZE = float(f_filter['stepSize'])
                    if STEP_SIZE > 1e-9:
                        if '.' in str(STEP_SIZE).rstrip('0'):
                            QTY_PRECISION = len(str(STEP_SIZE).split('.')[1].rstrip('0'))
                        else:
                            QTY_PRECISION = 0
                    else:
                        QTY_PRECISION = api_qty_precision
                        fenix_logger.warning(f"StepSize is zero or invalid ({STEP_SIZE}), using API quantityPrecision: {api_qty_precision}")
                elif f_filter['filterType'] == 'MIN_NOTIONAL':
                    MIN_NOTIONAL = float(f_filter.get('notional', f_filter.get('minNotional', MIN_NOTIONAL)))
            
            fenix_logger.info(
                f"Symbol Info for {symbol_to_check}: TickSize={TICK_SIZE}, StepSize={STEP_SIZE}, MinNotional={MIN_NOTIONAL}, "
                f"CalcPricePrecision={PRICE_PRECISION} (API: {api_price_precision}), CalcQtyPrecision={QTY_PRECISION} (API: {api_qty_precision})"
            )
            return True
        else:
            fenix_logger.error(f"No exchange information found for symbol {symbol_to_check} on Binance Futures.")
            return False
    except BinanceAPIException as bae:
        fenix_logger.error(f"Binance API Error fetching symbol info: {bae}", exc_info=True)
    except Exception as e:
        fenix_logger.error(f"Unexpected error fetching symbol info: {e}", exc_info=True)
    return False

def initialize_all_agents_and_risk_manager() -> bool:
    global sentiment_agent, technical_agent, visual_agent, qabba_agent, decision_agent, risk_manager
    try:
        fenix_logger.info("Initializing all agents and risk manager...")
        
        # Pasar configuraciones de APP_CONFIG a los constructores si es necesario
        # Ejemplo para EnhancedSentimentAnalyst si tuviera parámetros configurables:
        # sentiment_agent = EnhancedSentimentAnalyst(
        #     agent_type='sentiment',
        #     max_texts_fetch=APP_CONFIG.agents.sentiment_analyst.max_texts_to_fetch_per_source
        # )
        sentiment_agent = EnhancedSentimentAnalyst(agent_type='sentiment') # Asume que usa config interna o ModelManager
        technical_agent = EnhancedTechnicalAnalyst(
            agent_type='technical',
            sequence_length_llm4fts=20 # Podría venir de APP_CONFIG.agents.technical_analyst.sequence_length_llm4fts
        )
        visual_agent = EnhancedVisualAnalystAgent(
            agent_type='visual',
            save_charts_to_disk=APP_CONFIG.tools.chart_generator.save_charts_to_disk
        )
        qabba_agent = EnhancedQABBAAgent(agent_type='qabba', model_path=os.getenv("QABBA_ML_MODEL_PATH"))
        decision_agent = EnhancedDecisionAgent()
        
        if TICK_SIZE <= 0 or STEP_SIZE <= 0:
             fenix_logger.critical("TICK_SIZE or STEP_SIZE is invalid (zero or negative). Cannot initialize RiskManager properly.")
             return False

        # Usar parámetros de riesgo de APP_CONFIG
        initial_risk_params = RiskParameters(**APP_CONFIG.risk_management.model_dump())
        
        risk_manager = AdvancedRiskManager(
            symbol_tick_size=TICK_SIZE,
            symbol_step_size=STEP_SIZE,
            min_notional=MIN_NOTIONAL,
            initial_risk_params=initial_risk_params,
            portfolio_state_file=str(PORTFOLIO_STATE_FILE)
        )
        risk_manager.reset_daily_stats()
        
        fenix_logger.info("All agents and Risk Manager initialized successfully.")
        return True
    except Exception as e:
        fenix_logger.critical(f"Fatal error during agent or risk manager initialization: {e}", exc_info=True)
        return False

async def get_current_balance_usdt(max_retries: int = 3, initial_delay_seconds: float = 5.0) -> Optional[float]:
    if not binance_client:
        fenix_logger.error("Binance client not initialized. Cannot fetch balance.")
        return None
        
    for attempt in range(max_retries):
        try:
            account_balance_info = binance_client.futures_account_balance()
            for asset_balance in account_balance_info:
                if asset_balance['asset'] == 'USDT':
                    available_balance = float(asset_balance['availableBalance'])
                    fenix_logger.info(f"Current USDT Available Balance: {available_balance:.2f}")
                    return available_balance
            fenix_logger.warning("USDT (or main collateral) balance not found in futures account.")
            return 0.0
        except (BinanceAPIException, requests.exceptions.RequestException, Exception) as e:
            fenix_logger.error(f"Error fetching balance (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_duration = initial_delay_seconds * (2 ** attempt)
                fenix_logger.info(f"Retrying balance fetch in {sleep_duration:.1f} seconds...")
                await asyncio.sleep(sleep_duration)
            else:
                fenix_logger.critical("CRITICAL: Failed to fetch balance after multiple retries.")
                global is_bot_paused_due_to_api_issues
                is_bot_paused_due_to_api_issues = True
                return None
    return None

def format_quantity(quantity: float) -> str:
    if STEP_SIZE <= 1e-9:
        fenix_logger.warning(f"STEP_SIZE is invalid ({STEP_SIZE}). Formatting quantity using QTY_PRECISION ({QTY_PRECISION}) directly.")
        return f"{quantity:.{QTY_PRECISION}f}"
    
    value = math.floor(quantity / STEP_SIZE) * STEP_SIZE
    return f"{value:.{QTY_PRECISION}f}"

def format_price(price: float) -> str:
    if TICK_SIZE <= 1e-9:
        fenix_logger.warning(f"TICK_SIZE is invalid ({TICK_SIZE}). Formatting price using PRICE_PRECISION ({PRICE_PRECISION}) directly.")
        return f"{price:.{PRICE_PRECISION}f}"
        
    value = round(price / TICK_SIZE) * TICK_SIZE
    return f"{value:.{PRICE_PRECISION}f}"

async def get_order_status_with_retries(order_id: Optional[int], symbol: str, max_retries: Optional[int] = None, delay: Optional[float] = None) -> Optional[Dict[str, Any]]:
    if not binance_client or not order_id:
        if not order_id: fenix_logger.debug(f"No order_id provided to get_order_status_with_retries for {symbol}.")
        return None
    
    # Usar valores de config si no se pasan explícitamente
    retries = max_retries if max_retries is not None else APP_CONFIG.trading.order_status_max_retries
    initial_delay = delay if delay is not None else APP_CONFIG.trading.order_status_initial_delay

    for attempt in range(retries):
        try:
            order = binance_client.futures_get_order(symbol=symbol, orderId=order_id)
            if order:
                fenix_logger.debug(f"Order {order_id} status check (Attempt {attempt+1}): {order.get('status')}")
                return order
        except BinanceAPIException as bae:
            fenix_logger.warning(f"API Error getting status for order {order_id} (Attempt {attempt+1}): {bae.code} - {bae.message}")
            if bae.code == -2013:
                fenix_logger.error(f"Order {order_id} does not exist on Binance.")
                return {"status": "NOT_FOUND", "message": "Order does not exist."}
        except Exception as e:
            fenix_logger.error(f"Unexpected error getting status for order {order_id} (Attempt {attempt+1}): {e}")
        
        if attempt < retries - 1:
            await asyncio.sleep(initial_delay * (1.5**attempt))
    fenix_logger.error(f"Failed to get status for order {order_id} after {retries} retries.")
    return None

async def cancel_order_safely(order_id: Optional[int], symbol: str, order_type: str = "Unknown", max_retries: int = 3, delay: float = 0.5) -> bool:
    if not binance_client or not order_id:
        fenix_logger.warning(f"Cannot cancel {order_type} order: No client or orderId ({order_id}).")
        return False
    
    fenix_logger.info(f"Attempting to cancel {order_type} order {order_id} for {symbol}...")
    try:
        cancel_response = binance_client.futures_cancel_order(symbol=symbol, orderId=order_id)
        fenix_logger.info(f"Cancellation request sent for {order_type} order {order_id}. Initial Response: {cancel_response}")

        for i in range(max_retries):
            await asyncio.sleep(delay * (i + 1))
            order_status_details = await get_order_status_with_retries(order_id, symbol, max_retries=1, delay=0.1)
            
            current_status = order_status_details.get('status') if order_status_details else 'Unknown'
            fenix_logger.info(f"Polling cancellation for {order_type} order {order_id}: Attempt {i+1}/{max_retries}, Status: {current_status}")

            if current_status == 'CANCELED':
                fenix_logger.info(f"{order_type} order {order_id} confirmed CANCELED after polling.")
                return True
            elif current_status == 'FILLED':
                fenix_logger.warning(f"{order_type} order {order_id} was FILLED before cancellation could be confirmed.")
                return True
            elif current_status == 'NOT_FOUND':
                fenix_logger.info(f"{order_type} order {order_id} NOT_FOUND, assuming effectively cancelled or error in original placement.")
                return True
        
        fenix_logger.error(f"Failed to confirm cancellation of {order_type} order {order_id} after {max_retries} polling attempts. Last known status: {current_status if 'current_status' in locals() else 'Unknown'}")
        return False

    except BinanceAPIException as bae:
        if bae.code == -2011:
             fenix_logger.info(f"{order_type} order {order_id} likely already processed (filled/expired/cancelled) (API Code -2011). Msg: {bae.message}")
             final_check_status = await get_order_status_with_retries(order_id, symbol, max_retries=1, delay=0.1)
             if final_check_status and final_check_status.get('status') == 'CANCELED':
                 fenix_logger.info(f"{order_type} order {order_id} confirmed CANCELED on final check.")
                 return True
             elif final_check_status and final_check_status.get('status') == 'FILLED':
                 fenix_logger.warning(f"{order_type} order {order_id} found FILLED on final check.")
                 return True
             return True
        else:
            fenix_logger.error(f"Binance API Error cancelling {order_type} order {order_id}: {bae.code} - {bae.message}")
            return False
    except Exception as e:
        fenix_logger.error(f"Unexpected error cancelling {order_type} order {order_id}: {e}")
        return False

async def place_protective_orders(
    symbol: str,
    sltp_order_side: str,
    quantity_str: str,
    sl_trigger_price_str: str,
    tp_limit_price_str: str,
    entry_order_id_for_log: int
) -> Tuple[Optional[int], Optional[int], str]:
    sl_order_id: Optional[int] = None
    tp_order_id: Optional[int] = None
    error_message = ""

    try:
        fenix_logger.info(f"Placing STOP_MARKET (SL): Side={sltp_order_side}, StopPrice={sl_trigger_price_str}, Qty={quantity_str}")
        sl_order_resp_initial = binance_client.futures_create_order( # type: ignore
            symbol=symbol, side=sltp_order_side, type=binance_enums.FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=sl_trigger_price_str, quantity=quantity_str, reduceOnly=True
        )
        sl_order_id = sl_order_resp_initial.get('orderId')
        if not sl_order_id:
            raise BinanceOrderException(message=f"SL order placement failed to return orderId. Response: {sl_order_resp_initial}")
        
        sl_order_status_details = await get_order_status_with_retries(sl_order_id, symbol)
        if not sl_order_status_details or sl_order_status_details.get('status') != 'NEW':
            error_message += f"SL order {sl_order_id} failed confirmation as NEW. Status: {sl_order_status_details.get('status') if sl_order_status_details else 'Unknown'}. "
            fenix_logger.error(error_message)
            sl_order_id = None
        else:
            fenix_logger.info(f"STOP_MARKET (SL) order {sl_order_id} confirmed as {sl_order_status_details.get('status')}.")
    except (BinanceAPIException, BinanceOrderException) as e_sl:
        error_message += f"Error placing/confirming SL: {str(e_sl)}. "
        fenix_logger.error(error_message, exc_info=True)
        sl_order_id = None

    if sl_order_id:
        try:
            fenix_logger.info(f"Placing TAKE_PROFIT_MARKET (TP): Side={sltp_order_side}, StopPrice={tp_limit_price_str}, Qty={quantity_str}")
            tp_order_resp_initial = binance_client.futures_create_order( # type: ignore
                symbol=symbol, side=sltp_order_side, type=binance_enums.FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=tp_limit_price_str, quantity=quantity_str, reduceOnly=True
            )
            tp_order_id = tp_order_resp_initial.get('orderId')
            if not tp_order_id:
                raise BinanceOrderException(message=f"TP order placement failed to return orderId. Response: {tp_order_resp_initial}")

            tp_order_status_details = await get_order_status_with_retries(tp_order_id, symbol)
            if not tp_order_status_details or tp_order_status_details.get('status') != 'NEW':
                error_message += f"TP order {tp_order_id} failed confirmation as NEW. Status: {tp_order_status_details.get('status') if tp_order_status_details else 'Unknown'}. "
                fenix_logger.error(error_message)
                tp_order_id = None
            else:
                fenix_logger.info(f"TAKE_PROFIT_MARKET (TP) order {tp_order_id} confirmed as {tp_order_status_details.get('status')}.")
        except (BinanceAPIException, BinanceOrderException) as e_tp:
            error_message += f"Error placing/confirming TP: {str(e_tp)}. "
            fenix_logger.error(error_message, exc_info=True)
            tp_order_id = None
    else:
        error_message += "TP order not placed because SL order failed. "
        fenix_logger.warning(error_message)

    if not sl_order_id or not tp_order_id:
        error_message = "Failed to place both SL and TP orders successfully. " + error_message
        fenix_logger.error(f"{error_message} Attempting to close entry position {entry_order_id_for_log}.")
        try:
            close_qty_str = format_quantity(float(quantity_str))
            binance_client.futures_create_order( # type: ignore
                symbol=symbol, side=sltp_order_side, type=binance_enums.ORDER_TYPE_MARKET,
                quantity=close_qty_str, reduceOnly=True
            )
            fenix_logger.info(f"Market close order placed for {close_qty_str} {symbol} due to SL/TP setup failure.")
        except Exception as e_close:
            fenix_logger.critical(f"CRITICAL: Failed to close position for {symbol} after SL/TP setup error: {e_close}", exc_info=True)
        return sl_order_id, tp_order_id, error_message

    return sl_order_id, tp_order_id, "" # Success


async def place_binance_order(
    order_params: OrderDetails,
    trade_direction: Literal["BUY", "SELL"],
    symbol: str
) -> Dict[str, Any]:
    global active_position, binance_client
    if not binance_client:
        return {"status": "CLIENT_ERROR", "message": "Binance client not available."}

    entry_order_id: Optional[int] = None
    
    try:
        if not (order_params.position_size_contracts > 0 and
                order_params.stop_loss_price > 0 and
                order_params.take_profit_price > 0):
            return {"status": "INVALID_PARAMS", "message": "RiskManager provided invalid order parameters."}

        requested_qty = order_params.position_size_contracts
        formatted_entry_qty_str = format_quantity(requested_qty)
        
        if float(formatted_entry_qty_str) <= 0:
            return {"status": "ZERO_QUANTITY_ERROR", "message": "Formatted quantity for entry is zero."}

        binance_order_side = binance_enums.SIDE_BUY if trade_direction == "BUY" else binance_enums.SIDE_SELL
        fenix_logger.info(f"Attempting MARKET order: {trade_direction} {formatted_entry_qty_str} {symbol}...")
        
        entry_order_response = binance_client.futures_create_order(
            symbol=symbol, side=binance_order_side, type=binance_enums.ORDER_TYPE_MARKET,
            quantity=formatted_entry_qty_str
        )
        entry_order_id = entry_order_response.get('orderId')
        if not entry_order_id:
            return {"status": "PLACEMENT_ERROR", "message": "Market order failed to get ID.", "response": entry_order_response}

        filled_order_details = await get_order_status_with_retries(entry_order_id, symbol)
        if not filled_order_details or filled_order_details.get('status') != 'FILLED':
            status_msg = filled_order_details.get('status', 'Unknown') if filled_order_details else 'Unknown/Timeout'
            if filled_order_details and filled_order_details.get('status') not in ['CANCELED', 'EXPIRED', 'REJECTED', 'NOT_FOUND']:
                await cancel_order_safely(entry_order_id, symbol, "Entry")
            return {"status": "FILL_ERROR", "message": f"Market order {entry_order_id} not FILLED (Status: {status_msg}).", "details": filled_order_details}

        actual_filled_price = float(filled_order_details.get('avgPrice', 0.0))
        actual_executed_qty = float(filled_order_details.get('executedQty', 0.0))
        if actual_executed_qty <= 0:
            return {"status": "ZERO_FILL_QTY_ERROR", "message": f"Market order {entry_order_id} filled with zero quantity."}
        
        fenix_logger.info(f"MARKET order {entry_order_id} FILLED. Qty: {actual_executed_qty}, AvgPrice: {actual_filled_price:.{PRICE_PRECISION}f}")
        
        formatted_sltp_qty_str = format_quantity(actual_executed_qty)
        sltp_order_side_str = binance_enums.SIDE_SELL if trade_direction == "BUY" else binance_enums.SIDE_BUY
        
        sl_price_final = order_params.stop_loss_price
        tp_price_final = order_params.take_profit_price
        min_sl_dist_from_fill = max(TICK_SIZE * 3, actual_filled_price * 0.0005)

        if trade_direction == "BUY":
            if sl_price_final >= actual_filled_price - min_sl_dist_from_fill:
                sl_price_final = actual_filled_price - min_sl_dist_from_fill
            if tp_price_final <= actual_filled_price + min_sl_dist_from_fill * (order_params.reward_risk_ratio if order_params.reward_risk_ratio > 0 else 1.5) :
                tp_price_final = actual_filled_price + min_sl_dist_from_fill * (order_params.reward_risk_ratio if order_params.reward_risk_ratio > 0 else 1.5)
        else: # SELL
            if sl_price_final <= actual_filled_price + min_sl_dist_from_fill:
                sl_price_final = actual_filled_price + min_sl_dist_from_fill
            if tp_price_final >= actual_filled_price - min_sl_dist_from_fill * (order_params.reward_risk_ratio if order_params.reward_risk_ratio > 0 else 1.5):
                tp_price_final = actual_filled_price - min_sl_dist_from_fill * (order_params.reward_risk_ratio if order_params.reward_risk_ratio > 0 else 1.5)

        formatted_stop_price_sl_str = format_price(sl_price_final)
        formatted_trigger_price_tp_str = format_price(tp_price_final)

        sl_placed_id, tp_placed_id, protective_orders_error_msg = await place_protective_orders(
            symbol=symbol,
            sltp_order_side=sltp_order_side_str,
            quantity_str=formatted_sltp_qty_str,
            sl_trigger_price_str=formatted_stop_price_sl_str,
            tp_limit_price_str=formatted_trigger_price_tp_str,
            entry_order_id_for_log=entry_order_id
        )

        if protective_orders_error_msg or not sl_placed_id or not tp_placed_id:
            fenix_logger.error(f"SL/TP placement failed: {protective_orders_error_msg}. Entry position {entry_order_id} should have been closed by contingency logic in place_protective_orders.")
            return {"status": "SLTP_PLACEMENT_ERROR", "message": protective_orders_error_msg or "Unknown SL/TP error", "entry_details": filled_order_details}

        active_position_details = {
            "symbol": symbol, "side": trade_direction,
            "entry_price_requested": order_params.entry_price,
            "filled_price_actual": actual_filled_price,
            "quantity_requested": requested_qty,
            "quantity_executed": actual_executed_qty,
            "entry_order_id": entry_order_id,
            "sl_order_id": sl_placed_id,
            "tp_order_id": tp_placed_id,
            "sl_price_set": sl_price_final,
            "tp_price_set": tp_price_final,
            "status": "OPEN_WITH_SLTP_CONFIRMED",
            "entry_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "decision_context_at_entry": {}
        }
        active_position = active_position_details.copy()
        fenix_logger.info(f"REAL POSITION OPENED AND SL/TP CONFIRMED: {active_position}")
        return {"status": "PLACED_AND_CONFIRMED_SUCCESSFULLY", **active_position_details}

    except BinanceAPIException as bae:
        fenix_logger.error(f"Binance API Error during order placement for {symbol}: Code={bae.status_code}, Msg='{bae.message}'", exc_info=True)
        if entry_order_id:
             await cancel_order_safely(entry_order_id, symbol, "Entry (Setup Failed)")
        return {"status": "API_ERROR", "message": bae.message, "code": bae.status_code}
    except Exception as e:
        fenix_logger.error(f"Unexpected error placing order for {symbol}: {e}", exc_info=True)
        if entry_order_id:
             await cancel_order_safely(entry_order_id, symbol, "Entry (Setup Failed)")
        return {"status": "GENERAL_ERROR", "message": str(e)}


async def handle_kline_data(kline_payload: Dict[str, Any]):
    global active_position, last_trade_closed_time, last_sentiment_refresh_time, risk_manager, is_bot_paused_due_to_api_issues, technical_agent

    if is_bot_paused_due_to_api_issues:
        if datetime.now(timezone.utc).minute % 15 == 0:
            fenix_logger.info("Attempting to re-check Binance API status while paused...")
            balance_check = await get_current_balance_usdt()
            if balance_check is not None:
                fenix_logger.info("Binance API seems responsive again. Resuming bot operations.")
                is_bot_paused_due_to_api_issues = False
            else:
                fenix_logger.warning("Binance API still unresponsive. Bot remains paused.")
        return

    try:
        kline = kline_payload['k']
        is_candle_closed = kline['x']
        
        if not is_candle_closed:
            return

        fenix_logger.info(f"Processing closed candle for {SYMBOL} at {datetime.fromtimestamp(kline['T']/1000, tz=timezone.utc)}: O={kline['o']} H={kline['h']} L={kline['l']} C={kline['c']} V={kline['v']}")
        current_price_kline = float(kline['c'])
        high_price = float(kline['h'])
        low_price = float(kline['l'])
        volume = float(kline['v'])
        
        if not add_kline(current_price_kline, high_price, low_price, volume):
            fenix_logger.warning("Failed to add kline to buffers, skipping this cycle.")
            return
        
        if len(close_buf) < MIN_CANDLES_FOR_BOT_START:
            fenix_logger.info(f"Waiting for more candles ({len(close_buf)}/{MIN_CANDLES_FOR_BOT_START}) for full analysis...")
            return
        
        current_tech_metrics = get_current_indicators()
        if not current_tech_metrics or 'last_price' not in current_tech_metrics:
            fenix_logger.info("Technical metrics not yet available or incomplete. Waiting for more data.")
            return
        current_tech_metrics['last_price'] = current_price_kline
        
        # --- Active Position Management ---
        if active_position and active_position.get("status", "").startswith("OPEN"):
            pos_side = active_position["side"]
            sl_price_set = active_position.get("sl_price_set", 0.0)
            tp_price_set = active_position.get("tp_price_set", 0.0)
            entry_filled_price = active_position.get("filled_price_actual", 0.0)
            qty_executed = active_position.get("quantity_executed", 0.0)
            
            sl_order_id = active_position.get("sl_order_id")
            tp_order_id = active_position.get("tp_order_id")

            trade_closed_definitively = False
            close_reason = ""
            exit_price = current_price_kline
            pnl = 0.0

            # 1. Check actual order statuses on Binance first
            sl_order_status_details = await get_order_status_with_retries(sl_order_id, SYMBOL) if sl_order_id else None
            tp_order_status_details = await get_order_status_with_retries(tp_order_id, SYMBOL) if tp_order_id else None

            if sl_order_status_details and sl_order_status_details.get('status') == 'FILLED':
                fenix_logger.info(f"Stop Loss order {sl_order_id} confirmed FILLED on exchange.")
                trade_closed_definitively = True
                close_reason = f"STOP_LOSS_FILLED_ON_EXCHANGE (Order ID: {sl_order_id})"
                exit_price = float(sl_order_status_details.get('avgPrice', sl_price_set)) # Use actual fill price if available
                if tp_order_id:
                    fenix_logger.info(f"Attempting to cancel sibling TP order {tp_order_id} as SL filled.")
                    await cancel_order_safely(tp_order_id, SYMBOL, "TP (SL Filled)")
            
            elif tp_order_status_details and tp_order_status_details.get('status') == 'FILLED':
                fenix_logger.info(f"Take Profit order {tp_order_id} confirmed FILLED on exchange.")
                trade_closed_definitively = True
                close_reason = f"TAKE_PROFIT_FILLED_ON_EXCHANGE (Order ID: {tp_order_id})"
                exit_price = float(tp_order_status_details.get('avgPrice', tp_price_set))
                if sl_order_id:
                    fenix_logger.info(f"Attempting to cancel sibling SL order {sl_order_id} as TP filled.")
                    await cancel_order_safely(sl_order_id, SYMBOL, "SL (TP Filled)")
            
            # 2. If not closed by exchange fill, then check kline-based simulation
            # AND ensure the corresponding order is still NEW (not CANCELED or REJECTED by exchange issues)
            if not trade_closed_definitively:
                kline_triggered_closure_attempt = False
                if pos_side == "BUY":
                    if low_price <= sl_price_set and (not sl_order_status_details or sl_order_status_details.get('status') == 'NEW'):
                        fenix_logger.info(f"Kline indicates SL hit for BUY at {sl_price_set}. SL Order ID: {sl_order_id} (Status: {sl_order_status_details.get('status') if sl_order_status_details else 'N/A'}).")
                        kline_triggered_closure_attempt = True
                        exit_price = sl_price_set
                        close_reason = "STOP_LOSS_HIT (SIMULATED by KLINE - CANCELLING TP)"
                        if tp_order_id:
                            if await cancel_order_safely(tp_order_id, SYMBOL, "TP (Simulated SL Hit)"):
                                trade_closed_definitively = True
                            else:
                                fenix_logger.error(f"Failed to cancel TP order {tp_order_id} after simulated SL. Position remains open with risk.")
                    elif high_price >= tp_price_set and (not tp_order_status_details or tp_order_status_details.get('status') == 'NEW'):
                        fenix_logger.info(f"Kline indicates TP hit for BUY at {tp_price_set}. TP Order ID: {tp_order_id} (Status: {tp_order_status_details.get('status') if tp_order_status_details else 'N/A'}).")
                        kline_triggered_closure_attempt = True
                        exit_price = tp_price_set
                        close_reason = "TAKE_PROFIT_HIT (SIMULATED by KLINE - CANCELLING SL)"
                        if sl_order_id:
                            if await cancel_order_safely(sl_order_id, SYMBOL, "SL (Simulated TP Hit)"):
                                trade_closed_definitively = True
                            else:
                                fenix_logger.error(f"Failed to cancel SL order {sl_order_id} after simulated TP. Position remains open with risk.")
                elif pos_side == "SELL":
                    if high_price >= sl_price_set and (not sl_order_status_details or sl_order_status_details.get('status') == 'NEW'):
                        fenix_logger.info(f"Kline indicates SL hit for SELL at {sl_price_set}. SL Order ID: {sl_order_id} (Status: {sl_order_status_details.get('status') if sl_order_status_details else 'N/A'}).")
                        kline_triggered_closure_attempt = True
                        exit_price = sl_price_set
                        close_reason = "STOP_LOSS_HIT (SIMULATED by KLINE - CANCELLING TP)"
                        if tp_order_id:
                            if await cancel_order_safely(tp_order_id, SYMBOL, "TP (Simulated SL Hit)"):
                                trade_closed_definitively = True
                            else:
                                fenix_logger.error(f"Failed to cancel TP order {tp_order_id} after simulated SL. Position remains open with risk.")
                    elif low_price <= tp_price_set and (not tp_order_status_details or tp_order_status_details.get('status') == 'NEW'):
                        fenix_logger.info(f"Kline indicates TP hit for SELL at {tp_price_set}. TP Order ID: {tp_order_id} (Status: {tp_order_status_details.get('status') if tp_order_status_details else 'N/A'}).")
                        kline_triggered_closure_attempt = True
                        exit_price = tp_price_set
                        close_reason = "TAKE_PROFIT_HIT (SIMULATED by KLINE - CANCELLING SL)"
                        if sl_order_id:
                            if await cancel_order_safely(sl_order_id, SYMBOL, "SL (Simulated TP Hit)"):
                                trade_closed_definitively = True
                            else:
                                fenix_logger.error(f"Failed to cancel SL order {sl_order_id} after simulated TP. Position remains open with risk.")
                
                if kline_triggered_closure_attempt and not trade_closed_definitively:
                    fenix_logger.warning(f"Kline indicated SL/TP hit ({close_reason}), but failed to confirm cancellation of the sibling order. Manual check needed. Position NOT considered closed by bot.")
            
            if trade_closed_definitively:
                pnl = (exit_price - entry_filled_price) * qty_executed if pos_side == "BUY" else (entry_filled_price - exit_price) * qty_executed
                fenix_logger.info(
                    f"POSITION CLOSED: {pos_side} for {SYMBOL} by {close_reason} "
                    f"at price ~{exit_price:.{PRICE_PRECISION}f}. PNL Estimado: {pnl:.2f} USDT"
                )
                closed_trade_log = active_position.copy()
                closed_trade_log.update({
                    "status": "CLOSED_CONFIRMED_OR_SIMULATED_CANCELLED", "close_reason": close_reason,
                    "exit_price_actual_or_simulated": exit_price, "pnl_usd": pnl,
                    "exit_timestamp_utc": datetime.now(timezone.utc).isoformat()
                })
                
                trade_memory.save_trade(closed_trade_log)
                if risk_manager: risk_manager.update_portfolio_on_trade_close(
                    pnl_usd=pnl, symbol=SYMBOL, side=pos_side,
                    entry_price=entry_filled_price, exit_price=exit_price, quantity=qty_executed
                )
                
                # --- Registrar métricas de trading ---
                try:
                    record_trade_metrics(
                        order_details=closed_trade_log,
                        final_pnl=pnl,
                        exit_price=exit_price,
                        agent_signals={
                            "sentiment": sentiment_agent.run().model_dump(),
                            "technical": technical_agent.run().model_dump(),
                            "visual": visual_agent.run().model_dump(),
                            "qabba": qabba_agent.get_qabba_analysis(tech_metrics=current_tech_metrics, price_data_sequence=list(close_buf)).model_dump(),
                            "decision": decision_agent.run().model_dump()
                        }
                    )
                except Exception as e_metrics:
                    fenix_logger.error(f"Error registrando métricas de trading: {e_metrics}", exc_info=True)
                
                active_position = None
                last_trade_closed_time = time.monotonic()
                fenix_logger.info(f"Bot available for new trade after cooldown of {TRADE_COOLDOWN_AFTER_CLOSE_SECONDS}s.")
            else:
                fenix_logger.info(f"Monitoring active position: {active_position['side']} {active_position['quantity_executed']} {SYMBOL} @ {active_position['filled_price_actual']:.{PRICE_PRECISION}f}. SL Order: {sl_order_id} ({sl_price_set:.{PRICE_PRECISION}f}), TP Order: {tp_order_id} ({tp_price_set:.{PRICE_PRECISION}f})")
            return

        if (time.monotonic() - last_trade_closed_time) < TRADE_COOLDOWN_AFTER_CLOSE_SECONDS:
            cooldown_remaining = TRADE_COOLDOWN_AFTER_CLOSE_SECONDS - (time.monotonic() - last_trade_closed_time)
            fenix_logger.info(f"In cooldown after closing trade. Waiting {cooldown_remaining:.0f}s...")
            return
        
        if not all([sentiment_agent, technical_agent, visual_agent, qabba_agent, decision_agent, risk_manager]):
            fenix_logger.error("One or more agents are not initialized. Skipping decision cycle.")
            return

        decision_context_log: Dict[str, Any] = {
            "kline_open_time_utc": datetime.fromtimestamp(kline['t']/1000, tz=timezone.utc).isoformat(),
            "kline_close_time_utc": datetime.fromtimestamp(kline['T']/1000, tz=timezone.utc).isoformat(),
            "kline_close_price_at_decision": current_price_kline,
            "raw_tech_metrics_at_decision": current_tech_metrics.copy()
        }
        
        if (time.monotonic() - last_sentiment_refresh_time) > SENTIMENT_REFRESH_COOLDOWN_SECONDS:
            try:
                sentiment_agent.refresh() # type: ignore
                last_sentiment_refresh_time = time.monotonic()
            except Exception as e_sent_refresh:
                fenix_logger.error(f"Error during sentiment_agent.refresh(): {e_sent_refresh}", exc_info=True)
        
        sentiment_result: SentimentOutput = sentiment_agent.run() # type: ignore
        if not isinstance(sentiment_result, SentimentOutput):
            sentiment_result = SentimentOutput(overall_sentiment="NEUTRAL", reasoning="Sentiment agent failed or returned invalid data.", confidence_score=0.1)
        decision_context_log["sentiment_analysis"] = sentiment_result.model_dump()
        fenix_logger.info(f"Sentiment Analysis: {sentiment_result.overall_sentiment} (Conf: {sentiment_result.confidence_score:.2f})")

        indicator_seqs = get_indicator_sequences(sequence_length=getattr(technical_agent, '_sequence_length_for_llm4fts', 20))
        technical_result: EnhancedTechnicalAnalysisOutput = technical_agent.run( # type: ignore
            current_tech_metrics=current_tech_metrics,
            indicator_sequences=indicator_seqs,
            sentiment_label=sentiment_result.overall_sentiment,
            symbol_tick_size=TICK_SIZE
        )
        if not isinstance(technical_result, EnhancedTechnicalAnalysisOutput):
            technical_result = EnhancedTechnicalAnalysisOutput(signal="HOLD", reasoning="Technical agent failed or returned invalid data.", confidence_level="LOW")
        decision_context_log["numerical_technical_analysis"] = technical_result.model_dump()
        fenix_logger.info(f"Technical Analysis: {technical_result.signal} (Conf: {technical_result.confidence_level})")

        visual_result: EnhancedVisualChartAnalysisOutput = visual_agent.run( # type: ignore
            symbol=SYMBOL, timeframe_str=TIMEFRAME,
            close_buf_deque=close_buf, high_buf_deque=high_buf,
            low_buf_deque=low_buf, vol_buf_deque=vol_buf,
            tech_metrics=current_tech_metrics
        )
        if not isinstance(visual_result, EnhancedVisualChartAnalysisOutput):
            visual_result = EnhancedVisualChartAnalysisOutput(overall_visual_assessment="UNCLEAR", reasoning="Visual agent failed or returned invalid data.", chart_timeframe_analyzed=TIMEFRAME, pattern_clarity_score=0.0)
        decision_context_log["visual_technical_analysis"] = visual_result.model_dump()
        clarity_disp = f"{visual_result.pattern_clarity_score:.2f}" if visual_result.pattern_clarity_score is not None else "N/A"
        fenix_logger.info(f"Visual Analysis: {visual_result.overall_visual_assessment} (Clarity: {clarity_disp}, TF: {visual_result.chart_timeframe_analyzed})")

        qabba_result: QABBAAnalysisOutput = qabba_agent.get_qabba_analysis( # type: ignore
            tech_metrics=current_tech_metrics,
            price_data_sequence=list(close_buf)
        )
        if not isinstance(qabba_result, QABBAAnalysisOutput):
            qabba_result = QABBAAnalysisOutput(qabba_signal="NEUTRAL_QABBA", qabba_confidence=0.0, reasoning_short="QABBA agent failed or returned invalid data.")
        decision_context_log["qabba_validation_analysis"] = qabba_result.model_dump()
        fenix_logger.info(f"QABBA Analysis: {qabba_result.qabba_signal} (Conf: {qabba_result.qabba_confidence:.2f})")

        final_decision_output: FinalDecisionOutput = decision_agent.run( # type: ignore
            sentiment_analysis=sentiment_result,
            numerical_technical_analysis=technical_result,
            visual_technical_analysis=visual_result,
            qabba_validation_analysis=qabba_result,
            current_tech_metrics=current_tech_metrics
        )
        if not isinstance(final_decision_output, FinalDecisionOutput):
            final_decision_output = FinalDecisionOutput(final_decision="HOLD", combined_reasoning="Decision agent failed or returned invalid data.", confidence_in_decision="LOW")
        decision_context_log["final_decision_output"] = final_decision_output.model_dump()
        fenix_logger.info(f"Final Decision: {final_decision_output.final_decision} (Conf: {final_decision_output.confidence_in_decision})")

        current_balance = await get_current_balance_usdt()
        if current_balance is None or current_balance <= 0:
            is_bot_paused_due_to_api_issues = True
            return

        fenix_logger.info(f"Balance for RiskManager: ${current_balance:.2f} USDT")
        
        confidence_map = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}
        decision_conf_numeric = confidence_map.get(final_decision_output.confidence_in_decision, 0.5)

        risk_assessment: RiskAssessment = risk_manager.run( # type: ignore
            proposal_decision=final_decision_output.final_decision,
            current_balance=current_balance,
            tech_metrics=current_tech_metrics,
            decision_confidence=decision_conf_numeric,
            active_positions_list=[active_position] if active_position else []
        )
        decision_context_log["risk_assessment"] = risk_assessment.model_dump()
        fenix_logger.info(f"Risk Assessment: {risk_assessment.verdict} - Reason: '{risk_assessment.reason}' (Score: {risk_assessment.risk_score:.1f})")

        if risk_assessment.verdict in ["APPROVE", "APPROVE_REDUCED"] and risk_assessment.order_details:
            if final_decision_output.final_decision != "HOLD":
                fenix_logger.info(f"Trade APPROVED by Risk Manager ({risk_assessment.verdict}). Attempting to place order...")
                decision_context_for_active_pos = decision_context_log.copy()
                
                execution_result = await place_binance_order(
                    order_params=risk_assessment.order_details,
                    trade_direction=final_decision_output.final_decision,
                    symbol=SYMBOL
                )
                
                if execution_result and execution_result.get("status") == "PLACED_AND_CONFIRMED_SUCCESSFULLY":
                    if active_position:
                        active_position["decision_context_at_entry"] = decision_context_for_active_pos
                        fenix_logger.info(f"Order placed and SL/TP confirmed. Active position updated: EntryID {active_position.get('entry_order_id')}")
                    else:
                        fenix_logger.error("CRITICAL: place_binance_order reported success but global active_position is not set!")
                else:
                    status_msg = execution_result.get("message", "Unknown error during placement.") if execution_result else "place_binance_order returned None."
                    error_code = execution_result.get("code", "N/A") if execution_result else "N/A"
                    exec_status = execution_result.get('status', 'ERROR') if execution_result else 'ERROR_NO_EXEC_RESULT'
                    fenix_logger.error(f"Failed to place order or confirm SL/TP on Binance. Status: {exec_status}, Code: {error_code}, Message: {status_msg}")
            else:
                 fenix_logger.info("Final decision was HOLD. No order placed.")
        else:
            fenix_logger.info(f"Trade VETOED by Risk Manager. Verdict: {risk_assessment.verdict}. Reason: {risk_assessment.reason}")
        
        log_entry_performance = {
            "timestamp_decision_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": SYMBOL,
            "kline_close_price_at_decision": current_price_kline,
            "final_decision_from_agent": final_decision_output.final_decision,
            "risk_manager_verdict": risk_assessment.verdict,
            "order_placed_info": active_position.copy() if active_position and active_position.get("status", "").startswith("OPEN") and risk_assessment.verdict != "VETO" else None,
            "decision_context_full": decision_context_log
        }
        try:
            with open(TRADE_LOG_PERFORMANCE_PATH, "a") as f:
                f.write(json.dumps(log_entry_performance, default=str) + "\n")
        except Exception as e_log:
            fenix_logger.error(f"Error writing performance log: {e_log}", exc_info=True)

    except Exception as e_main:
        fenix_logger.error(f"CRITICAL error in handle_kline_data: {e_main}", exc_info=True)


async def connect_binance_ws():
    reconnect_delay_seconds = 5
    max_reconnect_delay_seconds = 300
    
    while True:
        try:
            fenix_logger.info(f"Attempting to connect to Binance WebSocket: {WS_URL}")
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as websocket:
                fenix_logger.info("Successfully connected to Binance WebSocket.")
                reconnect_delay_seconds = 5
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if data.get("e") == "kline":
                            await handle_kline_data(data)
                    except json.JSONDecodeError:
                        fenix_logger.error("Error decoding JSON message from WebSocket.")
                    except Exception as e_msg_proc:
                        fenix_logger.error(f"Error processing WebSocket message: {e_msg_proc}", exc_info=True)
        
        except websockets.exceptions.ConnectionClosedError as e_closed:
            fenix_logger.error(f"WebSocket connection closed (Code: {e_closed.code}, Reason: '{e_closed.reason}').")
        except websockets.exceptions.InvalidStatusCode as e_status:
             fenix_logger.error(f"WebSocket connection failed with invalid status code: {e_status.status_code}.")
        except ConnectionRefusedError:
            fenix_logger.error("WebSocket connection refused by server.")
        except asyncio.TimeoutError:
            fenix_logger.error("WebSocket connection attempt timed out.")
        except Exception as e_ws_general:
            fenix_logger.error(f"General WebSocket error: {e_ws_general}", exc_info=True)
        
        fenix_logger.info(f"Reconnecting in {reconnect_delay_seconds} seconds...")
        await asyncio.sleep(reconnect_delay_seconds)
        reconnect_delay_seconds = min(reconnect_delay_seconds * 2, max_reconnect_delay_seconds)

def preload_historical_data():
    global binance_client
    if not binance_client:
        fenix_logger.error("Binance client not initialized. Cannot preload historical data.")
        return
    try:
        # Usar el valor de configuración para MIN_CANDLES_FOR_BOT_START
        candles_to_load = max(0, APP_CONFIG.trading.min_candles_for_bot_start - len(close_buf))
        if candles_to_load == 0:
            fenix_logger.info(f"Kline buffers sufficiently populated ({len(close_buf)}/{APP_CONFIG.trading.min_candles_for_bot_start}). No further preloading needed.")
            return
        
        fenix_logger.info(f"Preloading up to {candles_to_load} historical klines for {SYMBOL} (Timeframe: {TIMEFRAME})...")
        
        klines_fetched = binance_client.futures_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=min(candles_to_load, 1000))
        
        if not klines_fetched:
            fenix_logger.warning(f"No historical klines received from Binance for {SYMBOL}.")
            return

        for k in klines_fetched:
            add_kline(
                close=float(k[4]),
                high=float(k[2]),
                low=float(k[3]),
                volume=float(k[5])
            )
        fenix_logger.info(f"Preloaded {len(klines_fetched)} klines. Total in buffer for {SYMBOL}: {len(close_buf)} (Target: {APP_CONFIG.trading.min_candles_for_bot_start})")
    except BinanceAPIException as bae:
        fenix_logger.error(f"Binance API Error during historical data preload: {bae}", exc_info=True)
    except Exception as e:
        fenix_logger.error(f"Unexpected error during historical data preload: {e}", exc_info=True)

def record_trade_metrics(order_details: OrderDetails, final_pnl: Optional[float] = None, 
                        exit_price: Optional[float] = None, agent_signals: Optional[Dict[str, Any]] = None):
    """Registra métricas de un trade completado"""
    if not MONITORING_AVAILABLE:
        return
        
    try:
        trade_id = f"{SYMBOL}_{int(datetime.now().timestamp())}"
        
        trade_metrics = TradeMetrics(
            trade_id=trade_id,
            symbol=SYMBOL,
            side=order_details.direction,
            entry_price=order_details.entry_price,
            exit_price=exit_price,
            quantity=order_details.position_size_contracts,
            pnl_usd=final_pnl,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc) if final_pnl is not None else None,
            duration_seconds=None,  # Calcular cuando se cierre
            max_drawdown_pct=None,  # Se actualizará durante el trade
            max_profit_pct=None,    # Se actualizará durante el trade
            agent_signals=agent_signals or {},
            risk_score=order_details.risk_score_at_open
        )
        
        metrics_collector.record_trade(trade_metrics)
        fenix_logger.info(f"Métricas de trade registradas: {trade_id}")
        
    except Exception as e:
        fenix_logger.error(f"Error registrando métricas de trade: {e}")

def update_portfolio_balance(new_balance: float, unrealized_pnl: float = 0.0):
    """Actualiza el balance del portfolio en las métricas"""
    if MONITORING_AVAILABLE:
        try:
            metrics_collector.update_real_time_balance(new_balance, unrealized_pnl)
        except Exception as e:
            fenix_logger.debug(f"Error actualizando balance en métricas: {e}")

def check_alert_conditions():
    """Verifica condiciones de alerta basadas en métricas actuales"""
    if not MONITORING_AVAILABLE:
        return
        
    try:
        # Obtener métricas actuales
        performance_summary = metrics_collector.get_performance_summary()
        
        # Verificar condiciones de alerta
        alert_manager.check_conditions(performance_summary)
        
    except Exception as e:
        fenix_logger.debug(f"Error verificando condiciones de alerta: {e}")

def record_api_call_metrics(endpoint: str, start_time: float, success: bool):
    """Registra métricas de llamadas API"""
    if MONITORING_AVAILABLE:
        try:
            response_time_ms = (time.time() - start_time) * 1000
            metrics_collector.record_api_call(endpoint, response_time_ms, success)
        except Exception as e:
            fenix_logger.debug(f"Error registrando métricas de API: {e}")

async def main():
    fenix_logger.info(f"Starting Fenix Trading Bot v0.9.7 (Config Loaded) for {APP_CONFIG.trading.symbol} on {APP_CONFIG.trading.timeframe}")
    fenix_logger.info(f"Using Testnet: {APP_CONFIG.trading.use_testnet}")
    
    # 🚀 ADVERTENCIA: CIRCUIT BREAKERS DESACTIVADOS 🚀
    # fenix_logger.warning("=" * 80)
    # fenix_logger.warning("🔥 ATENCIÓN: CIRCUIT BREAKERS TEMPORALMENTE DESACTIVADOS 🔥")
    # fenix_logger.warning("🦅 EL BOT ESTÁ EN MODO LIBRE - SIN RESTRICCIONES DE SEGURIDAD")
    # fenix_logger.warning("⚠️  MONITOREAR DE CERCA EL RENDIMIENTO")
    # fenix_logger.warning("📊 No hay límites de pérdida diaria, trades o drawdown")
    # fenix_logger.warning("🔄 Para reactivar: editar agents/risk.py y descomentar validaciones")
    # fenix_logger.warning("=" * 80)

    # Actualizar MAXLEN y MIN_CANDLES_FOR_RELIABLE_CALC en technical_tools.py
    # Esto es una forma de hacerlo si technical_tools.py no puede importar APP_CONFIG directamente.
    # Si technical_tools.py pudiera importar APP_CONFIG, sería más limpio.
    import tools.technical_tools as tt
    tt.MAXLEN = APP_CONFIG.technical_tools.maxlen_buffer
    tt.MIN_CANDLES_FOR_RELIABLE_CALC = APP_CONFIG.technical_tools.min_candles_for_reliable_calc
    # MIN_CANDLES_FOR_CALC también podría necesitar ajuste si es diferente.
    fenix_logger.info(f"Technical tools buffers configured: MAXLEN={tt.MAXLEN}, MIN_RELIABLE={tt.MIN_CANDLES_FOR_RELIABLE_CALC}")


    if not initialize_binance_client():
        fenix_logger.critical("Failed to initialize Binance client. Bot cannot start.")
        return

    if not get_symbol_info_from_binance(APP_CONFIG.trading.symbol): # Usar símbolo de config
        fenix_logger.critical(f"Failed to fetch symbol info for {APP_CONFIG.trading.symbol}. Bot cannot start.")
        return

    if not initialize_all_agents_and_risk_manager():
        fenix_logger.critical("Failed to initialize agents or Risk Manager. Bot cannot start.")
        return
    
    preload_historical_data()
    
    # Usar MIN_CANDLES_FOR_BOT_START de la configuración
    if len(close_buf) < APP_CONFIG.trading.min_candles_for_bot_start:
        fenix_logger.warning(
            f"Insufficient historical data after preload ({len(close_buf)} of {APP_CONFIG.trading.min_candles_for_bot_start} required). "
            f"Bot will wait for more live data from WebSocket stream before starting full analysis cycles."
        )
        
    await connect_binance_ws()

    # --- Ciclo principal adicional para monitoreo y ajustes dinámicos ---
    while True:
        try:
            await asyncio.sleep(10)  # Intervalo de ciclo, ajustar según necesidad

            # 1. Monitorear posiciones abiertas y decidir cerrar o mantener
            if active_position and active_position.get("status", "").startswith("OPEN"):
                monitor_open_positions_with_guardian([active_position], {
                    "technical": technical_agent,
                    "qabba": qabba_agent,
                    "sentiment": sentiment_agent,
                    "visual": visual_agent
                })

            # 2. Actualizar SL/TP dinámicamente según señales actuales de los agentes
            if active_position and active_position.get("status", "").startswith("OPEN"):
                update_dynamic_sl_tp(active_position, {
                    "technical": technical_agent,
                    "qabba": qabba_agent
                })

            # 3. Proactividad: Si la posición lleva mucho tiempo abierta o drawdown > 1%, forzar guardian
            if active_position and active_position.get("status", "").startswith("OPEN"):
                guardian = TradeGuardianAgent()
                tech = technical_agent.last_analysis if hasattr(technical_agent, 'last_analysis') else None
                qabba = qabba_agent.last_analysis if hasattr(qabba_agent, 'last_analysis') else None
                sentiment = sentiment_agent.last_analysis if hasattr(sentiment_agent, 'last_analysis') else None
                visual = visual_agent.last_analysis if hasattr(visual_agent, 'last_analysis') else None
                guardian_decision = guardian.run(active_position, tech, qabba, sentiment, visual)
                if guardian_decision.action == "CLOSE":
                    close_position(active_position, reason=guardian_decision.reasoning)

        except Exception as e_monitoring:
            fenix_logger.error(f"Error en ciclo de monitoreo y ajustes dinámicos: {e_monitoring}", exc_info=True)

def monitor_open_positions_with_guardian(open_positions, agents_dict):
    guardian = TradeGuardianAgent()
    for pos in open_positions:
        # Obtener los análisis más recientes de cada agente
        tech = agents_dict['technical'].last_analysis if hasattr(agents_dict['technical'], 'last_analysis') else None
        qabba = agents_dict['qabba'].last_analysis if hasattr(agents_dict['qabba'], 'last_analysis') else None
        sentiment = agents_dict['sentiment'].last_analysis if hasattr(agents_dict['sentiment'], 'last_analysis') else None
        visual = agents_dict['visual'].last_analysis if hasattr(agents_dict['visual'], 'last_analysis') else None
        guardian_decision = guardian.run(pos, tech, qabba, sentiment, visual)
        # Log SIEMPRE la decisión del guardian, aunque sea HOLD
        fenix_logger.info(f"[TradeGuardian] Decision: {guardian_decision.action} | Confianza: {guardian_decision.confidence:.2f} | Razonamiento: {guardian_decision.reasoning}")
        # Guardar decisión en logs/llm_responses/trade_guardian/
        try:
            import os
            import json
            from datetime import datetime
            log_dir = os.path.join(os.path.dirname(__file__), '../logs/llm_responses/trade_guardian')
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
            log_path = os.path.join(log_dir, f'{ts}.json')
            with open(log_path, 'w') as f:
                json.dump({
                    'timestamp': ts,
                    'position': pos,
                    'decision': guardian_decision.model_dump(),
                    'tech': getattr(tech, 'model_dump', lambda: str(tech))(),
                    'qabba': getattr(qabba, 'model_dump', lambda: str(qabba))(),
                    'sentiment': getattr(sentiment, 'model_dump', lambda: str(sentiment))(),
                    'visual': getattr(visual, 'model_dump', lambda: str(visual))()
                }, f, indent=2)
        except Exception as e:
            fenix_logger.error(f"[TradeGuardian] Error guardando log de decisión: {e}")
        if guardian_decision.action == "CLOSE":
            close_position(pos, reason=guardian_decision.reasoning)
        elif guardian_decision.action == "HOLD":
            continue
        # Si se implementan ADJUST_SL/ADJUST_TP, aquí se puede actualizar el SL/TP

# --- Dinamismo en SL/TP ---
def update_dynamic_sl_tp(position, agents_dict):
    """
    Ajusta el stop loss y take profit de la posición según las señales actuales de los agentes.
    """
    tech = agents_dict['technical'].last_analysis if hasattr(agents_dict['technical'], 'last_analysis') else None
    qabba = agents_dict['qabba'].last_analysis if hasattr(agents_dict['qabba'], 'last_analysis') else None
    # Ejemplo: si el técnico o QABBA sugieren un nuevo SL/TP más conservador, actualizar
    new_sl = None
    new_tp = None
    if tech and hasattr(tech, 'stop_loss_suggestion') and tech.stop_loss_suggestion:
        new_sl = tech.stop_loss_suggestion
    if tech and hasattr(tech, 'price_target') and tech.price_target:
        new_tp = tech.price_target
    # Aquí podrías agregar lógica para consensuar entre agentes
    if new_sl or new_tp:
        modify_position_sl_tp(position, stop_loss=new_sl, take_profit=new_tp)

def close_position(position, reason="Cierre sugerido por TradeGuardian"):
    """
    Lógica para cerrar una posición abierta en el exchange y registrar el cierre.
    """
    symbol = position.get('symbol')
    order_id = position.get('order_id')
    # Aquí deberías cancelar órdenes TP/SL pendientes y enviar orden de cierre (market o limit)
    try:
        # Ejemplo: cancelar TP/SL
        cancel_all_open_orders(symbol)
        # Enviar orden de cierre
        side = 'SELL' if position.get('direction') == 'BUY' else 'BUY'
        qty = position.get('quantity')
        close_order = binance_client.futures_create_order(
            symbol=symbol,
            side=side,
            type=binance_enums.FUTURE_ORDER_TYPE_MARKET,
            quantity=qty
        )
        fenix_logger.info(f"[TradeGuardian] Posición cerrada por guardian. Motivo: {reason}. Orden: {close_order}")
        # Registrar en logs/metrics
        record_trade_close(position, reason, close_order)
    except Exception as e:
        fenix_logger.error(f"Error al cerrar posición por guardian: {e}")


def modify_position_sl_tp(position, stop_loss=None, take_profit=None):
    """
    Lógica para modificar el stop loss y/o take profit de una posición abierta.
    """
    symbol = position.get('symbol')
    qty = position.get('quantity')
    # Cancelar órdenes previas de SL/TP si existen
    cancel_all_open_orders(symbol)
    # Colocar nuevas órdenes SL/TP si se especifican
    try:
        if stop_loss:
            sl_side = 'SELL' if position.get('direction') == 'BUY' else 'BUY'
            binance_client.futures_create_order(
                symbol=symbol,
                side=sl_side,
                type=binance_enums.FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=stop_loss,
                quantity=qty
            )
            fenix_logger.info(f"[TradeGuardian] Nuevo Stop Loss dinámico colocado: {stop_loss}")
        if take_profit:
            tp_side = 'SELL' if position.get('direction') == 'BUY' else 'BUY'
            binance_client.futures_create_order(
                symbol=symbol,
                side=tp_side,
                type=binance_enums.FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=take_profit,
                quantity=qty
            )
            fenix_logger.info(f"[TradeGuardian] Nuevo Take Profit dinámico colocado: {take_profit}")
    except Exception as e:
        fenix_logger.error(f"Error al modificar SL/TP dinámico: {e}")

def cancel_all_open_orders(symbol: str):
    """
    Cancela todas las órdenes abiertas (incluyendo SL/TP) para el símbolo dado en Binance Futures.
    """
    if not binance_client:
        fenix_logger.error("Binance client not initialized. Cannot cancel open orders.")
        return
    try:
        open_orders = binance_client.futures_get_open_orders(symbol=symbol)
        for order in open_orders:
            order_id = order.get('orderId')
            if order_id:
                try:
                    binance_client.futures_cancel_order(symbol=symbol, orderId=order_id)
                    fenix_logger.info(f"Orden cancelada: {order_id} para {symbol}")
                except Exception as e:
                    fenix_logger.error(f"Error cancelando orden {order_id} para {symbol}: {e}")
    except Exception as e:
        fenix_logger.error(f"Error obteniendo/cancelando órdenes abiertas para {symbol}: {e}")

# Adaptar el registro de cierre de trades para usar record_trade_metrics

def record_trade_close(position, reason, close_order):
    """
    Registra el cierre de una posición usando record_trade_metrics.
    """
    try:
        # Extraer detalles relevantes
        order_details = position.get('order_details')
        final_pnl = position.get('final_pnl')
        exit_price = position.get('exit_price')
        agent_signals = position.get('agent_signals')
        record_trade_metrics(order_details, final_pnl=final_pnl, exit_price=exit_price, agent_signals=agent_signals)
        fenix_logger.info(f"Cierre de trade registrado. Motivo: {reason}")
    except Exception as e:
        fenix_logger.error(f"Error registrando cierre de trade: {e}")

def get_last_agent_decision(agent_name: str, symbol: str, max_age_minutes: int = 10):
    """
    Devuelve la última decisión de un agente para un símbolo, solo si es reciente (por ejemplo, <10 minutos).
    """
    recent_trades = trade_memory.get_recent_trades(hours=1)
    now = datetime.now()
    for trade in reversed(recent_trades):
        if trade.get('symbol') == symbol:
            decision_ctx = trade.get('decision_context', {})
            agent_decision = decision_ctx.get(agent_name, {})
            timestamp_str = trade.get('timestamp')
            if timestamp_str:
                try:
                    ts = datetime.fromisoformat(timestamp_str)
                    age_minutes = (now - ts).total_seconds() / 60
                    if age_minutes <= max_age_minutes:
                        return agent_decision, ts
                except Exception:
                    continue
    return None, None

if __name__ == "__main__":
    print_fenix_banner()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        fenix_logger.info("Fenix Bot stopped manually by user (KeyboardInterrupt).")
    except Exception as e_main_run:
        fenix_logger.critical(f"Critical unhandled error in main bot execution: {e_main_run}", exc_info=True)
    finally:
        fenix_logger.info("Fenix Trading Bot shutdown sequence initiated.")
        if risk_manager:
            risk_manager._save_portfolio_state()
            fenix_logger.info("Final portfolio state saved.")
        fenix_logger.info("Fenix Trading Bot has terminated.")

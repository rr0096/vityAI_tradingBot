"""
FenixTradingBot - Order Simulator for Paper Trading
Simulador realista de órdenes que replica el comportamiento de Binance
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"

@dataclass
class SimulatedOrder:
    """Representa una orden simulada con comportamiento realista"""
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reduce_only: bool = False
    commission: float = 0.0
    commission_asset: str = "USDT"

class BinanceOrderSimulator:
    """
    Simulador realista de órdenes de Binance que replica:
    - Latencia de órdenes
    - Slippage realista
    - Partial fills
    - Comisiones
    - Comportamiento de TP/SL
    """
    
    def __init__(self, 
                 commission_rate: float = 0.0004,  # 0.04% comisión
                 slippage_rate: float = 0.0002,   # 0.02% slippage promedio
                 fill_probability: float = 0.98):  # 98% probabilidad de fill para market orders
        
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.fill_probability = fill_probability
        
        # Estado del simulador
        self.active_orders: Dict[str, SimulatedOrder] = {}
        self.order_history: List[SimulatedOrder] = []
        self.current_prices: Dict[str, float] = {}
        
        # Configuración realista
        self.latency_ms = 50  # 50ms latencia promedio
        self.max_slippage = 0.001  # 0.1% slippage máximo
        
        logger.info("🎭 BinanceOrderSimulator initialized")
        logger.info(f"   Commission: {commission_rate*100:.3f}%")
        logger.info(f"   Slippage: {slippage_rate*100:.3f}%")
    
    def update_market_price(self, symbol: str, price: float):
        """Actualiza el precio de mercado para un símbolo"""
        self.current_prices[symbol] = price
        # Procesar órdenes stop que podrían dispararse
        asyncio.create_task(self._process_stop_orders(symbol, price))
    
    async def place_order(self, 
                         symbol: str,
                         side: Literal["BUY", "SELL"],
                         order_type: OrderType,
                         quantity: float,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         reduce_only: bool = False) -> Dict[str, Any]:
        """
        Simula el placement de una orden con latencia y validaciones realistas
        """
        # Simular latencia de red
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Generar ID único para la orden
        order_id = str(uuid.uuid4())
        
        # Validaciones básicas
        if symbol not in self.current_prices:
            return {
                "status": "REJECTED",
                "message": f"Symbol {symbol} not found",
                "code": -1121
            }
        
        if quantity <= 0:
            return {
                "status": "REJECTED", 
                "message": "Invalid quantity",
                "code": -1013
            }
        
        # Crear orden simulada
        order = SimulatedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            reduce_only=reduce_only
        )
        
        self.active_orders[order_id] = order
        
        # Procesar inmediatamente si es market order
        if order_type == OrderType.MARKET:
            await self._process_market_order(order)
        
        # Respuesta similar a Binance
        return {
            "orderId": order_id,
            "symbol": symbol,
            "status": order.status.value,
            "side": side,
            "type": order_type.value,
            "origQty": str(quantity),
            "price": str(price) if price else "0.00000000",
            "stopPrice": str(stop_price) if stop_price else "0.00000000",
            "time": int(order.created_time.timestamp() * 1000)
        }
    
    async def _process_market_order(self, order: SimulatedOrder):
        """Procesa una orden de mercado con slippage realista"""
        current_price = self.current_prices.get(order.symbol, 0)
        if current_price <= 0:
            order.status = OrderStatus.REJECTED
            return
        
        # Simular slippage realista
        import random
        slippage_factor = random.uniform(-self.max_slippage, self.max_slippage)
        if order.side == "BUY":
            # Compra: precio aumenta (slippage negativo para el trader)
            fill_price = current_price * (1 + abs(slippage_factor))
        else:
            # Venta: precio disminuye (slippage negativo para el trader)
            fill_price = current_price * (1 - abs(slippage_factor))
        
        # Simular fill (98% de éxito típico)
        if random.random() < self.fill_probability:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = fill_price
            order.updated_time = datetime.now(timezone.utc)
            
            # Calcular comisión
            notional = order.quantity * fill_price
            order.commission = notional * self.commission_rate
            
            logger.debug(f"Order {order.order_id} FILLED: {order.quantity} {order.symbol} @ ${fill_price:.6f}")
        else:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order {order.order_id} REJECTED (simulation)")
    
    async def _process_stop_orders(self, symbol: str, current_price: float):
        """Procesa órdenes stop/take-profit cuando el precio las dispara"""
        triggered_orders = []
        
        for order_id, order in self.active_orders.items():
            if order.symbol != symbol or order.status != OrderStatus.NEW:
                continue
                
            should_trigger = False
            
            if order.order_type == OrderType.STOP_MARKET:
                if order.stop_price and order.side == "BUY" and current_price >= order.stop_price:
                    should_trigger = True
                elif order.stop_price and order.side == "SELL" and current_price <= order.stop_price:
                    should_trigger = True
                    
            elif order.order_type == OrderType.TAKE_PROFIT_MARKET:
                if order.stop_price and order.side == "BUY" and current_price >= order.stop_price:
                    should_trigger = True
                elif order.stop_price and order.side == "SELL" and current_price <= order.stop_price:
                    should_trigger = True
            
            if should_trigger:
                triggered_orders.append(order)
        
        # Procesar órdenes disparadas
        for order in triggered_orders:
            logger.info(f"Stop order {order.order_id} triggered at ${current_price:.6f}")
            await self._process_market_order(order)
    
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el estado de una orden"""
        if order_id not in self.active_orders:
            # Buscar en historial
            for historical_order in self.order_history:
                if historical_order.order_id == order_id:
                    return self._order_to_dict(historical_order)
            return None
        
        order = self.active_orders[order_id]
        return self._order_to_dict(order)
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancela una orden activa"""
        if order_id not in self.active_orders:
            return {
                "status": "ERROR",
                "message": "Order not found",
                "code": -2011
            }
        
        order = self.active_orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED]:
            return {
                "status": "ERROR", 
                "message": "Order already processed",
                "code": -2011
            }
        
        order.status = OrderStatus.CANCELED
        order.updated_time = datetime.now(timezone.utc)
        
        # Mover a historial
        self.order_history.append(order)
        del self.active_orders[order_id]
        
        return {
            "orderId": order_id,
            "status": "CANCELED",
            "symbol": order.symbol
        }
    
    def _order_to_dict(self, order: SimulatedOrder) -> Dict[str, Any]:
        """Convierte una orden a formato dict similar a Binance"""
        return {
            "orderId": order.order_id,
            "symbol": order.symbol,
            "status": order.status.value,
            "side": order.side,
            "type": order.order_type.value,
            "origQty": str(order.quantity),
            "executedQty": str(order.filled_quantity),
            "price": str(order.price) if order.price else "0.00000000",
            "stopPrice": str(order.stop_price) if order.stop_price else "0.00000000",
            "avgPrice": str(order.avg_fill_price) if order.filled_quantity > 0 else "0.00000000",
            "time": int(order.created_time.timestamp() * 1000),
            "updateTime": int(order.updated_time.timestamp() * 1000),
            "commission": str(order.commission),
            "commissionAsset": order.commission_asset
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Resumen del estado del portfolio simulado"""
        active_count = len(self.active_orders)
        total_orders = len(self.order_history) + active_count
        
        return {
            "active_orders": active_count,
            "total_orders": total_orders,
            "symbols_tracked": list(self.current_prices.keys()),
            "current_prices": self.current_prices.copy()
        }

# Global instance
order_simulator = BinanceOrderSimulator()

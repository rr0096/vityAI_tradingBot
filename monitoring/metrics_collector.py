# monitoring/metrics_collector.py
"""
Colector de métricas de rendimiento para el bot de trading
"""
from __future__ import annotations
import time
import json
import threading
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

import psutil
import numpy as np
from pydantic import BaseModel, Field

@dataclass
class TradeMetrics:
    """Métricas de un trade individual"""
    trade_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl_usd: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    duration_seconds: Optional[float]
    max_drawdown_pct: Optional[float]
    max_profit_pct: Optional[float]
    agent_signals: Dict[str, Any]
    risk_score: float
    
@dataclass
class SystemMetrics:
    """Métricas del sistema"""
    timestamp: datetime
    cpu_usage_pct: float
    memory_usage_pct: float
    memory_usage_mb: float
    network_connections: int
    open_file_descriptors: int
    api_response_times: Dict[str, float]
    websocket_latency_ms: float

@dataclass
class AgentMetrics:
    """Métricas de rendimiento de agentes"""
    agent_name: str
    timestamp: datetime
    execution_time_ms: float
    success: bool
    error_message: Optional[str]
    confidence_score: Optional[float]
    signal: Optional[str]

class MetricsCollector:
    """Colector centralizado de métricas"""
    
    def __init__(self, output_dir: str = "logs/metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Almacenamiento en memoria para análisis rápido
        self.trade_metrics: List[TradeMetrics] = []
        self.system_metrics: deque = deque(maxlen=1000)  # Últimas 1000 muestras
        self.agent_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # Threading para métricas del sistema
        self._system_monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()
        
        # Contadores de rendimiento
        self.performance_counters = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'api_calls': 0,
            'api_errors': 0,
            'agent_failures': defaultdict(int),
            'circuit_breaker_triggers': 0,
        }
        
        # Métricas de tiempo real
        self.real_time_metrics = {
            'current_balance': 0.0,
            'unrealized_pnl': 0.0,
            'daily_pnl': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'active_positions': 0,
        }
        
    def start_system_monitoring(self, interval_seconds: float = 30.0):
        """Inicia el monitoreo continuo del sistema"""
        if self._system_monitor_thread and self._system_monitor_thread.is_alive():
            return
            
        def monitor_loop():
            while not self._stop_monitoring.wait(interval_seconds):
                try:
                    self._collect_system_metrics()
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                    
        self._system_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._system_monitor_thread.start()
        
    def stop_system_monitoring(self):
        """Detiene el monitoreo del sistema"""
        self._stop_monitoring.set()
        if self._system_monitor_thread:
            self._system_monitor_thread.join(timeout=5.0)
            
    def _collect_system_metrics(self):
        """Recolecta métricas del sistema"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage_pct=process.cpu_percent(),
                memory_usage_pct=process.memory_percent(),
                memory_usage_mb=memory_info.rss / (1024 * 1024),
                network_connections=len(process.connections()),
                open_file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0,
                api_response_times={},  # Se actualiza desde otros componentes
                websocket_latency_ms=0.0  # Se actualiza desde el websocket
            )
            
            with self._lock:
                self.system_metrics.append(metrics)
                
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
            
    def record_trade(self, trade: TradeMetrics):
        """Registra métricas de un trade"""
        with self._lock:
            self.trade_metrics.append(trade)
            self.performance_counters['total_trades'] += 1
            
            if trade.pnl_usd is not None:
                if trade.pnl_usd > 0:
                    self.performance_counters['winning_trades'] += 1
                else:
                    self.performance_counters['losing_trades'] += 1
                    
            self._update_real_time_metrics()
            self._save_trade_to_file(trade)
            
    def record_agent_execution(self, agent_name: str, execution_time_ms: float, 
                             success: bool, error_message: Optional[str] = None,
                             confidence_score: Optional[float] = None,
                             signal: Optional[str] = None):
        """Registra métricas de ejecución de agente"""
        metrics = AgentMetrics(
            agent_name=agent_name,
            timestamp=datetime.now(timezone.utc),
            execution_time_ms=execution_time_ms,
            success=success,
            error_message=error_message,
            confidence_score=confidence_score,
            signal=signal
        )
        
        with self._lock:
            self.agent_metrics[agent_name].append(metrics)
            if not success:
                self.performance_counters['agent_failures'][agent_name] += 1
                
    def record_api_call(self, endpoint: str, response_time_ms: float, success: bool):
        """Registra métricas de llamadas API"""
        with self._lock:
            self.performance_counters['api_calls'] += 1
            if not success:
                self.performance_counters['api_errors'] += 1
                
            # Actualizar tiempos de respuesta
            if self.system_metrics:
                latest_metrics = self.system_metrics[-1]
                latest_metrics.api_response_times[endpoint] = response_time_ms
                
    def update_websocket_latency(self, latency_ms: float):
        """Actualiza la latencia del websocket"""
        with self._lock:
            if self.system_metrics:
                self.system_metrics[-1].websocket_latency_ms = latency_ms
                
    def update_real_time_balance(self, balance: float, unrealized_pnl: float = 0.0):
        """Actualiza el balance en tiempo real"""
        with self._lock:
            self.real_time_metrics['current_balance'] = balance
            self.real_time_metrics['unrealized_pnl'] = unrealized_pnl
            
    def _update_real_time_metrics(self):
        """Actualiza métricas en tiempo real basadas en trades"""
        if not self.trade_metrics:
            return
            
        # Calcular PnL total
        total_pnl = sum(t.pnl_usd for t in self.trade_metrics if t.pnl_usd is not None)
        self.real_time_metrics['total_pnl'] = total_pnl
        
        # Calcular win rate
        completed_trades = [t for t in self.trade_metrics if t.pnl_usd is not None]
        if completed_trades:
            winning_trades = sum(1 for t in completed_trades if t.pnl_usd is not None and t.pnl_usd > 0)
            self.real_time_metrics['win_rate'] = winning_trades / len(completed_trades)
            
        # Calcular PnL diario
        today = datetime.now(timezone.utc).date()
        daily_trades = [t for t in self.trade_metrics 
                       if t.exit_time and t.exit_time.date() == today and t.pnl_usd is not None]
        self.real_time_metrics['daily_pnl'] = sum(t.pnl_usd for t in daily_trades if t.pnl_usd is not None)
        
        # Calcular Sharpe ratio simplificado
        if len(completed_trades) > 5:
            returns = [t.pnl_usd for t in completed_trades if t.pnl_usd is not None]
            if returns and len(returns) > 1:
                mean_return = statistics.mean(returns)
                std_return = statistics.stdev(returns)
                if std_return > 0:
                    self.real_time_metrics['sharpe_ratio'] = mean_return / std_return
                
        # Calcular drawdown máximo
        cumulative_pnl = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for trade in completed_trades:
            if trade.pnl_usd is not None:
                cumulative_pnl += trade.pnl_usd
                peak = max(peak, cumulative_pnl)
                drawdown = (peak - cumulative_pnl) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        self.real_time_metrics['max_drawdown'] = max_drawdown
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de rendimiento"""
        with self._lock:
            return {
                'performance_counters': dict(self.performance_counters),
                'real_time_metrics': dict(self.real_time_metrics),
                'system_health': self._get_system_health_score(),
                'agent_health': self._get_agent_health_scores(),
            }
            
    def _get_system_health_score(self) -> float:
        """Calcula score de salud del sistema (0.0 - 1.0)"""
        if not self.system_metrics:
            return 0.5
            
        recent_metrics = list(self.system_metrics)[-10:]  # Últimas 10 muestras
        
        # Factores de salud
        avg_cpu = np.mean([m.cpu_usage_pct for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage_pct for m in recent_metrics])
        avg_latency = np.mean([m.websocket_latency_ms for m in recent_metrics if m.websocket_latency_ms > 0])
        
        # Score basado en umbrales
        cpu_score = max(0.0, 1.0 - (float(avg_cpu) / 80.0))  # Penalizar si CPU > 80%
        memory_score = max(0.0, 1.0 - (float(avg_memory) / 90.0))  # Penalizar si memoria > 90%
        latency_score = max(0.0, 1.0 - (float(avg_latency) / 1000.0)) if avg_latency > 0 else 1.0  # Penalizar si latencia > 1s
        
        return (cpu_score + memory_score + latency_score) / 3
        
    def _get_agent_health_scores(self) -> Dict[str, float]:
        """Calcula scores de salud por agente"""
        scores = {}
        
        for agent_name, metrics_deque in self.agent_metrics.items():
            if not metrics_deque:
                scores[agent_name] = 0.5
                continue
                
            recent_metrics = list(metrics_deque)[-20:]  # Últimas 20 ejecuciones
            
            # Tasa de éxito
            success_rate = np.mean([1 if m.success else 0 for m in recent_metrics])
            
            # Tiempo de ejecución promedio (penalizar si es muy alto)
            avg_exec_time = np.mean([m.execution_time_ms for m in recent_metrics])
            time_score = max(0.0, 1.0 - (float(avg_exec_time) / 30000.0))  # Penalizar si > 30s
            
            scores[agent_name] = (success_rate + time_score) / 2
            
        return scores
        
    def _save_trade_to_file(self, trade: TradeMetrics):
        """Guarda trade a archivo JSON Lines"""
        try:
            trade_file = self.output_dir / "trades.jsonl"
            with open(trade_file, 'a') as f:
                json.dump(asdict(trade), f, default=str)
                f.write('\n')
        except Exception as e:
            print(f"Error saving trade to file: {e}")
            
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """Exporta todas las métricas a archivo JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json"
            
        filepath = self.output_dir / filename
        
        with self._lock:
            data = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'trade_metrics': [asdict(t) for t in self.trade_metrics],
                'system_metrics': [asdict(m) for m in self.system_metrics],
                'agent_metrics': {
                    agent: [asdict(m) for m in metrics]
                    for agent, metrics in self.agent_metrics.items()
                },
                'performance_counters': dict(self.performance_counters),
                'real_time_metrics': dict(self.real_time_metrics),
            }
            
        with open(filepath, 'w') as f:
            json.dump(data, f, default=str, indent=2)
            
        return str(filepath)

# Instancia global del colector
metrics_collector = MetricsCollector()

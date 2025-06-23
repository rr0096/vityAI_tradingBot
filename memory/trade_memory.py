# memory/trade_memory.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TradeMemory:
    def __init__(self, memory_file: str = "logs/trade_memory.json", max_trades: int = 100):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.max_trades = max_trades
        self.trades: List[Dict[str, Any]] = []
        self._load_memory()
    
    def _load_memory(self):
        """Carga trades anteriores del archivo"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.trades = data.get('trades', [])
                    # Mantener solo los últimos max_trades
                    self.trades = self.trades[-self.max_trades:]
            except Exception as e:
                logger.error(f"Error cargando memoria de trades: {e}")
                self.trades = []
    
    def save_trade(self, trade_data: Dict[str, Any]):
        """Guarda un nuevo trade en memoria"""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': trade_data.get('symbol'),
            'side': trade_data.get('side'),
            'entry_price': trade_data.get('entry_price'),
            'exit_price': trade_data.get('exit_price'),
            'pnl': trade_data.get('pnl'),
            'decision_context': {
                'sentiment': trade_data.get('decision_context', {}).get('sentiment_analysis', {}),
                'technical': trade_data.get('decision_context', {}).get('numerical_technical_analysis', {}),
                'visual': trade_data.get('decision_context', {}).get('visual_technical_analysis', {}),
                'qabba': trade_data.get('decision_context', {}).get('qabba_validation_analysis', {}),
                'final_decision': trade_data.get('decision_context', {}).get('final_decision_output', {})
            },
            'risk_assessment': trade_data.get('decision_context', {}).get('risk_assessment', {}),
            'market_conditions': trade_data.get('market_conditions', {})
        }
        
        self.trades.append(trade_record)
        if len(self.trades) > self.max_trades:
            self.trades = self.trades[-self.max_trades:]
        
        self._save_to_file()
    
    def _save_to_file(self):
        """Guarda la memoria en archivo"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump({
                    'last_updated': datetime.now().isoformat(),
                    'trades': self.trades
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando memoria: {e}")
    
    def get_recent_trades(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Obtiene trades recientes"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []
        for trade in self.trades:
            try:
                trade_time = datetime.fromisoformat(trade['timestamp'])
                if trade_time > cutoff:
                    recent.append(trade)
            except:
                continue
        return recent
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Resumen de performance basado en memoria"""
        if not self.trades:
            return {'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0}
        
        wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        total = len(self.trades)
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        
        return {
            'total_trades': total,
            'win_rate': (wins / total) * 100 if total > 0 else 0,
            'avg_pnl': total_pnl / total if total > 0 else 0,
            'total_pnl': total_pnl,
            'recent_trades': self.get_recent_trades(24)
        }
    
    def get_similar_contexts(self, current_context: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Encuentra trades anteriores con contexto similar"""
        similar_trades = []
        
        current_sentiment = current_context.get('sentiment_analysis', {}).get('overall_sentiment')
        current_technical = current_context.get('numerical_technical_analysis', {}).get('signal')
        
        for trade in reversed(self.trades):  # Más recientes primero
            trade_sentiment = trade.get('decision_context', {}).get('sentiment', {}).get('overall_sentiment')
            trade_technical = trade.get('decision_context', {}).get('technical', {}).get('signal')
            
            if trade_sentiment == current_sentiment and trade_technical == current_technical:
                similar_trades.append(trade)
                if len(similar_trades) >= limit:
                    break
        
        return similar_trades

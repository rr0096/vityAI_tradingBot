# tests/test_monitoring.py
"""
Tests para el sistema de monitoreo
"""
import pytest
import tempfile
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

# Importar los módulos a testear
try:
    from monitoring.metrics_collector import MetricsCollector, TradeMetrics, SystemMetrics, AgentMetrics
    from monitoring.alerts import Alert, AlertSeverity, AlertManager, FileLogChannel
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

@pytest.mark.skipif(not MONITORING_AVAILABLE, reason="Sistema de monitoreo no disponible")
class TestMetricsCollector:
    """Tests para MetricsCollector"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = MetricsCollector(output_dir=self.temp_dir)
        
    def test_record_trade_metrics(self):
        """Test registrar métricas de trade"""
        trade_metrics = TradeMetrics(
            trade_id="test_trade_1",
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.01,
            pnl_usd=10.0,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            duration_seconds=300.0,
            max_drawdown_pct=0.02,
            max_profit_pct=0.05,
            agent_signals={"sentiment": "POSITIVE", "technical": "BUY"},
            risk_score=0.7
        )
        
        self.collector.record_trade(trade_metrics)
        
        # Verificar que el trade se registró
        assert len(self.collector.trade_metrics) == 1
        assert self.collector.trade_metrics[0].trade_id == "test_trade_1"
        assert self.collector.performance_counters['total_trades'] == 1
        assert self.collector.performance_counters['winning_trades'] == 1
        
    def test_record_agent_execution(self):
        """Test registrar ejecución de agente"""
        self.collector.record_agent_execution(
            agent_name="sentiment",
            execution_time_ms=150.5,
            success=True,
            confidence_score=0.8,
            signal="POSITIVE"
        )
        
        # Verificar que se registró
        assert "sentiment" in self.collector.agent_metrics
        assert len(self.collector.agent_metrics["sentiment"]) == 1
        
        metrics = self.collector.agent_metrics["sentiment"][0]
        assert metrics.agent_name == "sentiment"
        assert metrics.execution_time_ms == 150.5
        assert metrics.success is True
        
    def test_performance_summary(self):
        """Test obtener resumen de performance"""
        # Agregar algunos datos de prueba
        self.collector.record_api_call("binance_order", 250.0, True)
        self.collector.record_api_call("binance_balance", 300.0, False)
        
        summary = self.collector.get_performance_summary()
        
        assert 'performance_counters' in summary
        assert 'real_time_metrics' in summary
        assert 'system_health' in summary
        assert 'agent_health' in summary
        
        assert summary['performance_counters']['api_calls'] == 2
        assert summary['performance_counters']['api_errors'] == 1

@pytest.mark.skipif(not MONITORING_AVAILABLE, reason="Sistema de monitoreo no disponible")
class TestAlertSystem:
    """Tests para sistema de alertas"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.temp_file.close()
        self.alert_manager = AlertManager()
        self.file_channel = FileLogChannel(self.temp_file.name)
        self.alert_manager.add_channel(self.file_channel)
        
    def test_trigger_alert(self):
        """Test disparar una alerta"""
        alert = Alert(
            id="test_alert_1",
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="This is a test alert",
            timestamp=datetime.now(timezone.utc),
            category="test",
            data={"test_key": "test_value"}
        )
        
        result = self.alert_manager.trigger_alert(alert)
        assert result is True
        assert alert.id in self.alert_manager.active_alerts
        
        # Verificar que se escribió al archivo
        with open(self.temp_file.name, 'r') as f:
            content = f.read()
            assert "test_alert_1" in content
            
    def test_alert_suppression(self):
        """Test supresión de alertas duplicadas"""
        alert1 = Alert(
            id="test_alert_1",
            severity=AlertSeverity.MEDIUM,
            title="Duplicate Test",
            message="First alert",
            timestamp=datetime.now(timezone.utc),
            category="test",
            data={}
        )
        
        alert2 = Alert(
            id="test_alert_2",
            severity=AlertSeverity.MEDIUM,
            title="Duplicate Test",  # Mismo título
            message="Second alert",
            timestamp=datetime.now(timezone.utc),
            category="test",  # Misma categoría
            data={}
        )
        
        # Primera alerta debe pasar
        result1 = self.alert_manager.trigger_alert(alert1)
        assert result1 is True
        
        # Segunda alerta debe ser suprimida (mismo título y categoría)
        result2 = self.alert_manager.trigger_alert(alert2)
        assert result2 is False
        
        # Solo debe haber una alerta activa
        assert len(self.alert_manager.active_alerts) == 1
        
    def test_resolve_alert(self):
        """Test resolver una alerta"""
        alert = Alert(
            id="test_alert_resolve",
            severity=AlertSeverity.LOW,
            title="Resolvable Alert",
            message="This alert will be resolved",
            timestamp=datetime.now(timezone.utc),
            category="test",
            data={}
        )
        
        self.alert_manager.trigger_alert(alert)
        assert alert.id in self.alert_manager.active_alerts
        
        self.alert_manager.resolve_alert(alert.id, "Test resolution")
        assert alert.id not in self.alert_manager.active_alerts
        assert alert.resolved is True
        assert alert.data['resolution_note'] == "Test resolution"

class TestIntegration:
    """Tests de integración para el sistema completo"""
    
    @patch('monitoring.metrics_collector.psutil')
    def test_monitoring_integration(self, mock_psutil):
        """Test integración completa del sistema de monitoreo"""
        # Mock del proceso de psutil
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 25.0
        mock_process.memory_percent.return_value = 45.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024*100)  # 100MB
        mock_process.connections.return_value = []
        mock_process.num_fds.return_value = 50
        mock_psutil.Process.return_value = mock_process
        
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(output_dir=temp_dir)
            alert_manager = AlertManager()
            
            # Agregar canal de alertas
            alert_channel = FileLogChannel(f"{temp_dir}/alerts.jsonl")
            alert_manager.add_channel(alert_channel)
            
            # Simular algunas métricas
            collector.record_api_call("test_endpoint", 100.0, True)
            collector.update_real_time_balance(1000.0, 50.0)
            
            # Obtener resumen
            summary = collector.get_performance_summary()
            assert summary['real_time_metrics']['current_balance'] == 1000.0
            assert summary['performance_counters']['api_calls'] == 1
            
            # Verificar que no hay alertas críticas (sistema sano)
            alert_manager.check_conditions(summary)
            assert len(alert_manager.active_alerts) == 0

def test_config_loading():
    """Test cargar configuración de monitoreo"""
    try:
        from config.monitoring_config import DEFAULT_MONITORING_CONFIG
        
        config = DEFAULT_MONITORING_CONFIG
        assert config.metrics.enabled is True
        assert config.alerts.enabled is True
        assert config.dashboard.enabled is False  # Por defecto deshabilitado
        
        # Verificar umbrales
        assert "system_health_critical" in config.thresholds
        assert config.thresholds["daily_loss_limit"] == -500.0
        
    except ImportError:
        pytest.skip("Configuración de monitoreo no disponible")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

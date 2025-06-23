# monitoring/alerts.py
"""
Sistema de alertas y notificaciones para FenixTradingBot
"""
from __future__ import annotations
import logging
import smtplib
import json
from typing import Dict, Any, List, Literal, Optional, Callable
from datetime import datetime, timezone, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import requests

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Estructura de una alerta"""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    category: str
    data: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class AlertChannel:
    """Clase base para canales de alerta"""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        
    def send_alert(self, alert: Alert) -> bool:
        """Envía una alerta a través del canal"""
        raise NotImplementedError
        
    def is_available(self) -> bool:
        """Verifica si el canal está disponible"""
        return self.enabled

class EmailAlertChannel(AlertChannel):
    """Canal de alertas por email"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, to_emails: List[str], **kwargs):
        super().__init__("email", **kwargs)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.to_emails = to_emails
        
    def send_alert(self, alert: Alert) -> bool:
        """Envía alerta por email"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[FenixBot {alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alerta del Bot de Trading Fenix

Severidad: {alert.severity.value.upper()}
Categoría: {alert.category}
Tiempo: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{alert.message}

Datos adicionales:
{json.dumps(alert.data, indent=2, default=str)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
            logger.info(f"Email alert sent successfully: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert {alert.id}: {e}")
            return False

class DiscordWebhookChannel(AlertChannel):
    """Canal de alertas por Discord webhook"""
    
    def __init__(self, webhook_url: str, **kwargs):
        super().__init__("discord", **kwargs)
        self.webhook_url = webhook_url
        
    def send_alert(self, alert: Alert) -> bool:
        """Envía alerta por Discord"""
        try:
            color_map = {
                AlertSeverity.LOW: 0x00ff00,      # Verde
                AlertSeverity.MEDIUM: 0xffff00,   # Amarillo
                AlertSeverity.HIGH: 0xff8000,     # Naranja
                AlertSeverity.CRITICAL: 0xff0000  # Rojo
            }
            
            embed = {
                "title": f"🤖 FenixBot Alert: {alert.title}",
                "description": alert.message,
                "color": color_map.get(alert.severity, 0x808080),
                "timestamp": alert.timestamp.isoformat(),
                "fields": [
                    {"name": "Severidad", "value": alert.severity.value.upper(), "inline": True},
                    {"name": "Categoría", "value": alert.category, "inline": True},
                ]
            }
            
            # Agregar datos importantes como campos
            for key, value in alert.data.items():
                if isinstance(value, (str, int, float)) and len(str(value)) < 100:
                    embed["fields"].append({
                        "name": key.replace('_', ' ').title(),
                        "value": str(value),
                        "inline": True
                    })
            
            payload = {"embeds": [embed]}
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Discord alert sent successfully: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert {alert.id}: {e}")
            return False

class FileLogChannel(AlertChannel):
    """Canal de alertas por archivo log"""
    
    def __init__(self, log_file: str, **kwargs):
        super().__init__("file", **kwargs)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def send_alert(self, alert: Alert) -> bool:
        """Registra alerta en archivo"""
        try:
            alert_data = {
                "id": alert.id,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "category": alert.category,
                "data": alert.data,
                "resolved": alert.resolved
            }
            
            with open(self.log_file, 'a') as f:
                json.dump(alert_data, f, default=str)
                f.write('\n')
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to log alert {alert.id}: {e}")
            return False

class AlertManager:
    """Gestor central de alertas"""
    
    def __init__(self):
        self.channels: List[AlertChannel] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.rules: List[Callable[[Dict[str, Any]], Optional[Alert]]] = []
        
        # Configuración de supresión para evitar spam
        self.suppression_rules = {
            "duplicate_timeout": 300,  # 5 minutos
            "max_alerts_per_category_per_hour": 10,
        }
        
        self._last_alert_times: Dict[str, datetime] = {}
        self._alert_counts: Dict[str, List[datetime]] = {}
        
    def add_channel(self, channel: AlertChannel):
        """Añade un canal de alertas"""
        self.channels.append(channel)
        logger.info(f"Added alert channel: {channel.name}")
        
    def add_rule(self, rule_func: Callable[[Dict[str, Any]], Optional[Alert]]):
        """Añade una regla de alerta"""
        self.rules.append(rule_func)
        
    def check_conditions(self, metrics: Dict[str, Any]):
        """Evalúa condiciones y dispara alertas si es necesario"""
        for rule in self.rules:
            try:
                alert = rule(metrics)
                if alert:
                    self.trigger_alert(alert)
            except Exception as e:
                logger.error(f"Error evaluating alert rule: {e}")
                
    def trigger_alert(self, alert: Alert) -> bool:
        """Dispara una alerta"""
        # Verificar supresión
        if self._should_suppress_alert(alert):
            logger.debug(f"Alert suppressed: {alert.id}")
            return False
            
        # Registrar alerta
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Enviar a todos los canales activos
        success = False
        for channel in self.channels:
            if channel.is_available():
                try:
                    if channel.send_alert(alert):
                        success = True
                except Exception as e:
                    logger.error(f"Error sending alert through {channel.name}: {e}")
                    
        # Actualizar supresión
        self._update_suppression_tracking(alert)
        
        logger.info(f"Alert triggered: {alert.id} (severity: {alert.severity.value})")
        return success
        
    def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """Resuelve una alerta activa"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now(timezone.utc)
            alert.data['resolution_note'] = resolution_note
            
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert_id}")
            
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Determina si una alerta debe ser suprimida"""
        now = datetime.now(timezone.utc)
        
        # Verificar duplicados recientes
        alert_key = f"{alert.category}_{alert.title}"
        if alert_key in self._last_alert_times:
            last_time = self._last_alert_times[alert_key]
            if (now - last_time).total_seconds() < self.suppression_rules["duplicate_timeout"]:
                return True
                
        # Verificar límite por categoría
        if alert.category not in self._alert_counts:
            self._alert_counts[alert.category] = []
            
        # Limpiar alertas antiguas (más de 1 hora)
        hour_ago = now - timedelta(hours=1)
        self._alert_counts[alert.category] = [
            t for t in self._alert_counts[alert.category] if t > hour_ago
        ]
        
        # Verificar límite
        max_per_hour = self.suppression_rules["max_alerts_per_category_per_hour"]
        if len(self._alert_counts[alert.category]) >= max_per_hour:
            return True
            
        return False
        
    def _update_suppression_tracking(self, alert: Alert):
        """Actualiza el tracking para supresión"""
        now = datetime.now(timezone.utc)
        alert_key = f"{alert.category}_{alert.title}"
        self._last_alert_times[alert_key] = now
        
        if alert.category not in self._alert_counts:
            self._alert_counts[alert.category] = []
        self._alert_counts[alert.category].append(now)
        
    def get_active_alerts(self) -> List[Alert]:
        """Obtiene todas las alertas activas"""
        return list(self.active_alerts.values())
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de alertas"""
        active_by_severity = {}
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            active_by_severity[severity] = active_by_severity.get(severity, 0) + 1
            
        return {
            "active_alerts_count": len(self.active_alerts),
            "active_by_severity": active_by_severity,
            "total_alerts_today": len([
                a for a in self.alert_history 
                if a.timestamp.date() == datetime.now(timezone.utc).date()
            ]),
            "most_common_categories": self._get_most_common_categories(),
        }
        
    def _get_most_common_categories(self) -> List[Dict[str, Any]]:
        """Obtiene las categorías más comunes de alertas"""
        from collections import Counter
        today = datetime.now(timezone.utc).date()
        today_alerts = [a for a in self.alert_history if a.timestamp.date() == today]
        
        category_counts = Counter(a.category for a in today_alerts)
        return [
            {"category": cat, "count": count}
            for cat, count in category_counts.most_common(5)
        ]

# Reglas de alerta predefinidas
def create_system_health_rules() -> List[Callable[[Dict[str, Any]], Optional[Alert]]]:
    """Crea reglas de alerta para salud del sistema"""
    
    def high_cpu_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
        system_health = metrics.get('system_health', 0.5)
        
        if system_health < 0.3:  # Salud crítica
            return Alert(
                id=f"system_health_critical_{int(datetime.now().timestamp())}",
                severity=AlertSeverity.CRITICAL,
                title="Salud del Sistema Crítica",
                message=f"La salud del sistema ha caído a {system_health:.2%}. Revisar CPU, memoria y latencia.",
                timestamp=datetime.now(timezone.utc),
                category="system_health",
                data={"system_health_score": system_health, "metrics": metrics}
            )
        return None
        
    def high_error_rate_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
        performance = metrics.get('performance_counters', {})
        api_calls = performance.get('api_calls', 0)
        api_errors = performance.get('api_errors', 0)
        
        if api_calls > 10 and api_errors / api_calls > 0.2:  # > 20% error rate
            return Alert(
                id=f"high_error_rate_{int(datetime.now().timestamp())}",
                severity=AlertSeverity.HIGH,
                title="Alta Tasa de Errores API",
                message=f"Tasa de errores API: {api_errors}/{api_calls} ({api_errors/api_calls:.1%})",
                timestamp=datetime.now(timezone.utc),
                category="api_errors",
                data={"api_calls": api_calls, "api_errors": api_errors}
            )
        return None
        
    def large_loss_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
        real_time = metrics.get('real_time_metrics', {})
        daily_pnl = real_time.get('daily_pnl', 0)
        
        if daily_pnl < -500:  # Pérdida diaria > $500
            return Alert(
                id=f"large_daily_loss_{int(datetime.now().timestamp())}",
                severity=AlertSeverity.HIGH,
                title="Pérdida Diaria Significativa",
                message=f"Pérdida diaria: ${daily_pnl:.2f}. Revisar estrategia y condiciones del mercado.",
                timestamp=datetime.now(timezone.utc),
                category="trading_performance",
                data={"daily_pnl": daily_pnl, "real_time_metrics": real_time}
            )
        return None
        
    def max_drawdown_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
        real_time = metrics.get('real_time_metrics', {})
        max_drawdown = real_time.get('max_drawdown', 0)
        
        if max_drawdown > 0.15:  # Drawdown > 15%
            return Alert(
                id=f"high_drawdown_{int(datetime.now().timestamp())}",
                severity=AlertSeverity.MEDIUM,
                title="Drawdown Elevado",
                message=f"Drawdown máximo: {max_drawdown:.1%}. Considerar reducir tamaño de posición.",
                timestamp=datetime.now(timezone.utc),
                category="risk_management",
                data={"max_drawdown": max_drawdown}
            )
        return None
        
    return [high_cpu_rule, high_error_rate_rule, large_loss_rule, max_drawdown_rule]

# Instancia global del gestor de alertas
alert_manager = AlertManager()

# Añadir reglas predeterminadas
for rule in create_system_health_rules():
    alert_manager.add_rule(rule)

# monitoring/dashboard.py
"""
Dashboard web básico para monitoreo del bot de trading
"""
from __future__ import annotations
import json
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

try:
    from flask import Flask, render_template, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from .metrics_collector import metrics_collector
from .alerts import alert_manager

class TradingDashboard:
    """Dashboard web para monitoreo del bot"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.host = host
        self.port = port
        self.app: Optional[Flask] = None
        self.server_thread: Optional[threading.Thread] = None
        self._running = False
        
        if not FLASK_AVAILABLE:
            print("Flask no está disponible. Instala con: pip install flask")
            return
            
        self._setup_flask_app()
        
    def _setup_flask_app(self):
        """Configura la aplicación Flask"""
        if not FLASK_AVAILABLE:
            return
            
        self.app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
        
        # Ruta principal
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
            
        # API endpoints
        @self.app.route('/api/metrics')
        def get_metrics():
            return jsonify(metrics_collector.get_performance_summary())
            
        @self.app.route('/api/alerts')
        def get_alerts():
            alerts = alert_manager.get_active_alerts()
            return jsonify({
                'active_alerts': [
                    {
                        'id': alert.id,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'category': alert.category
                    }
                    for alert in alerts
                ],
                'summary': alert_manager.get_alert_summary()
            })
            
        @self.app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
        def resolve_alert(alert_id):
            data = request.get_json() or {}
            resolution_note = data.get('note', '')
            alert_manager.resolve_alert(alert_id, resolution_note)
            return jsonify({'success': True})
            
        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'bot_status': 'running',  # Esto debería venir del bot principal
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'uptime_seconds': 0,  # Calcular desde el inicio del bot
            })
            
    def start(self):
        """Inicia el servidor del dashboard"""
        if not FLASK_AVAILABLE or not self.app:
            print("Dashboard no disponible - Flask no instalado")
            return
            
        if self._running:
            return
            
        def run_server():
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
            
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self._running = True
        
        print(f"Dashboard iniciado en http://{self.host}:{self.port}")
        
    def stop(self):
        """Detiene el servidor del dashboard"""
        self._running = False
        # Flask no tiene un método clean shutdown en modo threading
        # En producción se usaría un servidor WSGI adecuado

# Instancia global del dashboard
dashboard = TradingDashboard()

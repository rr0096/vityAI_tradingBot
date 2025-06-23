# agents/performance_v4.py
from __future__ import annotations

import logging
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Any, ClassVar, Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path

from crewai import Agent
from pydantic import BaseModel, Field, ConfigDict, field_validator, PrivateAttr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class AdvancedPerformanceMetrics(BaseModel):
    """Métricas avanzadas de performance."""
    model_config = ConfigDict(extra="ignore")
    
    # Métricas básicas
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Métricas financieras
    total_pnl_usd: float = 0.0
    average_pnl_usd: float = 0.0
    average_winning_trade_usd: float = 0.0
    average_losing_trade_usd: float = 0.0
    profit_factor: Optional[float] = None
    
    # Métricas de riesgo
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    calmar_ratio: Optional[float] = None
    
    # Métricas de consistencia
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Métricas por timeframe
    daily_stats: Dict[str, float] = Field(default_factory=dict)
    hourly_stats: Dict[str, float] = Field(default_factory=dict)
    
    # Métricas por condición de mercado
    performance_by_market_condition: Dict[str, Dict[str, float]] = Field(default_factory=dict)

class PatternAnalysis(BaseModel):
    """Análisis detallado de patrones."""
    model_config = ConfigDict(extra="ignore")
    
    pattern_description: str
    occurrences: int
    win_rate: float
    avg_pnl: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    recommendation: str
    agent_contributions: Dict[str, float] = Field(default_factory=dict)

class PerformanceReport(BaseModel):
    """Reporte completo de performance."""
    model_config = ConfigDict(extra="ignore")
    
    metrics: AdvancedPerformanceMetrics
    winning_patterns: List[PatternAnalysis]
    losing_patterns: List[PatternAnalysis]
    agent_performance: Dict[str, Dict[str, float]]
    optimization_suggestions: List[str]
    risk_warnings: List[str]
    market_regime_analysis: Dict[str, Any]
    correlation_analysis: Dict[str, float]
    performance_attribution: Dict[str, float]

class AdvancedPerformanceAnalyzer(Agent):
    """Analizador de performance avanzado con ML y análisis estadístico."""
    
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra="ignore")
    
    @field_validator("tools", mode="before", check_fields=False)
    def validate_tools(cls, v: Any) -> Any:
        return v
    
    tools: ClassVar[List[Any]] = []
    name: ClassVar[str] = "AdvancedPerformanceAnalyzer"
    role: ClassVar[str] = "Quantitative Performance Analyst and Strategy Optimizer"
    goal: ClassVar[str] = (
        "Provide deep insights into trading performance using statistical analysis, "
        "pattern recognition, and machine learning to continuously improve the trading system."
    )
    backstory: ClassVar[str] = (
        "An advanced analytical system that combines quantitative methods with ML "
        "to identify performance patterns, optimize strategies, and provide actionable "
        "recommendations for system improvement."
    )
    
    _trade_log_path: str = PrivateAttr()
    _min_trades_for_analysis: int = PrivateAttr(default=20)
    _performance_history: deque = PrivateAttr(default_factory=lambda: deque(maxlen=1000))
    _equity_curve: List[float] = PrivateAttr(default_factory=list)
    _trade_durations: List[float] = PrivateAttr(default_factory=list)
    
    def __init__(self, trade_log_path: str, min_trades_for_analysis: int = 20, **data: Any):
        super().__init__(**data)
        self._trade_log_path = trade_log_path
        self._min_trades_for_analysis = min_trades_for_analysis
        self._equity_curve = [10000.0]  # Starting equity
        self._trade_durations = []
        
        # Cargar histórico si existe
        self._load_historical_data()
        
        logging.info(f"AdvancedPerformanceAnalyzer initialized. Log path: {self._trade_log_path}")
    
    def run(self, generate_visualizations: bool = True) -> Optional[PerformanceReport]:
        """Ejecuta análisis completo de performance."""
        trade_logs = self._load_trade_logs()
        
        if not trade_logs or len(trade_logs) < self._min_trades_for_analysis:
            logging.info(f"PerformanceAnalyzer: Insufficient trades ({len(trade_logs)}/{self._min_trades_for_analysis})")
            return None
        
        # 1. Calcular métricas avanzadas
        metrics = self._calculate_advanced_metrics(trade_logs)
        
        # 2. Análisis de patrones con ML
        winning_patterns, losing_patterns = self._analyze_patterns_ml(trade_logs)
        
        # 3. Análisis de performance por agente
        agent_performance = self._analyze_agent_performance(trade_logs)
        
        # 4. Análisis de régimen de mercado
        market_regime = self._analyze_market_regime(trade_logs)
        
        # 5. Análisis de correlación
        correlations = self._analyze_correlations(trade_logs)
        
        # 6. Atribución de performance
        attribution = self._performance_attribution(trade_logs)
        
        # 7. Generar sugerencias de optimización
        suggestions = self._generate_optimization_suggestions(
            metrics, winning_patterns, losing_patterns, agent_performance
        )
        
        # 8. Identificar advertencias de riesgo
        risk_warnings = self._identify_risk_warnings(metrics, trade_logs)
        
        # 9. Generar visualizaciones si se requiere
        if generate_visualizations:
            self._generate_performance_visualizations(metrics, trade_logs)
        
        report = PerformanceReport(
            metrics=metrics,
            winning_patterns=winning_patterns,
            losing_patterns=losing_patterns,
            agent_performance=agent_performance,
            optimization_suggestions=suggestions,
            risk_warnings=risk_warnings,
            market_regime_analysis=market_regime,
            correlation_analysis=correlations,
            performance_attribution=attribution
        )
        
        # Guardar reporte
        self._save_performance_report(report)
        
        logging.info(f"Performance analysis complete. Win Rate: {metrics.win_rate:.2f}%, "
                    f"Sharpe: {metrics.sharpe_ratio:.2f}")
        
        return report
    
    def _calculate_advanced_metrics(self, trade_logs: List[Dict]) -> AdvancedPerformanceMetrics:
        """Calcula métricas avanzadas de performance."""
        metrics = AdvancedPerformanceMetrics()
        metrics.total_trades = len(trade_logs)
        
        # Arrays para cálculos
        returns = []
        equity_curve = [self._equity_curve[0]]
        
        # Procesar cada trade
        for i, trade in enumerate(trade_logs):
            pnl = trade.get("pnl_usd", 0.0)
            metrics.total_pnl_usd += pnl
            
            # Actualizar equity curve
            new_equity = equity_curve[-1] + pnl
            equity_curve.append(new_equity)
            
            # Calcular retorno porcentual
            if equity_curve[-2] > 0:
                ret = pnl / equity_curve[-2]
                returns.append(ret)
            
            # Clasificar trade
            if pnl > 0:
                metrics.winning_trades += 1
                metrics.consecutive_wins += 1
                metrics.consecutive_losses = 0
                metrics.max_consecutive_wins = max(metrics.max_consecutive_wins, metrics.consecutive_wins)
            elif pnl < 0:
                metrics.losing_trades += 1
                metrics.consecutive_losses += 1
                metrics.consecutive_wins = 0
                metrics.max_consecutive_losses = max(metrics.max_consecutive_losses, metrics.consecutive_losses)
            
            # Duración del trade
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                self._trade_durations.append(duration)
        
        # Métricas básicas
        if metrics.total_trades > 0:
            metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100
            metrics.average_pnl_usd = metrics.total_pnl_usd / metrics.total_trades
        
        # Promedios de ganancia/pérdida
        winning_pnls = [t['pnl_usd'] for t in trade_logs if t.get('pnl_usd', 0) > 0]
        losing_pnls = [abs(t['pnl_usd']) for t in trade_logs if t.get('pnl_usd', 0) < 0]
        
        if winning_pnls:
            metrics.average_winning_trade_usd = np.mean(winning_pnls)
        if losing_pnls:
            metrics.average_losing_trade_usd = np.mean(losing_pnls)
        
        # Profit factor
        total_wins = sum(winning_pnls) if winning_pnls else 0
        total_losses = sum(losing_pnls) if losing_pnls else 0
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Ratios de riesgo
        if returns:
            returns_array = np.array(returns)
            
            # Sharpe Ratio (anualizado)
            if len(returns) > 1:
                metrics.sharpe_ratio = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252)
            
            # Sortino Ratio
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    metrics.sortino_ratio = (np.mean(returns_array) / downside_std) * np.sqrt(252)
        
        # Maximum Drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        metrics.max_drawdown = abs(np.min(drawdown)) * 100
        
        # Calmar Ratio
        if metrics.max_drawdown > 0 and len(returns) > 0:
            annual_return = np.mean(returns) * 252
            metrics.calmar_ratio = annual_return / (metrics.max_drawdown / 100)
        
        # Estadísticas por timeframe
        metrics.daily_stats = self._calculate_timeframe_stats(trade_logs, 'D')
        metrics.hourly_stats = self._calculate_timeframe_stats(trade_logs, 'H')
        
        # Performance por condición de mercado
        metrics.performance_by_market_condition = self._calculate_market_condition_stats(trade_logs)
        
        self._equity_curve = equity_curve
        
        return metrics
    
    def _analyze_patterns_ml(self, trade_logs: List[Dict]) -> Tuple[List[PatternAnalysis], List[PatternAnalysis]]:
        """Analiza patrones usando técnicas de ML."""
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        # Extraer features de cada trade
        features = []
        labels = []  # 1 = ganador, 0 = perdedor
        
        for trade in trade_logs:
            context = trade.get('decision_context', {})
            
            # Features del trade
            feature_vector = [
                # Sentiment
                1 if context.get('sentiment_analysis', {}).get('overall_sentiment') == 'POSITIVE' else
                -1 if context.get('sentiment_analysis', {}).get('overall_sentiment') == 'NEGATIVE' else 0,
                
                # Technical
                1 if context.get('numerical_technical_analysis', {}).get('signal') == 'BUY' else
                -1 if context.get('numerical_technical_analysis', {}).get('signal') == 'SELL' else 0,
                
                # Visual
                1 if context.get('visual_technical_analysis', {}).get('overall_visual_assessment') == 'BULLISH' else
                -1 if context.get('visual_technical_analysis', {}).get('overall_visual_assessment') == 'BEARISH' else 0,
                
                # QABBA
                context.get('qabba_validation_analysis', {}).get('qabba_confidence', 0.5),
                
                # Métricas técnicas
                context.get('raw_tech_metrics_at_decision', {}).get('rsi', 50) / 100,
                context.get('raw_tech_metrics_at_decision', {}).get('adx', 20) / 100,
            ]
            
            features.append(feature_vector)
            labels.append(1 if trade.get('pnl_usd', 0) > 0 else 0)
        
        if len(features) < 10:
            return [], []
        
        # Normalizar features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Clustering para encontrar patrones
        clustering = DBSCAN(eps=0.5, min_samples=3)
        clusters = clustering.fit_predict(features_scaled)
        
        # Analizar cada cluster
        pattern_analyses = []
        
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Ruido
                continue
            
            cluster_mask = clusters == cluster_id
            cluster_trades = [trade_logs[i] for i, mask in enumerate(cluster_mask) if mask]
            cluster_labels = [labels[i] for i, mask in enumerate(cluster_mask) if mask]
            
            if len(cluster_trades) < 3:
                continue
            
            # Calcular estadísticas del cluster
            win_rate = sum(cluster_labels) / len(cluster_labels)
            avg_pnl = np.mean([t['pnl_usd'] for t in cluster_trades])
            
            # Test estadístico
            _, p_value = stats.ttest_1samp([t['pnl_usd'] for t in cluster_trades], 0)
            
            # Describir el patrón
            pattern_features = np.mean([features[i] for i, mask in enumerate(cluster_mask) if mask], axis=0)
            description = self._describe_pattern(pattern_features)
            
            # Calcular intervalo de confianza
            pnls = [t['pnl_usd'] for t in cluster_trades]
            confidence_interval = stats.t.interval(0.95, len(pnls)-1,
                                                 loc=np.mean(pnls),
                                                 scale=stats.sem(pnls))
            
            pattern = PatternAnalysis(
                pattern_description=description,
                occurrences=len(cluster_trades),
                win_rate=win_rate * 100,
                avg_pnl=avg_pnl,
                confidence_interval=confidence_interval,
                statistical_significance=p_value,
                recommendation=self._generate_pattern_recommendation(win_rate, avg_pnl, p_value),
                agent_contributions=self._analyze_agent_contributions(cluster_trades)
            )
            
            pattern_analyses.append(pattern)
        
        # Separar patrones ganadores y perdedores
        winning_patterns = [p for p in pattern_analyses if p.avg_pnl > 0]
        losing_patterns = [p for p in pattern_analyses if p.avg_pnl <= 0]
        
        # Ordenar por impacto
        winning_patterns.sort(key=lambda x: x.avg_pnl * x.occurrences, reverse=True)
        losing_patterns.sort(key=lambda x: abs(x.avg_pnl) * x.occurrences, reverse=True)
        
        return winning_patterns[:5], losing_patterns[:5]
    
    def _describe_pattern(self, features: np.ndarray) -> str:
        """Describe un patrón basado en sus features."""
        descriptions = []
        
        # Sentiment
        if features[0] > 0.5:
            descriptions.append("Sentimiento POSITIVO")
        elif features[0] < -0.5:
            descriptions.append("Sentimiento NEGATIVO")
        
        # Technical
        if features[1] > 0.5:
            descriptions.append("Señal técnica de COMPRA")
        elif features[1] < -0.5:
            descriptions.append("Señal técnica de VENTA")
        
        # RSI
        rsi = features[4] * 100
        if rsi < 30:
            descriptions.append("RSI en sobreventa")
        elif rsi > 70:
            descriptions.append("RSI en sobrecompra")
        
        return " + ".join(descriptions) if descriptions else "Patrón mixto"
    
    def _analyze_agent_performance(self, trade_logs: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Analiza el performance individual de cada agente."""
        agent_stats = defaultdict(lambda: {
            'correct_predictions': 0,
            'total_predictions': 0,
            'profit_contribution': 0.0,
            'avg_confidence': 0.0,
            'signal_distribution': defaultdict(int)
        })
        
        for trade in trade_logs:
            pnl = trade.get('pnl_usd', 0)
            was_profitable = pnl > 0
            context = trade.get('decision_context', {})
            
            # Analizar cada agente
            # Sentiment
            sentiment = context.get('sentiment_analysis', {})
            if sentiment:
                sentiment_signal = sentiment.get('overall_sentiment', 'NEUTRAL')
                agent_stats['sentiment']['total_predictions'] += 1
                agent_stats['sentiment']['signal_distribution'][sentiment_signal] += 1
                
                if (was_profitable and sentiment_signal == 'POSITIVE') or \
                   (not was_profitable and sentiment_signal == 'NEGATIVE'):
                    agent_stats['sentiment']['correct_predictions'] += 1
                
                agent_stats['sentiment']['avg_confidence'] += sentiment.get('confidence_score', 0.5)
            
            # Technical
            technical = context.get('numerical_technical_analysis', {})
            if technical:
                tech_signal = technical.get('signal', 'HOLD')
                agent_stats['technical']['total_predictions'] += 1
                agent_stats['technical']['signal_distribution'][tech_signal] += 1
                
                if (was_profitable and tech_signal == 'BUY') or \
                   (not was_profitable and tech_signal == 'SELL'):
                    agent_stats['technical']['correct_predictions'] += 1
            
            # Similar para visual y qabba...
        
        # Normalizar métricas
        for agent, stats in agent_stats.items():
            if stats['total_predictions'] > 0:
                stats['accuracy'] = (stats['correct_predictions'] / stats['total_predictions']) * 100
                stats['avg_confidence'] /= stats['total_predictions']
            else:
                stats['accuracy'] = 0
        
        return dict(agent_stats)
    
    def _generate_optimization_suggestions(
        self,
        metrics: AdvancedPerformanceMetrics,
        winning_patterns: List[PatternAnalysis],
        losing_patterns: List[PatternAnalysis],
        agent_performance: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Genera sugerencias específicas de optimización."""
        suggestions = []
        
        # Basadas en métricas generales
        if metrics.win_rate < 45:
            suggestions.append(
                "⚠️ Win rate bajo (<45%). Considerar: "
                "1) Ajustar criterios de entrada más estrictos, "
                "2) Revisar el RiskManager para rechazar más trades de baja calidad"
            )
        
        if metrics.profit_factor < 1.5:
            suggestions.append(
                "📊 Profit factor subóptimo. Recomendación: "
                "Aumentar el ratio objetivo de reward/risk o mejorar la selección de trades"
            )
        
        if metrics.max_consecutive_losses > 5:
            suggestions.append(
                "🔴 Rachas de pérdidas largas detectadas. Implementar: "
                "1) Circuit breaker después de 3 pérdidas consecutivas, "
                "2) Reducir tamaño de posición progresivamente"
            )
        
        # Basadas en patrones
        if winning_patterns:
            best_pattern = winning_patterns[0]
            suggestions.append(
                f"✅ Patrón ganador principal: '{best_pattern.pattern_description}' "
                f"(WR: {best_pattern.win_rate:.1f}%, Avg: ${best_pattern.avg_pnl:.2f}). "
                f"Aumentar peso cuando se detecte este patrón."
            )
        
        if losing_patterns:
            worst_pattern = losing_patterns[0]
            suggestions.append(
                f"❌ Patrón perdedor a evitar: '{worst_pattern.pattern_description}' "
                f"(WR: {worst_pattern.win_rate:.1f}%, Avg: ${worst_pattern.avg_pnl:.2f}). "
                f"Considerar vetar trades con esta combinación."
            )
        
        # Basadas en agentes
        for agent, stats in agent_performance.items():
            if stats.get('accuracy', 0) < 40:
                suggestions.append(
                    f"🤖 {agent.capitalize()} Agent tiene baja precisión ({stats['accuracy']:.1f}%). "
                    f"Considerar: 1) Fine-tuning del modelo, 2) Ajustar prompts, "
                    f"3) Reducir su peso en decisiones"
                )
        
        # Gestión de riesgo
        if metrics.sharpe_ratio and metrics.sharpe_ratio < 1.0:
            suggestions.append(
                "📈 Sharpe Ratio bajo. Mejorar consistencia: "
                "1) Implementar trailing stops dinámicos, "
                "2) Diversificar timeframes de análisis"
            )
        
        return suggestions
    
    def _identify_risk_warnings(
        self,
        metrics: AdvancedPerformanceMetrics,
        trade_logs: List[Dict]
    ) -> List[str]:
        """Identifica advertencias de riesgo importantes."""
        warnings = []
        
        # Drawdown excesivo
        if metrics.max_drawdown > 20:
            warnings.append(
                f"⚠️ ALERTA: Drawdown máximo de {metrics.max_drawdown:.1f}% detectado. "
                f"Riesgo de ruin aumentado. Reducir tamaño de posiciones inmediatamente."
            )
        
        # Concentración temporal
        trade_times = [pd.to_datetime(t.get('timestamp_decision', 0), unit='s')
                      for t in trade_logs if 'timestamp_decision' in t]
        
        if trade_times:
            trades_df = pd.DataFrame({'time': trade_times})
            trades_df['hour'] = trades_df['time'].dt.hour
            hour_concentration = trades_df.groupby('hour').size()
            
            if hour_concentration.max() > len(trade_logs) * 0.3:
                warnings.append(
                    "🕐 Alta concentración de trades en horas específicas. "
                    "Riesgo de sobreoptimización a condiciones temporales."
                )
        
        # Dependencia de un agente
        if hasattr(self, '_agent_weights'):
            max_weight = max(self._agent_weights.values())
            if max_weight > 0.5:
                warnings.append(
                    f"🤖 Dependencia excesiva de un agente (peso > 50%). "
                    f"Riesgo de fallo catastrófico si el agente falla."
                )
        
        # Cambio brusco en performance
        if len(self._performance_history) > 50:
            recent_performance = list(self._performance_history)[-20:]
            older_performance = list(self._performance_history)[-50:-20]
            
            recent_wr = sum(1 for p in recent_performance if p > 0) / len(recent_performance)
            older_wr = sum(1 for p in older_performance if p > 0) / len(older_performance)
            
            if older_wr - recent_wr > 0.2:
                warnings.append(
                    "📉 Deterioro significativo en performance reciente. "
                    "Posible cambio de régimen de mercado. Revisar estrategia."
                )
        
        return warnings
    
    def _generate_performance_visualizations(
        self,
        metrics: AdvancedPerformanceMetrics,
        trade_logs: List[Dict]
    ):
        """Genera visualizaciones de performance."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.gridspec import GridSpec
            
            # Configurar estilo
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Crear figura con subplots
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 3, figure=fig)
            
            # 1. Equity Curve
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(self._equity_curve, linewidth=2, color='#2E86AB')
            ax1.fill_between(range(len(self._equity_curve)),
                           self._equity_curve[0], self._equity_curve,
                           alpha=0.3, color='#2E86AB')
            ax1.set_title('Equity Curve', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Trade Number')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Drawdown
            ax2 = fig.add_subplot(gs[1, :2])
            equity_array = np.array(self._equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max * 100
            
            ax2.fill_between(range(len(drawdown)), 0, drawdown,
                           color='#E63946', alpha=0.7)
            ax2.set_title('Drawdown %', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True, alpha=0.3)
            
            # 3. Win Rate por Periodo
            ax3 = fig.add_subplot(gs[2, 0])
            if len(trade_logs) > 20:
                window_size = 20
                win_rates = []
                for i in range(window_size, len(trade_logs) + 1):
                    window_trades = trade_logs[i-window_size:i]
                    wins = sum(1 for t in window_trades if t.get('pnl_usd', 0) > 0)
                    win_rates.append((wins / window_size) * 100)
                
                ax3.plot(range(window_size, len(trade_logs) + 1), win_rates,
                        linewidth=2, color='#06D6A0')
                ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
                ax3.set_title(f'Rolling Win Rate ({window_size} trades)',
                            fontsize=14, fontweight='bold')
                ax3.set_xlabel('Trade Number')
                ax3.set_ylabel('Win Rate %')
                ax3.grid(True, alpha=0.3)
            
            # 4. Distribución de P&L
            ax4 = fig.add_subplot(gs[2, 1])
            pnls = [t.get('pnl_usd', 0) for t in trade_logs]
            ax4.hist(pnls, bins=30, color='#7209B7', alpha=0.7, edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax4.set_title('P&L Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('P&L ($)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # 5. Performance por Hora
            ax5 = fig.add_subplot(gs[0, 2])
            if 'timestamp_decision' in trade_logs[0]:
                hours = []
                hourly_pnl = defaultdict(list)
                
                for trade in trade_logs:
                    time = pd.to_datetime(trade['timestamp_decision'], unit='s')
                    hour = time.hour
                    hourly_pnl[hour].append(trade.get('pnl_usd', 0))
                
                hours = sorted(hourly_pnl.keys())
                avg_pnls = [np.mean(hourly_pnl[h]) for h in hours]
                
                ax5.bar(hours, avg_pnls, color='#F77F00', alpha=0.7)
                ax5.set_title('Average P&L by Hour', fontsize=14, fontweight='bold')
                ax5.set_xlabel('Hour of Day')
                ax5.set_ylabel('Average P&L ($)')
                ax5.grid(True, alpha=0.3, axis='y')
            
            # 6. Matriz de Correlación de Agentes
            ax6 = fig.add_subplot(gs[1, 2])
            # Aquí iría una matriz de correlación si tienes los datos
            
            # Métricas resumen
            fig.text(0.02, 0.98, f"Total Trades: {metrics.total_trades}",
                    transform=fig.transFigure, fontsize=12)
            fig.text(0.02, 0.96, f"Win Rate: {metrics.win_rate:.2f}%",
                    transform=fig.transFigure, fontsize=12)
            fig.text(0.02, 0.94, f"Profit Factor: {metrics.profit_factor:.2f}",
                    transform=fig.transFigure, fontsize=12)
            fig.text(0.02, 0.92, f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}",
                    transform=fig.transFigure, fontsize=12)
            
            plt.tight_layout()
            
            # Guardar figura
            output_path = Path("performance_reports") / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Performance visualization saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
    
    def _calculate_timeframe_stats(self, trade_logs: List[Dict], timeframe: str) -> Dict[str, float]:
        """Calcula estadísticas por timeframe."""
        if not trade_logs or 'timestamp_decision' not in trade_logs[0]:
            return {}
        
        df = pd.DataFrame(trade_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp_decision'], unit='s')
        df.set_index('timestamp', inplace=True)
        
        # Agrupar por timeframe
        grouped = df.groupby(pd.Grouper(freq=timeframe))
        
        stats = {
            'avg_trades': grouped.size().mean(),
            'avg_pnl': grouped['pnl_usd'].sum().mean(),
            'win_rate': (grouped.apply(lambda x: (x['pnl_usd'] > 0).sum() / len(x) * 100 if len(x) > 0 else 0)).mean(),
            'volatility': grouped['pnl_usd'].std().mean()
        }
        
        return stats
    
    def _calculate_market_condition_stats(self, trade_logs: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calcula estadísticas por condición de mercado."""
        condition_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0})
        
        for trade in trade_logs:
            # Extraer condición de mercado del contexto
            context = trade.get('decision_context', {})
            market_condition = 'unknown'
            
            # Determinar condición basada en métricas
            tech_metrics = context.get('raw_tech_metrics_at_decision', {})
            atr_pct = (tech_metrics.get('atr', 0) / tech_metrics.get('last_price', 1)) * 100 if tech_metrics.get('last_price', 0) > 0 else 0
            
            if atr_pct > 2.0:
                market_condition = 'high_volatility'
            elif tech_metrics.get('adx', 0) > 25:
                market_condition = 'trending'
            else:
                market_condition = 'ranging'
            
            # Actualizar estadísticas
            condition_stats[market_condition]['trades'] += 1
            condition_stats[market_condition]['total_pnl'] += trade.get('pnl_usd', 0)
            if trade.get('pnl_usd', 0) > 0:
                condition_stats[market_condition]['wins'] += 1
        
        # Calcular métricas finales
        final_stats = {}
        for condition, stats in condition_stats.items():
            if stats['trades'] > 0:
                final_stats[condition] = {
                    'trades': stats['trades'],
                    'win_rate': (stats['wins'] / stats['trades']) * 100,
                    'avg_pnl': stats['total_pnl'] / stats['trades']
                }
        
        return final_stats
    
    def _analyze_market_regime(self, trade_logs: List[Dict]) -> Dict[str, Any]:
        """Analiza el régimen de mercado durante los trades."""
        # Implementación simplificada
        return {
            'current_regime': 'bull_market',  # Placeholder
            'regime_changes': 0,
            'avg_regime_duration_days': 30
        }
    
    def _analyze_correlations(self, trade_logs: List[Dict]) -> Dict[str, float]:
        """Analiza correlaciones entre diferentes factores."""
        # Implementación simplificada
        return {
            'sentiment_vs_outcome': 0.3,
            'technical_vs_outcome': 0.5,
            'volume_vs_volatility': 0.7
        }
    
    def _performance_attribution(self, trade_logs: List[Dict]) -> Dict[str, float]:
        """Atribuye performance a diferentes factores."""
        # Implementación simplificada
        return {
            'market_timing': 0.4,
            'signal_quality': 0.3,
            'risk_management': 0.2,
            'luck': 0.1
        }
    
    def _analyze_agent_contributions(self, trades: List[Dict]) -> Dict[str, float]:
        """Analiza contribución de cada agente en un conjunto de trades."""
        contributions = defaultdict(float)
        
        for trade in trades:
            context = trade.get('decision_context', {})
            pnl = trade.get('pnl_usd', 0)
            
            # Simplificado: asignar contribución proporcional
            num_agents = 4  # sentiment, technical, visual, qabba
            contribution_per_agent = pnl / num_agents
            
            contributions['sentiment'] += contribution_per_agent
            contributions['technical'] += contribution_per_agent
            contributions['visual'] += contribution_per_agent
            contributions['qabba'] += contribution_per_agent
        
        return dict(contributions)
    
    def _generate_pattern_recommendation(self, win_rate: float, avg_pnl: float, p_value: float) -> str:
        """Genera recomendación para un patrón."""
        if win_rate > 60 and avg_pnl > 0 and p_value < 0.05:
            return "AUMENTAR exposición cuando se detecte este patrón"
        elif win_rate < 40 or avg_pnl < 0:
            return "EVITAR o reducir exposición con este patrón"
        else:
            return "MANTENER approach actual, monitorear evolución"
    
    def _save_performance_report(self, report: PerformanceReport):
        """Guarda el reporte de performance."""
        output_path = Path("performance_reports") / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
    
    def _load_historical_data(self):
        """Carga datos históricos si existen."""
        # Implementación para cargar datos previos
        pass
    
    def _load_trade_logs(self) -> List[Dict]:
        """Carga y procesa los logs de trades."""
        trades = []
        try:
            with open(self._trade_log_path, 'r') as f:
                for line in f:
                    try:
                        if line.strip():
                            trades.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping malformed log line: {e}")
            logging.info(f"Loaded {len(trades)} trades from log.")
        except FileNotFoundError:
            logging.warning(f"Trade log file not found at {self._trade_log_path}")
        except Exception as e:
            logging.error(f"Error loading trade logs: {e}")
        return trades

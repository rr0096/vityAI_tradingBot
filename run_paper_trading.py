"""
FenixTradingBot - Paper Trading Runner
Script para ejecutar fácilmente el sistema de paper trading
"""

import asyncio
import logging
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/paper_trading.log")
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """Imprime el banner del sistema"""
    print("=" * 60)
    print("🎭 FENIX TRADING BOT - PAPER TRADING SYSTEM")
    print("💹 Realistic Trading Simulation with Full Agent Architecture")
    print("=" * 60)
    print()

async def run_paper_trading_session(args):
    """Ejecuta una sesión de paper trading"""
    try:
        # Import aquí para evitar problemas de circular imports
        from paper_trading_system import PaperTradingSystem
        
        logger.info(f"Starting paper trading session:")
        logger.info(f"  Initial Balance: ${args.balance:,.2f}")
        logger.info(f"  Duration: {args.duration} minutes")
        logger.info(f"  Symbols: {args.symbols}")
        
        # Crear sistema
        system = PaperTradingSystem(initial_balance=args.balance)
        
        # Actualizar símbolos si se especificaron
        if args.symbols:
            system.symbols = args.symbols.split(",")
        
        # Inicializar
        logger.info("🔧 Initializing paper trading system...")
        if not await system.initialize():
            logger.error("❌ Failed to initialize system")
            return
        
        logger.info("✅ System initialized successfully")
        logger.info("🚀 Starting paper trading...")
        
        # Crear task de trading
        trading_task = asyncio.create_task(system.start_trading())
        
        try:
            # Ejecutar por el tiempo especificado
            if args.duration > 0:
                await asyncio.sleep(args.duration * 60)
                logger.info(f"⏰ {args.duration} minutes completed")
            else:
                logger.info("🔄 Running indefinitely (Ctrl+C to stop)")
                await trading_task
                
        except KeyboardInterrupt:
            logger.info("⛔ Stopping paper trading (user interrupt)")
        
        # Detener sistema
        system.stop_trading()
        trading_task.cancel()
        
        # Mostrar resumen final
        logger.info("📊 Generating final report...")
        summary = system.get_portfolio_summary()
        
        print_final_report(summary, args.balance)
        
        # Guardar reporte
        save_session_report(summary, args)
        
    except Exception as e:
        logger.error(f"❌ Error in paper trading session: {e}")
        import traceback
        traceback.print_exc()

def print_final_report(summary, initial_balance):
    """Imprime el reporte final de la sesión"""
    print("\n" + "=" * 60)
    print("📊 PAPER TRADING SESSION REPORT")
    print("=" * 60)
    
    print(f"💰 Initial Balance:      ${initial_balance:>12,.2f}")
    print(f"💳 Final Cash:           ${summary['current_cash']:>12,.2f}")
    print(f"📈 Unrealized P&L:       ${summary['unrealized_pnl']:>+12,.2f}")
    print(f"🎯 Total Portfolio:      ${summary['total_portfolio_value']:>12,.2f}")
    print("─" * 60)
    print(f"📊 Total Return:         ${summary['total_return']:>+12,.2f}")
    print(f"📈 Return %:             {summary['return_percentage']:>+12.2f}%")
    print("─" * 60)
    print(f"🏠 Active Positions:     {len(summary['active_positions']):>12}")
    
    if summary['active_positions']:
        print("\n🔍 ACTIVE POSITIONS:")
        for pos in summary['active_positions']:
            pnl_color = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
            print(f"  {pnl_color} {pos['symbol']}: {pos['side']} {pos['quantity']:.6f} @ ${pos['entry_price']:.6f}")
            print(f"      Current: ${pos['current_price']:.6f} | P&L: ${pos['unrealized_pnl']:+.2f}")
    
    print("=" * 60)

def save_session_report(summary, args):
    """Guarda el reporte de la sesión"""
    try:
        reports_dir = Path("logs/paper_trading_reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"session_{timestamp}.json"
        
        report_data = {
            "session_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "initial_balance": args.balance,
                "duration_minutes": args.duration,
                "symbols": args.symbols
            },
            "results": summary
        }
        
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"📄 Session report saved: {report_file}")
        
    except Exception as e:
        logger.error(f"Error saving session report: {e}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="FenixTradingBot Paper Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for 30 minutes with $10,000
  python run_paper_trading.py --balance 10000 --duration 30
  
  # Run with custom symbols
  python run_paper_trading.py --symbols "BTCUSDT,ETHUSDT" --duration 60
  
  # Run indefinitely (stop with Ctrl+C)
  python run_paper_trading.py --duration 0
        """
    )
    
    parser.add_argument(
        "--balance", 
        type=float, 
        default=10000.0,
        help="Initial balance in USDT (default: 10000)"
    )
    
    parser.add_argument(
        "--duration", 
        type=int, 
        default=30,
        help="Session duration in minutes (0 = indefinite, default: 30)"
    )
    
    parser.add_argument(
        "--symbols", 
        type=str,
        help="Trading symbols (comma-separated, default: from config)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    try:
        asyncio.run(run_paper_trading_session(args))
    except KeyboardInterrupt:
        print("\n👋 Paper trading session interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()

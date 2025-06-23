#!/usr/bin/env python3

"""
ASCII Banner para FENIX Trading Bot
Creado para celebrar el lanzamiento del proyecto
Usando el logo tipográfico seleccionado por Giovanni
"""

def print_fenix_banner():
    """Imprime el banner principal del bot FENIX"""
    
    fenix_banner = r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                                                                               ║
║     ███████╗███████╗███╗   ██╗██╗██╗  ██╗                                     ║
║     ██╔════╝██╔════╝████╗  ██║██║╚██╗██╔╝                                     ║
║     █████╗  █████╗  ██╔██╗ ██║██║ ╚███╔╝                                      ║
║     ██╔══╝  ██╔══╝  ██║╚██╗██║██║ ██╔██╗                                      ║
║     ██║     ███████╗██║ ╚████║██║██╔╝ ██╗                                     ║
║     ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝                                     ║
║                                                                               ║
║                         🔥 TRADING BOT 🔥                                    ║
║                      🚀 AUTONOMOUS AI TRADER 🚀                              ║
║                                                                               ║
║               ═══════════════════════════════════════════                     ║
║                                                                               ║
║               🏆 PROYECTO COMPLETADO - LISTO PARA TRADING 🏆                 ║
║                                                                               ║
║               📅 Desarrollado con dedicación durante 2 meses                 ║
║               🌟 Licencia: Apache 2.0 (Open Source)                          ║
║               👨‍💻 Desarrollador: Giovanni Arangio                             ║
║               🤖 Bot de Trading Completamente Autónomo                        ║
║                                                                               ║
║               🎯 CARACTERÍSTICAS PRINCIPALES:                                 ║
║               • 🧠 Sistema Multi-Agente con IA Avanzada                      ║
║               • 📊 Captura Real de Gráficos (Bitget Integration)             ║
║               • 🔍 Análisis Visual Inteligente de Patrones                   ║
║               • 📰 Análisis de Sentimiento en Tiempo Real                    ║
║               • ⚡ Ejecución Automática de Trades                           ║
║               • 🛡️ Gestión Avanzada de Riesgo                               ║
║               • 📈 Backtesting y Paper Trading                               ║
║               • 🔄 WebSocket Real-Time Data Feed                             ║
║                                                                               ║
║               🚀 FENIX HAS RISEN - READY TO CONQUER MARKETS 🚀               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    
    print(fenix_banner)
    
    # Información adicional con colores
    try:
        # ANSI color codes
        RED = '\033[91m'
        YELLOW = '\033[93m'
        GREEN = '\033[92m'
        BLUE = '\033[94m'
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        
        print(f"{RED}🔥 SISTEMA: {BOLD}FENIX TRADING BOT INICIADO{RESET} {RED}🔥{RESET}")
        print(f"{GREEN}✅ TIMEFRAME: {BOLD}5 MINUTOS{RESET} {GREEN}(OPTIMIZADO){RESET}")
        print(f"{BLUE}📊 GRÁFICOS: {BOLD}CAPTURA REAL DE BITGET{RESET} {BLUE}ACTIVA{RESET}")
        print(f"{PURPLE}🎯 STATUS: {BOLD}READY TO TRADE{RESET} {PURPLE}- QUE COMIENCE LA MAGIA{RESET}")
        print(f"{CYAN}{'═' * 80}{RESET}")
        
    except Exception:
        # Fallback sin colores
        print("🔥 SISTEMA: FENIX TRADING BOT INICIADO 🔥")
        print("✅ TIMEFRAME: 5 MINUTOS (OPTIMIZADO)")
        print("📊 GRÁFICOS: CAPTURA REAL DE BITGET ACTIVA")
        print("🎯 STATUS: READY TO TRADE - QUE COMIENCE LA MAGIA")
        print("═" * 80)

def print_startup_banner():
    """Banner compacto para el inicio del sistema"""
    
    startup = r"""
╔═══════════════════════════════════════════════════════════╗
║  ███████╗███████╗███╗   ██╗██╗██╗  ██╗                    ║
║  ██╔════╝██╔════╝████╗  ██║██║╚██╗██╔╝                    ║
║  █████╗  █████╗  ██╔██╗ ██║██║ ╚███╔╝                     ║
║  ██╔══╝  ██╔══╝  ██║╚██╗██║██║ ██╔██╗                     ║
║  ██║     ███████╗██║ ╚████║██║██╔╝ ██╗                    ║
║  ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝                    ║
║                                                           ║
║           🔥 TRADING BOT - INICIANDO SISTEMA 🔥          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(startup)

def print_trade_banner():
    """Banner para cuando se ejecuta un trade"""
    
    trade = r"""
╔═════════════════════════════════════════════════════════════╗
║   ███████╗███████╗███╗   ██╗██╗██╗  ██╗                     ║
║   ██╔════╝██╔════╝████╗  ██║██║╚██╗██╔╝                     ║
║   █████╗  █████╗  ██╔██╗ ██║██║ ╚███╔╝                      ║
║   ██╔══╝  ██╔══╝  ██║╚██╗██║██║ ██╔██╗                      ║
║   ██║     ███████╗██║ ╚████║██║██╔╝ ██╗                     ║
║   ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝                     ║
║                                                             ║
║              🚀 TRADE EXECUTION IN PROGRESS 🚀              ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝
"""
    print(trade)

def print_farewell_banner():
    """Banner de despedida cuando el bot se cierra"""
    
    farewell = r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     ███████╗███████╗███╗   ██╗██╗██╗  ██╗                                     ║
║     ██╔════╝██╔════╝████╗  ██║██║╚██╗██╔╝                                     ║
║     █████╗  █████╗  ██╔██╗ ██║██║ ╚███╔╝                                      ║
║     ██╔══╝  ██╔══╝  ██║╚██╗██║██║ ██╔██╗                                      ║
║     ██║     ███████╗██║ ╚████║██║██╔╝ ██╗                                     ║
║     ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝                                     ║
║                                                                               ║
║                      🔥 TRADING SESSION ENDED 🔥                             ║
║                                                                               ║
║               ═══════════════════════════════════════════                     ║
║                                                                               ║
║                   ✨ GRACIAS POR USAR FENIX BOT ✨                           ║
║                                                                               ║
║                 🚀 THE PHOENIX RESTS, BUT ALWAYS RETURNS 🚀                  ║
║                                                                               ║
║                     📊 TRADING FINALIZADO CON ÉXITO 📊                       ║
║                                                                               ║
║                      🌟 HASTA LA PRÓXIMA SESIÓN 🌟                           ║
║                                                                               ║
║               ═══════════════════════════════════════════                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(farewell)

# Alias para compatibilidad con el import que ya está en live_trading.py
def print_farewell_message():
    """Alias para print_farewell_banner() para compatibilidad"""
    print_farewell_banner()

def print_error_banner():
    """Banner para errores críticos"""
    
    error = r"""
╔═══════════════════════════════════════════════════════════════╗
║  ███████╗███████╗███╗   ██╗██╗██╗  ██╗                        ║
║  ██╔════╝██╔════╝████╗  ██║██║╚██╗██╔╝                        ║
║  █████╗  █████╗  ██╔██╗ ██║██║ ╚███╔╝                         ║
║  ██╔══╝  ██╔══╝  ██║╚██╗██║██║ ██╔██╗                         ║
║  ██║     ███████╗██║ ╚████║██║██╔╝ ██╗                        ║
║  ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝                        ║
║                                                               ║
║               🚨 CRITICAL ERROR DETECTED 🚨                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(error)

if __name__ == "__main__":
    print("=== FENIX TRADING BOT - BANNER DEMO ===\n")
    
    print("1. Banner Principal:")
    print_fenix_banner()
    
    input("\nPresiona Enter para ver el banner de inicio...")
    print("\n2. Banner de Inicio:")
    print_startup_banner()
    
    input("\nPresiona Enter para ver el banner de trade...")
    print("\n3. Banner de Trade:")
    print_trade_banner()
    
    input("\nPresiona Enter para ver el banner de despedida...")
    print("\n4. Banner de Despedida:")
    print_farewell_banner()
    
    input("\nPresiona Enter para ver el banner de error...")
    print("\n5. Banner de Error:")
    print_error_banner()
    
    print("\n🎉 ¡Todos los banners están listos para tu bot FENIX!")

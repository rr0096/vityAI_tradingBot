#!/bin/bash

# CONFIGURACIÓN AUTOMÁTICA DE SAFARI WEBDRIVER PARA MACOS
# Este script te ayuda a habilitar Safari WebDriver

echo "🦆 CONFIGURACIÓN DE SAFARI WEBDRIVER PARA DUCKDUCKGO CAPTURE"
echo "============================================================"

echo ""
echo "📋 PASOS MANUALES NECESARIOS:"
echo ""

echo "1️⃣ HABILITAR SAFARI WEBDRIVER:"
echo "   • Abre Safari"
echo "   • Ve a Safari > Preferencias (⌘,)"
echo "   • Pestaña 'Avanzadas'"
echo "   • ✅ Marca 'Mostrar menú Desarrollo en la barra de menús'"
echo "   • Cierra Preferencias"
echo ""

echo "2️⃣ ACTIVAR AUTOMATIZACIÓN REMOTA:"
echo "   • En Safari, ve a Desarrollo > Permitir automatización remota"
echo "   • ✅ Asegúrate de que esté marcado"
echo ""

echo "3️⃣ VERIFICAR PERMISOS (si es necesario):"
echo "   • Configuración del Sistema > Privacidad y seguridad"
echo "   • Buscar 'Automatización' o 'Accesibilidad'"
echo "   • ✅ Permitir acceso a Terminal o Python"
echo ""

echo "🧪 DESPUÉS DE CONFIGURAR, EJECUTA:"
echo "   python tools/tradingview_duckduckgo_capture.py"
echo ""

echo "💡 NOTA: Safari WebDriver es lo que usa DuckDuckGo en macOS"
echo "    para automatización. ¡Es completamente seguro!"
echo ""

# Verificar si Selenium está instalado
python3 -c "import selenium; print('✅ Selenium está instalado')" 2>/dev/null || {
    echo "❌ Selenium no está instalado. Instalando..."
    pip3 install selenium
}

echo "🎯 ¡LISTO! Ahora ejecuta el script de captura."

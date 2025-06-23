#!/usr/bin/env python3

"""
Script para verificar qué modelos tienes disponibles en Ollama
y cuáles funcionan mejor con Instructor
"""

import subprocess
import json

def check_ollama_models():
    """Verifica los modelos disponibles en Ollama"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("🔍 MODELOS DISPONIBLES EN OLLAMA:")
            print(result.stdout)
            return result.stdout
        else:
            print("❌ Error al obtener modelos de Ollama")
            return None
    except FileNotFoundError:
        print("❌ Ollama no está instalado o no está en PATH")
        return None

def get_recommended_models():
    """Lista de modelos recomendados para Instructor"""
    
    models = {
        "🥇 EXCELENTES (Instructor funciona perfecto)": [
            "llama3.1:8b-instruct-q4_k_m",
            "llama3.1:13b-instruct", 
            "mistral:7b-instruct",
            "codellama:13b-instruct",
            "neural-chat:7b-v3.3-q4_k_m"
        ],
        "🥈 BUENOS (Compatibles con configuración)": [
            "llama2:13b-chat",
            "vicuna:13b-v1.5",
            "openchat:7b",
            "starling-lm:7b-alpha"
        ],
        "🥉 PROBLEMÁTICOS (Pueden tener issues)": [
            "qwen2.5:7b-instruct-q5_k_m",
            "nous-hermes2pro",
            "dolphin-mixtral:8x7b"
        ]
    }
    
    print("\n📊 COMPATIBILIDAD CON INSTRUCTOR:")
    for category, model_list in models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"  • {model}")
    
    return models

def recommend_best_model(available_models_text):
    """Recomienda el mejor modelo disponible"""
    
    if not available_models_text:
        return None
        
    recommended_models = [
        "llama3.1:8b-instruct-q4_k_m",
        "llama3.1:13b-instruct", 
        "mistral:7b-instruct",
        "codellama:13b-instruct",
        "llama2:13b-chat"
    ]
    
    available_list = available_models_text.lower()
    
    print("\n🎯 RECOMENDACIÓN:")
    for model in recommended_models:
        if model.lower() in available_list:
            print(f"✅ USAR: {model}")
            print(f"   Razón: Excelente compatibilidad con Instructor")
            return model
    
    print("❌ No se encontraron modelos altamente compatibles")
    print("💡 Sugerencia: Instalar un modelo recomendado")
    
    return None

def show_installation_commands():
    """Muestra comandos para instalar modelos recomendados"""
    
    print("\n📥 INSTALAR MODELOS RECOMENDADOS:")
    commands = [
        "ollama pull llama3.1:8b-instruct-q4_k_m",
        "ollama pull mistral:7b-instruct", 
        "ollama pull codellama:13b-instruct"
    ]
    
    for cmd in commands:
        print(f"  {cmd}")

if __name__ == "__main__":
    print("🤖 VERIFICADOR DE MODELOS PARA INSTRUCTOR\n")
    
    # Verificar modelos disponibles
    available = check_ollama_models()
    
    # Mostrar compatibilidad
    get_recommended_models()
    
    # Recomendar mejor modelo
    if available:
        best = recommend_best_model(available)
        if not best:
            show_installation_commands()
    else:
        print("\n💡 OPCIONES SIN OLLAMA:")
        print("  • Usar OpenAI: gpt-3.5-turbo")
        print("  • Usar OpenAI: gpt-4o-mini")
        print("  • Instalar Ollama y un modelo compatible")

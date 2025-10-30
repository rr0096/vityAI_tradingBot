# Usamos la imagen base de Python 3.11, como pide el proyecto
FROM python:3.11-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalamos las dependencias del sistema necesarias para TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib es complejo de instalar. Lo descargamos y compilamos.
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copiamos primero el archivo de requisitos para cachear la capa de instalación
COPY requirements.txt .

# Instalamos las dependencias de Python (la librería 'openai' ya está en requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código del proyecto al directorio de trabajo
COPY . .

# Comando por defecto para ejecutar (puedes cambiarlo o sobreescribirlo)
CMD ["python", "run_paper_trading.py"]

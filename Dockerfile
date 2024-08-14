# Ana imaj olarak Python imajını kullanın
FROM python:3.9-slim

# Uygulamanın çalışacağı dizini oluşturun
WORKDIR /app

# Gerekli bağımlılıkları kopyalayın ve yükleyin
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \ 
    libglib2.0-0 -y\
    libgomp1
COPY requirements.txt .
RUN pip install python-multipart
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyalayın
COPY . .

# Docker içinde FastAPI uygulamasını çalıştırın
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]


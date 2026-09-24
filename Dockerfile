# Uygulama imaji (Render icin). Yalnizca RUNTIME bagimliliklari kurulur -
# requirements-dev.txt bilerek kurulmuyor: augraphy/opencv/faker/pytest imajda
# gereksiz, ayrica augraphy tam opencv-python'u cekip libgl1/libglib2.0-0/libxcb1
# sistem paketlerini zorunlu kilardi (bkz. requirements-dev.txt'teki not).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodu + acilista indekslenecek bilgi tabani + statik frontend.
# data/chroma_db KOPYALANMIYOR: Render'da disk gecici oldugu icin koleksiyon
# her acilista yeniden kuruluyor (bkz. app/rag_index.py::ensure_indexed).
COPY app/ ./app/
COPY data/knowledge_base/ ./data/knowledge_base/
COPY frontend/ ./frontend/

# Kok olarak calismasin. /app sahipligi gerekiyor: ChromaDB acilista
# data/chroma_db klasorunu buraya yaziyor.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Render PORT'u ortam degiskeniyle veriyor; yerelde varsayilan 8000.
# Tek worker: 512 MB sinirinda her worker kendi Python surecini (+ RAGAgent'in
# actigi MCP alt surecini) tasiyor, ikinci bir worker sinira sigmaz.
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]

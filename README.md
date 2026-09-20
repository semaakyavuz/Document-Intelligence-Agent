# Document Intelligence Agent

[![CI](https://github.com/semaakyavuz/Document-Intelligence-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/semaakyavuz/Document-Intelligence-Agent/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=semaakyavuz_Document-Intelligence-Agent&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=semaakyavuz_Document-Intelligence-Agent)

Türkçe fatura görsellerini dört ajanlı bir yapay zekâ hattından geçirip, çıkarılan veriyi
kural tabanlı bir bilgi tabanına karşı doğrulayan, açıklanabilir bir risk puanı üreten ve
sonucu kalıcı olarak kaydeden uçtan uca bir belge zekâsı sistemi.

## Pipeline

LangGraph `StateGraph` ile sabit sıralı dört ajan (`app/graph.py`) — koşullu dallanma ya da
yeniden deneme yok:

| # | Ajan | Ne yapar | LLM kullanır mı |
|---|------|----------|-----------------|
| 01 | **Vision** (`app/agents/vision_agent.py`) | Görseli görsel-destekli LLM'e gönderir, JSON çıkarır | Evet |
| 02 | **RAG** (`app/agents/rag_agent.py`) | İlgili kuralları getirir; MCP client olarak çalışır | Hayır (embedding) |
| 03 | **Validation** (`app/agents/validation_agent.py`) | Dört deterministik kural sınıfıyla denetler | Hayır |
| 04 | **Report** (`app/agents/report_agent.py`) | Risk puanı, açıklama ve `final_report`'u kurar | Hayır |

RAG'ın kural sorgulaması ayrı bir süreçte çalışır: `app/mcp/rules_server.py`, resmî `mcp`
SDK'sıyla bağımsız bir **stdio alt süreci** olarak başlatılır ve ChromaDB üzerinde benzerlik
araması yapar. Bilgi tabanı `data/knowledge_base/` altındaki statik metin dosyalarıdır.

Doğrulama tamamen kod tabanlıdır (`MathConsistencyCheck`, `VatRateCheck`, `FormatCheck`,
`PriceRangeCheck`) — hiçbir anomali kararı LLM'e sorulmaz.

## Teknolojiler

Python 3.12 · LangGraph · FastAPI (SSE) · ChromaDB · MCP · SQLAlchemy 2.0 + PostgreSQL
(psycopg 3) · Pydantic Settings · vanilla JS + Chart.js (derleme adımı yok)

LLM/embedding sağlayıcısı `.env` üzerinden değiştirilebilir (Gemini, Ollama, Groq) —
`app/providers/factory.py`.

## Kurulum

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # sağlayıcı ve API anahtarlarını doldurun
```

PostgreSQL yerelde Docker ile:

```bash
docker compose up -d
```

## Çalıştırma

```bash
uvicorn app.api.main:app --reload
```

- Ana sayfa: `http://127.0.0.1:8000/static/upload.html`
- Panel: `http://127.0.0.1:8000/static/dashboard.html`

Uç noktalar: `POST /invoices`, `POST /invoices/stream` (SSE), `POST /invoices/bulk`,
`GET /invoices`, `GET /invoices/{id}`

## Testler

```bash
pytest -m "not slow"    # CI'ın çalıştırdığı set; dış servise dokunmaz
pytest                  # slow dahil: gerçek Ollama, gerçek MCP alt süreci, ChromaDB
```

`slow` işaretli testler gerçek dış servislere bağlanır ve CI'da çalıştırılmaz
(bkz. `pytest.ini`). Geri kalan testler sahte sağlayıcılar ve `sqlite:///:memory:`
kullanır.

Lint:

```bash
ruff check .            # kural seti: ruff.toml
```

## Katkı

Değişiklikleri birleştirmeden önce [`CODE_REVIEW_CHECKLIST.md`](CODE_REVIEW_CHECKLIST.md)
üzerinden geçin.

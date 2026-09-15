"""
FastAPI endpoint'lerini (app/api/main.py) TestClient ile test eder. Gercek Gemini/Ollama'ya
baglanmadan: FakeLLMProvider (test_graph.py ile ayni desen) create_app()'e enjekte edilir,
RAG adimi icin gercek MCP subprocess'i yerine StubRAGAgent kullanilir.

DB icin gercek Postgres yerine tmp_path altinda bir sqlite dosyasi kullanilir (in-memory
degil - sessionmaker() her istekte yeni bir SQLAlchemy baglantisi actigindan in-memory
sqlite her seferinde sifirlanirdi; bkz. test_db_models.py'deki ayni desen).

@pytest.mark.slow ile isaretli tek test, create_app()'i HICBIR override olmadan (gercek
.env ayarlariyla) kurar: gercek Gemini + gercek MCP/Ollama + ayaktaki gercek Postgres
container'ina (docker-compose.yml) karsi uctan uca calisir; olusturdugu kaydi kendi siler.
"""

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.db.models import InvoiceRecord
from app.providers.base import LLMProvider
from app.state import PipelineState

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAKE_EXTRACTION = {
    "invoice_no": "2025123456",
    "invoice_date": "01.01.2025",
    "seller_name": "Test A.Ş.",
    "seller_tax_no": "1234567890",
    "buyer_name": "Alıcı Ltd.",
    "items": [
        {
            "description": "Klavye",
            "quantity": 2,
            "unit_price": 500.0,
            "vat_rate": 0.20,
            "line_total": 1000.0,
            "vat_amount": 200.0,
        }
    ],
    "subtotal": 1000.0,
    "vat_total": 200.0,
    "grand_total": 1200.0,
}


class FakeLLMProvider(LLMProvider):
    """Aga cikmaz; sabit gecerli bir extraction JSON'u dondurur (test_graph.py ile ayni desen)."""

    def __init__(self, response_text: str = json.dumps(FAKE_EXTRACTION)):
        self.response_text = response_text

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        return self.response_text


class StubRAGAgent:
    """RAGAgent.arun() sozlesmesini taklit eder; gercek MCP subprocess'e/Ollama'ya hic dokunmaz."""

    def __init__(self, rules: list[str] | None = None):
        self.rules = rules or []

    async def arun(self, state: PipelineState) -> PipelineState:
        return state.model_copy(update={"retrieved_rules": self.rules})


@pytest.fixture
def app(tmp_path):
    settings = Settings(_env_file=None, DATABASE_URL=f"sqlite:///{tmp_path / 'test.db'}")
    return create_app(settings=settings, llm_provider=FakeLLMProvider(), rag_agent=StubRAGAgent())


def _post_invoice(client: TestClient, filename: str = "fatura.png", content: bytes = b"fake-image-bytes"):
    return client.post("/invoices", files={"file": (filename, io.BytesIO(content), "image/png")})


# --- POST /invoices --------------------------------------------------------

def test_create_invoice_returns_final_report_and_id(app):
    with TestClient(app) as client:
        response = _post_invoice(client)

    assert response.status_code == 201
    body = response.json()
    assert body["id"] is not None
    assert body["invoice_no"] == "2025123456"
    assert body["seller_name"] == "Test A.Ş."
    assert body["is_valid"] is True
    assert body["summary"] == "Fatura başarıyla işlendi, anomali yok."


def test_create_invoice_persists_a_record(app):
    with TestClient(app) as client:
        response = _post_invoice(client, filename="ornek.png")
        record_id = response.json()["id"]

        session_factory = app.state.session_factory
        with session_factory() as session:
            record = session.get(InvoiceRecord, record_id)
            assert record is not None
            assert record.image_filename == "ornek.png"
            assert record.invoice_no == "2025123456"


# --- GET /invoices ----------------------------------------------------------

def test_list_invoices_returns_newest_first_and_excludes_full_report(app):
    with TestClient(app) as client:
        _post_invoice(client, filename="birinci.png")
        _post_invoice(client, filename="ikinci.png")

        response = client.get("/invoices")

    assert response.status_code == 200
    body = response.json()
    assert [row["image_filename"] for row in body] == ["ikinci.png", "birinci.png"]
    assert "full_report" not in body[0]
    assert "anomalies" not in body[0]


def test_list_invoices_respects_limit_and_offset(app):
    with TestClient(app) as client:
        for i in range(3):
            _post_invoice(client, filename=f"fatura_{i}.png")

        response = client.get("/invoices", params={"limit": 1, "offset": 1})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["image_filename"] == "fatura_1.png"


# --- GET /invoices/{id} ------------------------------------------------------

def test_get_invoice_returns_full_detail(app):
    with TestClient(app) as client:
        record_id = _post_invoice(client).json()["id"]
        response = client.get(f"/invoices/{record_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["full_report"]["invoice_no"] == "2025123456"
    assert body["anomalies"] == []


def test_get_invoice_returns_404_when_missing(app):
    with TestClient(app) as client:
        response = client.get("/invoices/999999")

    assert response.status_code == 404


# --- gercek servislerle uctan uca --------------------------------------------

@pytest.mark.slow
def test_real_end_to_end_with_gemini_mcp_ollama_and_real_postgres():
    """create_app() hicbir override olmadan (.env'deki gercek ayarlarla) kurulur: gercek
    Gemini + gercek MCP/Ollama + ayaktaki gercek Postgres container'ina karsi calisir.
    Asil kanit: async endpoint icinden graph.ainvoke() (RAGAgent.run()'a hic ugramadan)
    gercek bir istekte de sorunsuz calisiyor."""
    real_app = create_app()
    image_path = PROJECT_ROOT / "data" / "golden_mixed" / "images" / "invoice_0003_aug04.png"

    with TestClient(real_app) as client, image_path.open("rb") as f:
        response = client.post("/invoices", files={"file": (image_path.name, f, "image/png")})
        assert response.status_code == 201
        body = response.json()
        record_id = body["id"]

        # Test kendi olusturdugu kaydi siler; gercek Postgres'te iz birakmaz.
        session_factory = real_app.state.session_factory
        with session_factory() as session:
            record = session.get(InvoiceRecord, record_id)
            session.delete(record)
            session.commit()

    assert body["seller_name"] is not None

"""
POST /invoices/stream (SSE) endpoint'ini test eder. Gercek Gemini/Ollama'ya baglanmadan:
FakeLLMProvider/StubRAGAgent (test_api.py ile ayni desen) create_app()'e enjekte edilir.

graph.astream(stream_mode="updates") her node bitince o ana kadarki TAM PipelineState'i
verdigi icin (canlica dogrulandi - bkz. app/api/main.py'deki _stream_pipeline_events
docstring'i), "report" adiminin state_dict'i zaten dolu final_report tasir; "final" SSE
olayi bunu ayri bir ainvoke() cagrisi olmadan kullanir.

TestClient, StreamingResponse'u sonuna kadar tuketip response.text'te tum "data: ...\\n\\n"
bloklarini biriktirir; testler bu metni "\\n\\n" ile ayirip her olayi JSON olarak parse eder.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.providers.base import LLMProvider
from app.state import PipelineState

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
    """Aga cikmaz; sabit gecerli bir extraction JSON'u dondurur."""

    def __init__(self, response_text: str = json.dumps(FAKE_EXTRACTION)):
        self.response_text = response_text

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        return self.response_text


class RaisingLLMProvider(LLMProvider):
    """Vision adiminda kasitli olarak gercek bir exception firlatir (bkz. baglanti hatasi senaryosu)."""

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        raise RuntimeError("saglayiciya baglanilamadi")


class StubRAGAgent:
    """RAGAgent.arun() sozlesmesini taklit eder; gercek MCP subprocess'e/Ollama'ya hic dokunmaz."""

    def __init__(self, rules: list[str] | None = None):
        self.rules = rules or []

    async def arun(self, state: PipelineState) -> PipelineState:
        return state.model_copy(update={"retrieved_rules": self.rules})


def _build_app(tmp_path, llm_provider):
    settings = Settings(_env_file=None, DATABASE_URL=f"sqlite:///{tmp_path / 'test.db'}")
    return create_app(settings=settings, llm_provider=llm_provider, rag_agent=StubRAGAgent())


def _post_stream(client: TestClient, content: bytes = b"fake-image-bytes"):
    return client.post("/invoices/stream", files={"file": ("fatura.png", content, "image/png")})


def _parse_events(response_text: str) -> list[dict]:
    events = []
    for block in response_text.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data:")
        events.append(json.loads(block[len("data:") :].strip()))
    return events


def test_stream_emits_all_four_steps_then_final_in_order(tmp_path):
    app = _build_app(tmp_path, FakeLLMProvider())
    with TestClient(app) as client:
        response = _post_stream(client)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_events(response.text)
    steps = [e["step"] for e in events]
    assert steps == ["vision", "rag", "validation", "report", "final"]
    assert all(e["status"] == "done" for e in events[:4])


def test_stream_final_event_has_full_report_and_db_id(tmp_path):
    app = _build_app(tmp_path, FakeLLMProvider())
    with TestClient(app) as client:
        response = _post_stream(client)

    events = _parse_events(response.text)
    final_event = events[-1]

    assert final_event["step"] == "final"
    assert final_event["report"]["invoice_no"] == "2025123456"
    assert final_event["report"]["seller_name"] == "Test A.Ş."
    assert final_event["report"]["is_valid"] is True
    assert isinstance(final_event["id"], int)


def test_stream_persists_record_to_database(tmp_path):
    app = _build_app(tmp_path, FakeLLMProvider())
    with TestClient(app) as client:
        response = _post_stream(client)
        record_id = _parse_events(response.text)[-1]["id"]

    from app.db.models import InvoiceRecord

    with app.state.session_factory() as session:
        record = session.get(InvoiceRecord, record_id)
        assert record is not None
        assert record.invoice_no == "2025123456"


def test_stream_emits_error_event_when_a_node_raises(tmp_path):
    app = _build_app(tmp_path, RaisingLLMProvider())
    with TestClient(app) as client:
        response = _post_stream(client)

    assert response.status_code == 200
    events = _parse_events(response.text)

    assert len(events) == 1
    assert events[0]["step"] == "error"
    assert "saglayiciya baglanilamadi" in events[0]["message"]


def test_stream_error_does_not_write_to_database(tmp_path):
    app = _build_app(tmp_path, RaisingLLMProvider())
    with TestClient(app) as client:
        _post_stream(client)

    from app.db.models import InvoiceRecord

    with app.state.session_factory() as session:
        assert session.query(InvoiceRecord).count() == 0


@pytest.mark.slow
def test_real_stream_with_gemini_mcp_ollama_and_real_postgres():
    """create_app() hicbir override olmadan (.env'deki gercek ayarlarla): gercek Gemini +
    gercek MCP/Ollama + ayaktaki gercek Postgres container'ina karsi SSE akisini dener."""
    from pathlib import Path

    from app.db.models import InvoiceRecord

    real_app = create_app()
    image_path = Path(__file__).resolve().parents[1] / "data" / "golden_mixed" / "images" / "invoice_0003_aug04.png"

    with TestClient(real_app) as client:
        with image_path.open("rb") as f:
            response = client.post("/invoices/stream", files={"file": (image_path.name, f, "image/png")})

        assert response.status_code == 200
        events = _parse_events(response.text)
        assert [e["step"] for e in events[:4]] == ["vision", "rag", "validation", "report"]
        assert events[-1]["step"] == "final"
        record_id = events[-1]["id"]

        with real_app.state.session_factory() as session:
            record = session.get(InvoiceRecord, record_id)
            session.delete(record)
            session.commit()

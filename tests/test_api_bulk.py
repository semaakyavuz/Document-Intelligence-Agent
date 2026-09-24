"""
POST /invoices/bulk (toplu yukleme, en fazla 10 dosya) endpoint'ini test eder. Gercek
Gemini/Ollama'ya baglanmadan: FakeLLMProvider/StubRAGAgent (test_api_stream.py ile ayni
desen) create_app()'e enjekte edilir.

Dosyalar sirayla (bir "for" dongusunde, await ile) islendigi icin VisionAgent'in
llm_provider.generate() cagrilari da dosya sirasiyla birebir eslesir - FailOnNthCallProvider
bunu kullanarak "N'inci dosya" gibi belirli bir dosyanin hata vermesini simule eder.
"""

import json

from fastapi.testclient import TestClient

from app.api.main import CLIENT_ERROR_MESSAGE, create_app
from app.config import Settings
from app.db.models import InvoiceRecord
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
    """Aga cikmaz; sabit gecerli bir extraction JSON'u dondurur, cagri sirasini kaydeder."""

    def __init__(self, response_text: str = json.dumps(FAKE_EXTRACTION)):
        self.response_text = response_text
        self.call_order: list[str | None] = []

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        self.call_order.append(image_path)
        return self.response_text


class FailOnNthCallProvider(LLMProvider):
    """N'inci generate() cagrisinda (yani N'inci dosyada) kasitli hata firlatir."""

    def __init__(self, fail_index: int, response_text: str = json.dumps(FAKE_EXTRACTION)):
        self.fail_index = fail_index
        self.response_text = response_text
        self.call_count = 0

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        self.call_count += 1
        if self.call_count == self.fail_index:
            raise RuntimeError(f"kasitli hata: dosya #{self.fail_index}")
        return self.response_text


class StubRAGAgent:
    """RAGAgent.arun() sozlesmesini taklit eder; gercek MCP subprocess'e/Ollama'ya hic dokunmaz."""

    def __init__(self, rules: list[str] | None = None):
        self.rules = rules or []

    async def arun(self, state: PipelineState) -> PipelineState:
        return state.model_copy(update={"retrieved_rules": self.rules})


def _build_app(tmp_path, llm_provider):
    settings = Settings(_env_file=None, DATABASE_URL=f"sqlite:///{tmp_path / 'test.db'}")
    return create_app(settings=settings, llm_provider=llm_provider, rag_agent=StubRAGAgent())


def _files_payload(names: list[str]) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (name, b"fake-image-bytes", "image/png")) for name in names]


def _parse_events(response_text: str) -> list[dict]:
    events = []
    for block in response_text.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data:")
        events.append(json.loads(block[len("data:") :].strip()))
    return events


def test_bulk_rejects_more_than_ten_files_without_processing_any(tmp_path):
    app = _build_app(tmp_path, FakeLLMProvider())
    names = [f"fatura_{i}.png" for i in range(11)]

    with TestClient(app) as client:
        response = client.post("/invoices/bulk", files=_files_payload(names))

        assert response.status_code == 422
        assert response.json()["detail"] == "En fazla 10 fatura aynı anda yüklenebilir."

        with app.state.session_factory() as session:
            assert session.query(InvoiceRecord).count() == 0


def test_bulk_accepts_exactly_ten_files(tmp_path):
    provider = FakeLLMProvider()
    app = _build_app(tmp_path, provider)
    names = [f"fatura_{i}.png" for i in range(10)]

    with TestClient(app) as client:
        response = client.post("/invoices/bulk", files=_files_payload(names))

    assert response.status_code == 200
    events = _parse_events(response.text)
    assert events[-1] == {"step": "batch_complete", "processed": 10, "failed": 0}


def test_bulk_processes_files_sequentially_in_submitted_order(tmp_path):
    provider = FakeLLMProvider()
    app = _build_app(tmp_path, provider)
    names = ["birinci.png", "ikinci.png", "ucuncu.png"]

    with TestClient(app) as client:
        response = client.post("/invoices/bulk", files=_files_payload(names))

    events = _parse_events(response.text)
    per_file_events = events[:-1]

    assert [e["file"] for e in per_file_events] == names
    assert [e["index"] for e in per_file_events] == [1, 2, 3]
    assert all(e["total"] == 3 for e in per_file_events)
    assert all(e["status"] == "done" for e in per_file_events)
    assert len(provider.call_order) == 3


def test_bulk_writes_each_successful_file_to_database(tmp_path):
    app = _build_app(tmp_path, FakeLLMProvider())
    names = ["birinci.png", "ikinci.png"]

    with TestClient(app) as client:
        client.post("/invoices/bulk", files=_files_payload(names))

    with app.state.session_factory() as session:
        assert session.query(InvoiceRecord).count() == 2


def test_bulk_one_failing_file_does_not_affect_others(tmp_path):
    provider = FailOnNthCallProvider(fail_index=2)
    app = _build_app(tmp_path, provider)
    names = ["birinci.png", "ikinci.png", "ucuncu.png"]

    with TestClient(app) as client:
        response = client.post("/invoices/bulk", files=_files_payload(names))

    events = _parse_events(response.text)
    per_file_events = events[:-1]

    assert per_file_events[0]["status"] == "done"
    assert per_file_events[1]["status"] == "error"
    # Ic detay sizmamali: istemci genel mesaji gorur, saglayicinin metni ("dosya #2")
    # yalnizca sunucu logunda kalir.
    assert per_file_events[1]["message"] == CLIENT_ERROR_MESSAGE
    assert "dosya #2" not in per_file_events[1]["message"]
    assert per_file_events[2]["status"] == "done"

    assert events[-1] == {"step": "batch_complete", "processed": 2, "failed": 1}

    with app.state.session_factory() as session:
        assert session.query(InvoiceRecord).count() == 2
        filenames = {r.image_filename for r in session.query(InvoiceRecord).all()}
        assert filenames == {"birinci.png", "ucuncu.png"}

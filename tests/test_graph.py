"""
app/graph.py'deki LangGraph StateGraph'inin dogru sirada (vision -> rag -> validation
-> report) calistigini ve her adimda state'in dogru alanlarla guncellendigini test eder.

Gercek Ollama/Gemini'ye ve gercek MCP subprocess'ine baglanmaz: sahte LLMProvider
dogrudan build_pipeline_graph()'a enjekte edilir, RAG adimi icinse gercek RAGAgent
yerine arun() sozlesmesini taklit eden hafif bir StubRAGAgent kullanilir (RAGAgent'in
kendisi, gercek MCP server'a karsi tests/test_rag_agent.py'de test edilir).

rag_node artik async oldugu icin (bkz. app/graph.py) graph tek yonlu senkron
.invoke()/.stream() yerine .ainvoke()/.astream() ile calistirilmali (LangGraph bunu
zorunlu kiliyor). pytest-asyncio eklemeden test etmek icin test fonksiyonlari senkron
kalir, icten asyncio.run() ile async graph API'sini cagirir.
"""

import asyncio
import json

import pytest

from app.config import Settings
from app.graph import build_pipeline_graph
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

KLAVYE_RULE_TEXT = "Tipik birim fiyat aralığı: Klavye genellikle 300-2500 TL arasındadır."


class FakeLLMProvider(LLMProvider):
    """Aga cikmaz; sabit gecerli bir extraction JSON'u dondurur."""

    def __init__(self, response_text: str = json.dumps(FAKE_EXTRACTION)):
        self.response_text = response_text
        self.calls: list[str | None] = []

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        self.calls.append(image_path)
        return self.response_text


class StubRAGAgent:
    """RAGAgent.arun() sozlesmesini taklit eder; gercek MCP subprocess'e/Ollama'ya hic dokunmaz."""

    def __init__(self, rules: list[str]):
        self.rules = rules

    async def arun(self, state: PipelineState) -> PipelineState:
        return state.model_copy(update={"retrieved_rules": self.rules})


@pytest.fixture
def fake_llm_provider():
    return FakeLLMProvider()


@pytest.fixture
def graph(fake_llm_provider):
    settings = Settings(_env_file=None, RAG_TOP_K=3)
    stub_rag_agent = StubRAGAgent(rules=[KLAVYE_RULE_TEXT])
    return build_pipeline_graph(settings=settings, llm_provider=fake_llm_provider, rag_agent=stub_rag_agent)


async def _astream_steps(graph, image_path: str) -> list[tuple[str, dict]]:
    initial_state = PipelineState(image_path=image_path)
    steps = []
    async for update in graph.astream(initial_state, stream_mode="updates"):
        (node_name, state_dict), = update.items()
        steps.append((node_name, state_dict))
    return steps


def _run_steps(graph, image_path: str = "data/golden/images/x.png") -> list[tuple[str, dict]]:
    """graph.astream(stream_mode='updates') sonuclarini [(node_adi, o_andaki_tam_state), ...] olarak toplar."""
    return asyncio.run(_astream_steps(graph, image_path))


# --- Sira ve adim adim state guncellemesi ------------------------------------

def test_graph_visits_all_four_nodes_in_order(graph):
    steps = _run_steps(graph)
    assert [name for name, _ in steps] == ["vision", "rag", "validation", "report"]


def test_vision_step_fills_raw_extraction(graph):
    steps = dict(_run_steps(graph))
    assert steps["vision"]["raw_extraction"] == FAKE_EXTRACTION
    # bu asamada sonraki alanlar henuz dolmamis olmali
    assert steps["vision"]["retrieved_rules"] == []
    assert steps["vision"]["anomalies"] == []


def test_rag_step_fills_retrieved_rules(graph):
    steps = dict(_run_steps(graph))
    assert KLAVYE_RULE_TEXT in steps["rag"]["retrieved_rules"]


def test_validation_step_fills_anomalies_and_is_valid(graph):
    steps = dict(_run_steps(graph))
    # FAKE_EXTRACTION matematiksel olarak tutarli, KDV/format gecerli, fiyat araliginda:
    # anomali beklenmiyor.
    assert steps["validation"]["anomalies"] == []
    assert steps["validation"]["is_valid"] is True


def test_report_step_fills_final_report(graph):
    steps = dict(_run_steps(graph))
    final_report = steps["report"]["final_report"]
    assert final_report["summary"] == "Fatura başarıyla işlendi, anomali yok."
    assert final_report["invoice_no"] == "2025123456"
    assert final_report["is_valid"] is True


# --- ainvoke() ile uctan uca ----------------------------------------------------

def test_ainvoke_produces_the_same_final_report_as_streaming(graph):
    initial_state = PipelineState(image_path="data/golden/images/x.png")
    raw_result = asyncio.run(graph.ainvoke(initial_state))
    # graph.ainvoke() duz bir dict dondurur; None kalan alanlar (burada hicbiri) dusebilir,
    # PipelineState(**raw_result) varsayilanlarla geri tamamlar (run_pipeline_graph.py ile ayni desen).
    final_state = PipelineState(**raw_result)

    assert final_state.final_report["summary"] == "Fatura başarıyla işlendi, anomali yok."
    assert final_state.final_report["anomaly_count"] == 0
    assert final_state.is_valid is True
    assert final_state.raw_extraction == FAKE_EXTRACTION


def test_image_path_is_passed_through_to_vision_provider(graph, fake_llm_provider):
    initial_state = PipelineState(image_path="data/golden/images/specific.png")
    asyncio.run(graph.ainvoke(initial_state))
    assert fake_llm_provider.calls == ["data/golden/images/specific.png"]

"""
RAGAgent'in sorgu kurma mantigini ve MCP client davranisini test eder.

Gercek Ollama'ya ve gercek app/mcp/rules_server.py'ye (factory'den Ollama kurar)
hic baglanmaz: RAGAgent, server_command/server_args ile hafif, tek dosyalik sahte
bir MCP server'a (tmp_path'e yazilir, gercek bir alt surec olarak baslatilir)
yonlendirilir. Boylece MCP stdio protokolu gercekten sinanmis olur, sadece
Ollama/ChromaDB'ye ihtiyac duyulmaz.

app/mcp/rules_server.py'nin kendisi (embed+ChromaDB mantigi) tests/test_rules_server.py'de
test edilir.
"""

import asyncio
import sys

import pytest

from app.agents.rag_agent import RAGAgent, RuleQueryBuilder, RuleQueryError
from app.state import PipelineState

VALID_EXTRACTION = {
    "invoice_no": "2025123456",
    "seller_tax_no": "1234567890",
    "items": [
        {"description": "Klavye", "vat_rate": 0.20},
        {"description": "Mouse", "vat_rate": 0.01},
    ],
}

# query_rules(query, top_k): query icinde "__bos__" gecerse ToolError firlatir (RAGAgent'in
# bunu RuleQueryError'a cevirdigini test etmek icin); aksi halde her cagriyi ve top_k'yi
# kural metnine gomup, gercek rules_server.py'nin {"text","score","source"} seklini
# taklit ederek dondurur (RAGAgent'in dogru query/top_k gonderdigini ve dict seklini
# oldugu gibi ilettigini kanitlamak icin).
FAKE_SERVER_SOURCE = '''
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("fake-rules")


@server.tool()
def query_rules(query: str, top_k: int = 3) -> list[dict]:
    if "__bos__" in query:
        raise ToolError("Bilgi tabani bos. Once 'python scripts/index_knowledge_base.py' calistirin.")
    return [
        {"text": f"kural[{i}] icin sorgu: {query} (top_k={top_k})", "score": 0.1 * i, "source": f"fake_rule_{i}"}
        for i in range(top_k)
    ]


if __name__ == "__main__":
    server.run()
'''


@pytest.fixture
def fake_server_args(tmp_path) -> list[str]:
    """Sahte MCP server'i tmp_path'e yazar; RAGAgent'a server_args olarak verilecek yolu dondurur."""
    path = tmp_path / "fake_rules_server.py"
    path.write_text(FAKE_SERVER_SOURCE, encoding="utf-8")
    return [str(path)]


def _agent(fake_server_args, **kwargs) -> RAGAgent:
    return RAGAgent(server_command=sys.executable, server_args=fake_server_args, **kwargs)


# --- RuleQueryBuilder (degismedi) ---------------------------------------------

def test_query_builder_combines_items_vat_and_ids():
    query = RuleQueryBuilder().build(VALID_EXTRACTION)
    assert "Klavye" in query
    assert "Mouse" in query
    assert "%20" in query
    assert "%1" in query
    assert "2025123456" in query
    assert "1234567890" in query


def test_query_builder_skips_malformed_items_without_crashing():
    broken = {
        "items": [
            "bu bir dict degil",
            {"description": 123},  # description string degil
            {"description": "Klavye", "vat_rate": "gecersiz"},  # vat_rate sayiya cevrilemiyor
            {"description": "Mouse", "vat_rate": 0.01},  # bu gecerli
        ],
    }
    query = RuleQueryBuilder().build(broken)
    assert "Mouse" in query
    assert "%1" in query
    assert "Klavye" in query  # description gecerli, sadece vat_rate atlanir


def test_query_builder_returns_empty_string_for_unusable_extraction():
    assert RuleQueryBuilder().build({}) == ""
    assert RuleQueryBuilder().build({"foo": "bar"}) == ""
    assert RuleQueryBuilder().build({"items": []}) == ""


# --- RAGAgent kisa devre (MCP'ye hic dokunmaz) --------------------------------

def test_returns_empty_rules_when_raw_extraction_is_none():
    # Kasitli gecersiz bir komut: MCP cagrisi gercekten denenseydi bu patlardi.
    agent = RAGAgent(server_command="komut-yok-boyle-bir-sey")
    result = agent.run(PipelineState(image_path="x.png", raw_extraction=None))
    assert result.retrieved_rules == []


def test_returns_empty_rules_when_query_text_is_empty():
    agent = RAGAgent(server_command="komut-yok-boyle-bir-sey")
    result = agent.run(PipelineState(image_path="x.png", raw_extraction={"foo": "bar"}))
    assert result.retrieved_rules == []


# --- RAGAgent MCP round-trip (sahte server, gercek stdio) ----------------------

def test_run_sends_query_and_top_k_and_fills_retrieved_rules(fake_server_args):
    agent = _agent(fake_server_args, top_k=2)
    extraction = {"items": [{"description": "Klavye"}]}

    result = agent.run(PipelineState(image_path="x.png", raw_extraction=extraction))

    assert len(result.retrieved_rules) == 2
    assert "Ürünler: Klavye." in result.retrieved_rules[0]["text"]
    assert "top_k=2" in result.retrieved_rules[0]["text"]
    assert result.retrieved_rules[0]["source"] == "fake_rule_0"
    assert result.retrieved_rules[0]["score"] == 0.0


def test_run_records_mcp_query_duration_in_execution_trace(fake_server_args):
    agent = _agent(fake_server_args, top_k=1)
    extraction = {"items": [{"description": "Klavye"}]}

    result = agent.run(PipelineState(image_path="x.png", raw_extraction=extraction))

    assert len(result.execution_trace) == 1
    assert result.execution_trace[0]["step"] == "rag.mcp_query"
    assert isinstance(result.execution_trace[0]["duration_ms"], float)
    assert result.execution_trace[0]["duration_ms"] >= 0


def test_run_does_not_mutate_input_state(fake_server_args):
    agent = _agent(fake_server_args, top_k=1)
    original = PipelineState(image_path="x.png", raw_extraction={"items": [{"description": "Klavye"}]})

    result = agent.run(original)

    assert len(result.retrieved_rules) == 1
    assert original.retrieved_rules == []  # model_copy: girdi degismedi


def test_run_raises_rule_query_error_when_tool_reports_error(fake_server_args):
    agent = _agent(fake_server_args)
    extraction = {"items": [{"description": "__bos__"}]}

    state = PipelineState(image_path="x.png", raw_extraction=extraction)
    with pytest.raises(RuleQueryError, match="index_knowledge_base"):
        agent.run(state)


def test_arun_is_the_real_async_path(fake_server_args):
    """app/graph.py'nin rag_node'u run() yerine dogrudan arun()'u await ediyor; bu da calismali.

    pytest-asyncio eklemeden test etmek icin asyncio.run() ile senkron bir test icinden cagrilir."""
    agent = _agent(fake_server_args, top_k=1)
    extraction = {"items": [{"description": "Klavye"}]}

    result = asyncio.run(agent.arun(PipelineState(image_path="x.png", raw_extraction=extraction)))

    assert len(result.retrieved_rules) == 1

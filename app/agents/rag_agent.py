"""
rag_agent.py

Vision Agent'in cikardigi verilerden bir sorgu metni kurar ve app/mcp/rules_server.py'yi
bir MCP stdio subprocess'i olarak baslatip query_rules aracini cagirir. Embed+ChromaDB
sorgulama mantigi artik tamamen server tarafindadir (bkz. app/mcp/rules_server.py);
RAGAgent yalnizca sorgu metni kurma ve MCP client protokolu ile ilgilenir.

RAGAgent de BaseAgent.__init__'i (llm_provider bekler) miras almiyor: bir
EmbeddingProvider/ChromaDB'ye artik dogrudan ihtiyaci yok, MCP server'i nasil
baslatacagini biliyor. BaseAgent.__init__ soyut olmadigi icin bu gecerli bir
ABC kullanimi (ayni gerekce daha once de kullanildi).

Async/sync koprusu: gercek mantik arun()'da (native async, MCP client'i asyncio
uzerine kurulu). run(), BaseAgent'in senkron sozlesmesini korumak icin
asyncio.run(self.arun(state)) ile koprü kurar - run_pipeline_manual.py gibi duz
senkron cagiranlar hic degismeden calismaya devam eder. LangGraph'in async node
destegi oldugu icin app/graph.py'deki rag_node dogrudan arun()'u await eder,
run()'daki asyncio.run() koprusunu (ic ice event loop calistirma riskini) hic
kullanmaz.
"""

import asyncio
import os
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.agents.base import BaseAgent
from app.state import PipelineState


class RuleQueryError(RuntimeError):
    """query_rules MCP araci hata dondurdu (bilgi tabani bos, saglayici hatasi, vb.).

    MCP sinir hattinin otesinde orijinal exception tipini (KnowledgeBaseEmptyError,
    ProviderError alt siniflari...) birebir yeniden kurmak pratik degil; server bu
    hatalari ToolError'a cevirip mesaji koruyarak iletir (bkz. rules_server.py),
    client tarafinda tek, birlesik bu tip olarak yeniden firlatilir."""


class RuleQueryBuilder:
    """Vision Agent ciktisindan (raw_extraction) kisa bir Turkce sorgu metni kurar.

    Girdi bir LLM ciktisi oldugu icin sema garantisi yoktur (Validation Agent henuz
    calismadi); eksik/bozuk alanlar sessizce atlanir, hicbir zaman hata firlatilmaz.
    """

    def build(self, raw_extraction: dict) -> str:
        parts: list[str] = []

        items = raw_extraction.get("items")
        if isinstance(items, list):
            descriptions = [
                item["description"] for item in items
                if isinstance(item, dict) and isinstance(item.get("description"), str) and item["description"]
            ]
            if descriptions:
                parts.append("Ürünler: " + ", ".join(descriptions) + ".")

            vat_percentages = [self._as_percentage(item.get("vat_rate")) for item in items if isinstance(item, dict)]
            vat_percentages = [p for p in vat_percentages if p is not None]
            if vat_percentages:
                parts.append("KDV oranları: " + ", ".join(vat_percentages) + ".")

        invoice_no = raw_extraction.get("invoice_no")
        if isinstance(invoice_no, str) and invoice_no:
            parts.append(f"Fatura numarası: {invoice_no}.")

        seller_tax_no = raw_extraction.get("seller_tax_no")
        if isinstance(seller_tax_no, str) and seller_tax_no:
            parts.append(f"Vergi numarası: {seller_tax_no}.")

        return " ".join(parts)

    @staticmethod
    def _as_percentage(vat_rate) -> str | None:
        try:
            return f"%{float(vat_rate) * 100:.0f}"
        except (TypeError, ValueError):
            return None


class RAGAgent(BaseAgent):
    """Cikarilan fatura verisiyle ilgili kural belgelerini bir MCP server'dan getirir."""

    def __init__(
        self,
        top_k: int = 3,
        query_builder: RuleQueryBuilder | None = None,
        server_command: str | None = None,
        server_args: list[str] | None = None,
    ):
        self.top_k = top_k
        self._query_builder = query_builder or RuleQueryBuilder()
        # Varsayilan: bu Python yorumlayicisiyla "python -m app.mcp.rules_server" calistir.
        self._server_command = server_command or sys.executable
        self._server_args = server_args if server_args is not None else ["-m", "app.mcp.rules_server"]

    def run(self, state: PipelineState) -> PipelineState:
        return asyncio.run(self.arun(state))

    async def arun(self, state: PipelineState) -> PipelineState:
        if not state.raw_extraction:
            # Vision Agent basarisiz olmus ya da henuz calismamis: sorgulanacak bir sey yok,
            # bu bir hata degil; bos liste ile devam edilir.
            return state.model_copy(update={"retrieved_rules": []})

        query_text = self._query_builder.build(state.raw_extraction)
        if not query_text:
            return state.model_copy(update={"retrieved_rules": []})

        start = time.perf_counter()
        retrieved_rules = await self._call_query_rules(query_text)
        mcp_duration_ms = round((time.perf_counter() - start) * 1000, 1)
        trace = [*state.execution_trace, {"step": "rag.mcp_query", "duration_ms": mcp_duration_ms}]

        return state.model_copy(update={"retrieved_rules": retrieved_rules, "execution_trace": trace})

    async def _call_query_rules(self, query_text: str) -> list[dict]:
        # env acikca veriliyor: MCP SDK'sinin stdio client'i varsayilan olarak alt surece
        # ebeveyn ortamini AKTARMIYOR (kisitli, "guvenli" bir ortam kuruyor). MCP server
        # kendi Settings'ini kurdugu icin EMBEDDING_PROVIDER/GEMINI_API_KEY gibi degerleri
        # gormezse varsayilana (ollama) duser.
        #
        # Yerelde bu fark edilmiyordu: alt surec CWD'deki .env dosyasini okuyup dogru
        # degerleri aliyordu. Konteynerde .env YOK (olmamali da - sir imaja girmez), bu
        # yuzden alt surec Ollama'ya gidip "localhost:11434'e ulasilamadi" ile patladi
        # (512 MB testinde canlica yakalandi).
        params = StdioServerParameters(
            command=self._server_command, args=self._server_args, env=os.environ.copy()
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("query_rules", {"query": query_text, "top_k": self.top_k})

        # is_error kontrolu/raise, ic ice async with bloklarinin DISINDA yapilir: anyio'nun
        # TaskGroup'u, bloklarin icinde firlatilan bir exception'i (Python 3.11+ ExceptionGroup
        # gruplamasi yuzunden) sarmalayip cagirana ExceptionGroup olarak iletiyor - canlica
        # yakalandi (pytest.raises(RuleQueryError) esleşmiyordu). Burada, task group tamamen
        # kapandiktan sonra raise edilince RuleQueryError temiz haliyle propagate ediyor.
        if result.is_error:
            message = result.content[0].text if result.content else "query_rules basarisiz"
            raise RuleQueryError(message)
        return result.structured_content["result"]

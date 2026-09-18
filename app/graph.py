"""
graph.py

Dort ajani (Vision, RAG, Validation, Report) gercek bir LangGraph StateGraph'i
olarak sabit sirayla zincirler: Vision -> RAG -> Validation -> Report -> END.
Henuz kosullu gecis/retry yok; hepsi tek yonlu, sabit kenarlarla (add_edge) baglanir.

Node fonksiyonlari kasitli olarak sade: state alir, ilgili agent'in run()'ini
cagirir, guncel state'i dondurur - donusum/mantik icermez, tum is agent'larda.
Tek istisna: her node, agent'in suresini olcup state.execution_trace'e ekleyen
ortak _timed() sarmalayicisindan geciyor (bkz. asagisi) - bu da "donusum" degil,
tum node'larda ayni sekilde tekrarlanan, tek yerde tutulan bir olcum katmani.

RAGAgent artik bir MCP client'i (bkz. app/agents/rag_agent.py): gercek async
mantigi arun()'dadir. rag_node bu yuzden async tanimlanip dogrudan arun()'u
await eder (RAGAgent.run()'daki senkron asyncio.run() koprusunu atlayarak, ic
ice event loop calistirma riskinden kacinir). LangGraph, bir graph'ta tek bir
async node bile olsa .invoke()/.stream() ile calismayi reddedip .ainvoke()/
.astream() istiyor (canlica dogrulandi); dolayisiyla bu graph'i calistiranlar
(scripts/run_pipeline_graph.py, tests/test_graph.py) async API kullanmali.
"""

import time

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session, sessionmaker

from app.agents.rag_agent import RAGAgent
from app.agents.report_agent import ReportAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.vision_agent import VisionAgent
from app.config import Settings
from app.providers.base import LLMProvider
from app.providers.factory import get_llm_provider
from app.state import PipelineState


def _record_duration(state: PipelineState, step: str, start: float) -> PipelineState:
    """execution_trace'e {"step","duration_ms"} ekler; final_report zaten olusmussa
    (yalniz "report" adiminda gecerli) icindeki execution_trace'i de gunceller -
    ReportAgent kendi suresini bilemez (henuz olculmedi), bu yuzden final_report'un
    kendi execution_trace kopyasi burada, agent'tan sonra tamamlanir."""
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    trace = [*state.execution_trace, {"step": step, "duration_ms": duration_ms}]
    updates: dict = {"execution_trace": trace}
    if state.final_report is not None:
        updates["final_report"] = {**state.final_report, "execution_trace": trace}
    return state.model_copy(update=updates)


def build_pipeline_graph(
    settings: Settings | None = None,
    llm_provider: LLMProvider | None = None,
    rag_agent: RAGAgent | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> CompiledStateGraph:
    """Derlenmis (compile edilmis) pipeline graph'ini kurar.

    llm_provider verilmezse factory'den (.env'deki ayarlara gore) kurulur. rag_agent
    verilmezse varsayilan RAGAgent() kurulur (calistiginda gercek bir MCP subprocess'i
    baslatir); testlerde gercek Ollama'ya/subprocess'e hic dokunmadan sahte/stub bir
    RAGAgent dogrudan buraya enjekte edilebilir. session_factory verilmezse ReportAgent'in
    price_history'si hep bos liste doner (DB'ye hic dokunulmaz) - bunu isteyen cagiranlar
    (app/api/main.py, scripts/save_pipeline_result.py) kendi session_factory'lerini
    acikca gecer."""
    settings = settings or Settings()
    llm_provider = llm_provider or get_llm_provider(settings)
    rag_agent = rag_agent or RAGAgent(top_k=settings.RAG_TOP_K)

    vision_agent = VisionAgent(llm_provider)
    validation_agent = ValidationAgent(format_profile=settings.FORMAT_PROFILE)
    report_agent = ReportAgent(session_factory=session_factory)

    def vision_node(state: PipelineState) -> PipelineState:
        start = time.perf_counter()
        return _record_duration(vision_agent.run(state), "vision", start)

    async def rag_node(state: PipelineState) -> PipelineState:
        start = time.perf_counter()
        return _record_duration(await rag_agent.arun(state), "rag", start)

    def validation_node(state: PipelineState) -> PipelineState:
        start = time.perf_counter()
        return _record_duration(validation_agent.run(state), "validation", start)

    def report_node(state: PipelineState) -> PipelineState:
        start = time.perf_counter()
        return _record_duration(report_agent.run(state), "report", start)

    builder = StateGraph(PipelineState)
    builder.add_node("vision", vision_node)
    builder.add_node("rag", rag_node)
    builder.add_node("validation", validation_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "vision")
    builder.add_edge("vision", "rag")
    builder.add_edge("rag", "validation")
    builder.add_edge("validation", "report")
    builder.add_edge("report", END)

    return builder.compile()

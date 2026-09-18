"""
main.py

Pipeline'i (app/graph.py) HTTP uzerinden sunan FastAPI uygulamasi.

Onemli mimari nokta: RAGAgent.run() icinde asyncio.run(self.arun(state)) ile senkron
koprü kurar (bkz. rag_agent.py). FastAPI zaten kendi event loop'unda calistigi icin
async bir endpoint'ten run() cagirmak "asyncio.run() cannot be called from a running
event loop" ile patlar (canlica dogrulandi). Bu yuzden endpoint'ler run()'a hic
ugramaz: dogrudan build_pipeline_graph()'in dondurdugu, zaten async destekli
graph.ainvoke()'u cagirir - graph.ainvoke() de rag_node uzerinden RAGAgent.arun()'u
dogrudan await eder (bkz. graph.py), ic ice event loop calistirmaz.

create_app(), projenin build_pipeline_graph()/build_server() deseniyle aynidir:
Settings bu modulun icinde degil, sadece factory fonksiyonuna parametre olarak
okunur. init_db() cagrisi app kurulurken degil, FastAPI'nin lifespan'inda calisir -
boylece create_app()'i cagirmak/bu modulu import etmek tek basina canli bir
veritabani (gercek Postgres) gerektirmez; yalnizca sunucu/TestClient gercekten
baslarken (uvicorn ile ya da testte "with TestClient(app) as client:") devreye girer.
"""

import datetime
import json
import tempfile
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session, sessionmaker

from app.agents.rag_agent import RAGAgent
from app.config import Settings
from app.db.models import InvoiceRecord
from app.db.session import get_session_factory, init_db
from app.graph import build_pipeline_graph
from app.providers.base import LLMProvider
from app.state import PipelineState

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


class InvoiceSummary(BaseModel):
    """GET /invoices liste gorunumu: full_report/anomalies haric, ozet alanlar."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    image_filename: str
    processed_at: datetime.datetime
    invoice_no: str | None
    seller_name: str | None
    grand_total: float | None
    is_valid: bool
    anomaly_count: int


class InvoiceDetail(InvoiceSummary):
    """GET /invoices/{id}: tam detay, full_report dahil."""

    anomalies: list[dict]
    full_report: dict


def get_db(request: Request) -> Generator[Session, None, None]:
    """Her istekte yeni bir session acar, istek bitince kapatir."""
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    with session_factory() as session:
        yield session


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


router = APIRouter()

GraphDep = Annotated[CompiledStateGraph, Depends(get_graph)]
DbDep = Annotated[Session, Depends(get_db)]


@router.post("/invoices", status_code=201)
async def create_invoice(file: Annotated[UploadFile, File()], graph: GraphDep, db: DbDep) -> dict:
    suffix = Path(file.filename).suffix if file.filename else ""
    with tempfile.NamedTemporaryFile(suffix=suffix or ".png", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        raw_result = await graph.ainvoke(PipelineState(image_path=tmp_path))
        # graph.ainvoke() duz bir dict dondurur; None kalan alanlar dusebilir
        # (bkz. run_pipeline_graph.py'deki ayni not); PipelineState(**raw_result) geri tamamlar.
        final_state = PipelineState(**raw_result)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    final_report = final_state.final_report
    record = InvoiceRecord.from_final_report(file.filename or Path(tmp_path).name, final_report)
    db.add(record)
    db.commit()
    db.refresh(record)

    return {"id": record.id, **final_report}


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_pipeline_events(
    graph: CompiledStateGraph,
    session_factory: sessionmaker[Session],
    image_filename: str,
    tmp_path: str,
) -> AsyncIterator[str]:
    """graph.astream(stream_mode='updates') adimlarini SSE olaylarina cevirir.

    Her node'un donusu (bkz. app/graph.py'deki node fonksiyonlari) zaten o anki TAM
    PipelineState kopyasi oldugu icin, "updates" akisindaki her adim aslinda o ana
    kadarki butun state'tir (canlica dogrulandi) - "report" adiminin state_dict'i
    zaten dolu final_report'u tasir, ayrica bir ainvoke() cagrisina gerek yok.

    DB session'i burada Depends(get_db) yerine dogrudan session_factory ile aciliyor:
    bu generator, route fonksiyonu StreamingResponse'u dondurdukten SONRA da calismaya
    devam ediyor, ve yield'li bir FastAPI dependency'nin bu durumda ne zaman kapatilacagi
    belirsiz bir alan - session_factory'yi dogrudan kullanmak bu belirsizligi tamamen
    ortadan kaldirir.
    """
    final_report: dict | None = None
    try:
        async for update in graph.astream(PipelineState(image_path=tmp_path), stream_mode="updates"):
            (node_name, state_dict), = update.items()
            yield _sse_event({"step": node_name, "status": "done"})
            if node_name == "report":
                final_report = state_dict["final_report"]
    except Exception as exc:  # noqa: BLE001 -- SSE sinirinda kasitli: herhangi bir node
        # hatasi (saglayici, MCP, ...) burada bir "error" olayina cevrilip stream duzgunce
        # kapatilmali, yeniden raise edilmemeli. Node'un firlattigi exception async for'a
        # oldugu gibi (sarmalanmadan) geliyor (canlica dogrulandi).
        yield _sse_event({"step": "error", "message": str(exc)})
        return
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    record = InvoiceRecord.from_final_report(image_filename, final_report)
    with session_factory() as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        record_id = record.id

    yield _sse_event({"step": "final", "report": final_report, "id": record_id})


@router.post("/invoices/stream")
async def create_invoice_stream(request: Request, file: Annotated[UploadFile, File()]) -> StreamingResponse:
    """POST /invoices ile ayni isi yapar, ama sonucu tek seferde degil, her ajan
    bitince bir SSE olayiyla akitir (bkz. frontend/upload.html)."""
    suffix = Path(file.filename).suffix if file.filename else ""
    with tempfile.NamedTemporaryFile(suffix=suffix or ".png", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    generator = _stream_pipeline_events(
        graph=request.app.state.graph,
        session_factory=request.app.state.session_factory,
        image_filename=file.filename or Path(tmp_path).name,
        tmp_path=tmp_path,
    )
    return StreamingResponse(generator, media_type="text/event-stream")


MAX_BULK_FILES = 10


async def _stream_bulk_events(
    graph: CompiledStateGraph,
    session_factory: sessionmaker[Session],
    files: list[tuple[str, str]],
) -> AsyncIterator[str]:
    """Birden fazla faturayi SIRAYLA isler (ayni anda degil - Gemini kotasini korumak
    icin). files: [(orijinal_dosya_adi, gecici_yol), ...]. Her dosya kendi try/except'i
    icinde izole edilir: biri hata verirse digerlerinin islenmesi etkilenmez (bkz.
    _stream_pipeline_events'teki tek-dosya versiyonuyla ayni SSE-sinir gerekcesi)."""
    total = len(files)
    processed = 0
    failed = 0
    try:
        for index, (filename, tmp_path) in enumerate(files, start=1):
            try:
                raw_result = await graph.ainvoke(PipelineState(image_path=tmp_path))
                final_state = PipelineState(**raw_result)
                final_report = final_state.final_report

                record = InvoiceRecord.from_final_report(filename, final_report)
                with session_factory() as session:
                    session.add(record)
                    session.commit()
                    session.refresh(record)
                    record_id = record.id

                processed += 1
                yield _sse_event({
                    "file": filename, "index": index, "total": total,
                    "status": "done", "report": final_report, "id": record_id,
                })
            except Exception as exc:  # noqa: BLE001 -- dosya bazinda izolasyon: biri
                # patlarsa digerlerinin islenmeye devam etmesi gerekiyor.
                failed += 1
                yield _sse_event({
                    "file": filename, "index": index, "total": total,
                    "status": "error", "message": str(exc),
                })
    finally:
        for _, tmp_path in files:
            Path(tmp_path).unlink(missing_ok=True)

    yield _sse_event({"step": "batch_complete", "processed": processed, "failed": failed})


@router.post("/invoices/bulk")
async def create_invoices_bulk(
    request: Request, files: Annotated[list[UploadFile], File()]
) -> StreamingResponse:
    """En fazla MAX_BULK_FILES dosya alir, sirayla isler (bkz. _stream_bulk_events).
    Sinir asilirsa hicbir dosyaya dokunmadan (StreamingResponse hic baslamadan) 422 doner."""
    if len(files) > MAX_BULK_FILES:
        raise HTTPException(status_code=422, detail=f"En fazla {MAX_BULK_FILES} fatura aynı anda yüklenebilir.")

    saved: list[tuple[str, str]] = []
    for file in files:
        suffix = Path(file.filename).suffix if file.filename else ""
        with tempfile.NamedTemporaryFile(suffix=suffix or ".png", delete=False) as tmp:
            tmp.write(await file.read())
            saved.append((file.filename or Path(tmp.name).name, tmp.name))

    generator = _stream_bulk_events(
        graph=request.app.state.graph,
        session_factory=request.app.state.session_factory,
        files=saved,
    )
    return StreamingResponse(generator, media_type="text/event-stream")


@router.get("/invoices", response_model=list[InvoiceSummary])
def list_invoices(
    db: DbDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[InvoiceRecord]:
    return (
        db.query(InvoiceRecord)
        .order_by(InvoiceRecord.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


@router.get("/invoices/{invoice_id}", response_model=InvoiceDetail)
def get_invoice(invoice_id: int, db: DbDep) -> InvoiceRecord:
    record = db.get(InvoiceRecord, invoice_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"invoice_id={invoice_id} bulunamadi")
    return record


def create_app(
    settings: Settings | None = None,
    llm_provider: LLMProvider | None = None,
    rag_agent: RAGAgent | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    """FastAPI uygulamasini kurar. Hicbir parametre verilmezse .env'deki gercek
    ayarlarla (gercek LLM saglayicisi, gercek RAGAgent/MCP, gercek Postgres) calisir;
    testlerde tumu sahte/gecici olanlarla degistirilebilir."""
    settings = settings or Settings()
    session_factory = session_factory or get_session_factory(settings=settings)
    graph = build_pipeline_graph(
        settings=settings, llm_provider=llm_provider, rag_agent=rag_agent, session_factory=session_factory,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(settings=settings)
        yield

    app = FastAPI(title="Invoice Vision Agent API", lifespan=lifespan)
    app.state.graph = graph
    app.state.session_factory = session_factory
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    return app


app = create_app()

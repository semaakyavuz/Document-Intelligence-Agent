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
import tempfile
from collections.abc import Generator
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
    graph = build_pipeline_graph(settings=settings, llm_provider=llm_provider, rag_agent=rag_agent)
    session_factory = session_factory or get_session_factory(settings=settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(settings=settings)
        yield

    app = FastAPI(title="Invoice Vision Agent API", lifespan=lifespan)
    app.state.graph = graph
    app.state.session_factory = session_factory
    app.include_router(router)
    return app


app = create_app()

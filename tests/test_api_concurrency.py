"""
Ayni anda calisan pipeline sayisini sinirlayan semaforu (app/api/main.py, sinir
Settings.MAX_CONCURRENT_PIPELINES) ve slot beklenirken gonderilen "queued" SSE
olayini test eder.

Neden TestClient degil: TestClient senkron, yani ayni anda birden fazla istek ayakta
tutup "3.'su kuyrukta bekledi mi" sorusunu deterministik olarak olcemiyor. Testler bu
yuzden SSE generator'larini (_stream_pipeline_events / _stream_bulk_events) dogrudan
asyncio ile surer; olculen sey uretim kodunun tam kendisi, kopyasi degil.

Hicbir dis servise dokunmaz: graph, session ve semafor sahte/yerel nesneler.
"""

import asyncio
import io
import json

import pytest
from fastapi.datastructures import UploadFile
from pydantic import ValidationError

from app.api.main import (
    QUEUED_MESSAGE,
    _stream_bulk_events,
    _stream_pipeline_events,
    create_app,
    create_invoice,
)
from app.config import Settings

# Sinir artik Settings'ten geliyor. Burada ILAN EDILEN VARSAYILAN okunuyor, ortamda
# cozulmus deger degil: Settings(_env_file=None) bile ortam degiskenlerini okur, ve
# bu makinede MAX_CONCURRENT_PIPELINES=6 tanimliysa asagidaki 5 istekli senaryolar
# hic kuyruk olusturmadan "gecmis" gorunurdu - test sessizce bir sey olcmez olurdu.
MAX_CONCURRENT_PIPELINES = Settings.model_fields["MAX_CONCURRENT_PIPELINES"].default

FINAL_REPORT = {
    "invoice_no": "2025123456",
    "seller_name": "Test A.Ş.",
    "grand_total": 1200.0,
    "is_valid": True,
    "anomaly_count": 0,
    "anomalies": [],
    "risk_score": 10,
}


class ConcurrencyTracker:
    """Ayni anda kac pipeline'in ayakta oldugunu ve tepe degerini kaydeder."""

    def __init__(self):
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        self.current -= 1


class CountingSemaphore(asyncio.Semaphore):
    """Kac kez acquire edildigini sayar - "toplu yukleme tek slot kullaniyor mu"
    sorusunu dolayli degil dogrudan olcmek icin."""

    def __init__(self, value: int):
        super().__init__(value)
        self.acquire_count = 0

    async def acquire(self) -> bool:
        self.acquire_count += 1
        return await super().acquire()


class FakeGraph:
    """graph.astream/ainvoke sozlesmesini taklit eder; gercek ajanlara hic dokunmaz.

    Her cagri tracker'a giris/cikis bildirir, boylece eszamanlilik olculebilir.
    Adimlar arasinda await veriyor ki birden fazla istek gercekten ic ice gecebilsin.
    """

    def __init__(self, tracker: ConcurrencyTracker | None = None, step_delay: float = 0.01):
        self.tracker = tracker or ConcurrencyTracker()
        self.step_delay = step_delay

    async def astream(self, state, stream_mode="updates"):
        self.tracker.enter()
        try:
            for node in ("vision", "rag", "validation"):
                await asyncio.sleep(self.step_delay)
                yield {node: {}}
            await asyncio.sleep(self.step_delay)
            yield {"report": {"final_report": dict(FINAL_REPORT)}}
        finally:
            self.tracker.leave()

    async def ainvoke(self, state):
        self.tracker.enter()
        try:
            await asyncio.sleep(self.step_delay * 4)
            return {"image_path": state.image_path, "final_report": dict(FINAL_REPORT)}
        finally:
            self.tracker.leave()


class FakeSession:
    """session_factory() sozlesmesinin (context manager + add/commit/refresh) sahtesi;
    veritabanina hic dokunulmuyor, refresh sadece bir id atar."""

    def __init__(self):
        self.added = []

    def __call__(self) -> "FakeSession":
        return self

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def add(self, record) -> None:
        self.added.append(record)

    def commit(self) -> None:
        pass

    def refresh(self, record) -> None:
        record.id = len(self.added)


def _payload(event_text: str) -> dict:
    assert event_text.startswith("data:")
    return json.loads(event_text[len("data:") :].strip())


def _stream(graph, slots, tmp_path, name="fatura.png"):
    return _stream_pipeline_events(
        graph=graph,
        session_factory=FakeSession(),
        image_filename=name,
        tmp_path=str(tmp_path / name),
        slots=slots,
    )


def _bulk(graph, slots, tmp_path, count):
    files = [(f"f{i}.png", str(tmp_path / f"f{i}.png")) for i in range(count)]
    return _stream_bulk_events(
        graph=graph, session_factory=FakeSession(), files=files, slots=slots
    )


async def _drain(generator) -> list[dict]:
    return [_payload(event) async for event in generator]


# ---------------------------------------------------------------- tek fatura


def test_stream_emits_no_queued_event_when_a_slot_is_free(tmp_path):
    slots = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)
    events = asyncio.run(_drain(_stream(FakeGraph(), slots, tmp_path)))

    assert [e["step"] for e in events] == ["vision", "rag", "validation", "report", "final"]


def test_stream_emits_queued_event_before_waiting_when_all_slots_busy(tmp_path):
    async def scenario():
        slots = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)
        for _ in range(MAX_CONCURRENT_PIPELINES):
            await slots.acquire()

        generator = _stream(FakeGraph(), slots, tmp_path)
        first = _payload(await anext(generator))
        # Slot dolu oldugu icin akis burada bekliyor: bir slot bosaltilinca devam etmeli.
        for _ in range(MAX_CONCURRENT_PIPELINES):
            slots.release()
        await _drain(generator)
        return first

    first = asyncio.run(scenario())

    assert first["step"] == "queued"
    assert first["message"] == QUEUED_MESSAGE


def test_stream_continues_normally_after_queued_event(tmp_path):
    """"queued" akisi bozmuyor: slot acilinca ayni istek normal adimlarini tamamliyor."""

    async def scenario():
        slots = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)
        for _ in range(MAX_CONCURRENT_PIPELINES):
            await slots.acquire()

        generator = _stream(FakeGraph(), slots, tmp_path)
        await anext(generator)
        for _ in range(MAX_CONCURRENT_PIPELINES):
            slots.release()
        return await _drain(generator)

    rest = asyncio.run(scenario())

    assert [e["step"] for e in rest] == ["vision", "rag", "validation", "report", "final"]
    assert rest[-1]["report"]["invoice_no"] == "2025123456"


def test_at_most_two_pipelines_run_concurrently(tmp_path):
    """Asil koruma: 5 istek ayni anda gelse bile ayakta olan pipeline sayisi 2'yi asmaz."""
    tracker = ConcurrencyTracker()

    async def scenario():
        slots = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)
        graph = FakeGraph(tracker=tracker)
        await asyncio.gather(
            *(_drain(_stream(graph, slots, tmp_path, f"f{i}.png")) for i in range(5))
        )

    asyncio.run(scenario())

    assert tracker.peak == MAX_CONCURRENT_PIPELINES
    assert tracker.current == 0


def test_all_queued_requests_eventually_complete(tmp_path):
    """Fazlalik reddedilmiyor, kuyrukta bekleyip sonunda tamamlaniyor."""

    async def scenario():
        slots = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)
        graph = FakeGraph()
        return await asyncio.gather(
            *(_drain(_stream(graph, slots, tmp_path, f"f{i}.png")) for i in range(5))
        )

    results = asyncio.run(scenario())

    assert len(results) == 5
    assert all(events[-1]["step"] == "final" for events in results)


def test_queued_event_is_emitted_for_the_requests_that_had_to_wait(tmp_path):
    """5 istekten 2'si hemen slot buluyor, kalanlar "queued" goruyor."""

    async def scenario():
        slots = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)
        graph = FakeGraph()
        return await asyncio.gather(
            *(_drain(_stream(graph, slots, tmp_path, f"f{i}.png")) for i in range(5))
        )

    results = asyncio.run(scenario())
    queued = [events for events in results if events[0]["step"] == "queued"]

    assert len(queued) == 5 - MAX_CONCURRENT_PIPELINES


# ---------------------------------------------------------------- toplu yukleme


def test_bulk_acquires_exactly_one_slot_for_the_whole_batch(tmp_path):
    """Toplu yukleme dosyalari SIRAYLA isliyor, yani ayni anda tek pipeline ayakta -
    semafor da dosya basina degil parti basina bir kez alinmali."""

    async def scenario():
        slots = CountingSemaphore(MAX_CONCURRENT_PIPELINES)
        events = await _drain(_bulk(FakeGraph(), slots, tmp_path, count=5))
        return slots.acquire_count, events

    acquire_count, events = asyncio.run(scenario())

    assert acquire_count == 1
    assert events[-1] == {"step": "batch_complete", "processed": 5, "failed": 0}


def test_bulk_never_runs_more_than_one_pipeline_at_a_time(tmp_path):
    tracker = ConcurrencyTracker()
    asyncio.run(_drain(_bulk(FakeGraph(tracker=tracker), asyncio.Semaphore(2), tmp_path, count=5)))

    assert tracker.peak == 1


def test_bulk_leaves_a_slot_free_for_a_single_upload(tmp_path):
    """Parti tek slot tuttugu icin (ikisinden biri) es zamanli tek fatura yuklemesi
    kuyruga girmeden calisabiliyor - "queued" olayi hic gonderilmiyor."""

    async def scenario():
        slots = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)
        graph = FakeGraph()
        bulk_task = asyncio.create_task(_drain(_bulk(graph, slots, tmp_path, count=5)))
        await asyncio.sleep(0.02)  # parti slotunu almis olsun
        single = await _drain(_stream(graph, slots, tmp_path))
        await bulk_task
        return single

    single = asyncio.run(scenario())

    assert single[0]["step"] == "vision"


def test_bulk_emits_queued_event_when_all_slots_busy(tmp_path):
    async def scenario():
        slots = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)
        for _ in range(MAX_CONCURRENT_PIPELINES):
            await slots.acquire()

        generator = _bulk(FakeGraph(), slots, tmp_path, count=2)
        first = _payload(await anext(generator))
        for _ in range(MAX_CONCURRENT_PIPELINES):
            slots.release()
        return first, await _drain(generator)

    first, rest = asyncio.run(scenario())

    assert first == {"step": "queued", "message": QUEUED_MESSAGE}
    assert rest[-1]["processed"] == 2


# ---------------------------------------------------------------- POST /invoices


def test_non_streaming_endpoint_also_respects_the_limit(tmp_path):
    """SSE'siz uc "queued" gonderemiyor ama semafora bagli olmak zorunda: aksi halde
    bellek siniri bu uctan asilabilirdi."""
    tracker = ConcurrencyTracker()

    async def scenario():
        slots = asyncio.Semaphore(1)
        graph = FakeGraph(tracker=tracker)

        async def one(index: int):
            upload = UploadFile(file=io.BytesIO(b"fake"), filename=f"f{index}.png")
            return await create_invoice(file=upload, graph=graph, db=FakeSession(), slots=slots)

        return await asyncio.gather(one(0), one(1), one(2))

    results = asyncio.run(scenario())

    assert tracker.peak == 1
    assert all(r["invoice_no"] == "2025123456" for r in results)


# ---------------------------------------------------------------- ayar baglantisi


def _slot_count(app) -> int:
    """Semaforun kac slotu oldugunu YALNIZCA public API ile sayar: bloklamadan
    alinabilen slot sayisi = kapasite. locked(), "artik hemen alinamaz" demektir."""

    async def count():
        slots = app.state.pipeline_slots
        acquired = 0
        while not slots.locked():
            await slots.acquire()
            acquired += 1
        return acquired

    return asyncio.run(count())


def _app_with_limit(tmp_path, limit: int):
    """create_app() hicbir saglayici override'i almadan kurulur: varsayilan
    LLM_PROVIDER=ollama yalnizca nesne kuruyor, ag istegi yapmiyor; lifespan'a
    (init_db / indeksleme) hic girilmiyor cunku TestClient acilmiyor."""
    settings = Settings(
        _env_file=None,
        MAX_CONCURRENT_PIPELINES=limit,
        DATABASE_URL=f"sqlite:///{tmp_path / 'test.db'}",
    )
    return create_app(settings=settings)


def test_create_app_takes_the_limit_from_settings(tmp_path):
    assert _slot_count(_app_with_limit(tmp_path, 3)) == 3
    assert _slot_count(_app_with_limit(tmp_path, 1)) == 1


def test_create_app_slot_count_always_matches_its_settings(tmp_path):
    """Sinir acikca verilmediginde de baglanti korunuyor: semafor, uygulamaya verilen
    Settings nesnesinin degeriyle ayni. Sabit bir sayiya degil o nesneye karsi
    karsilastiriliyor, boylece ortamda deger tanimliysa test yanlis kirmizi olmuyor."""
    settings = Settings(_env_file=None, DATABASE_URL=f"sqlite:///{tmp_path / 'test.db'}")
    assert _slot_count(create_app(settings=settings)) == settings.MAX_CONCURRENT_PIPELINES


def test_limit_below_one_is_rejected_before_the_server_starts():
    """0 verilirse her istek sonsuza kadar beklerdi; ayar okunurken reddedilmeli."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_CONCURRENT_PIPELINES=0)

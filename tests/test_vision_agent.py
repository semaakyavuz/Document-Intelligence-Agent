"""
VisionAgent'in JSON parse mantigini gercek LLM'e baglanmadan test eder.

FakeLLMProvider sabit bir metin dondurur ve kendisine gelen image_path'i kaydeder;
boylece hem parse senaryolari hem de agent'in gorseli dogru iletip iletmedigi kontrol edilir.
"""

import pytest

from app.agents.vision_agent import JsonResponseParser, VisionAgent
from app.providers.base import LLMProvider, ProviderResponseError
from app.state import PipelineState

VALID_JSON = '{"invoice_no": "2025864544", "grand_total": 20789.03, "items": [{"description": "Klavye", "quantity": 2}]}'


class FakeLLMProvider(LLMProvider):
    """Aga cikmaz; verilen cevabi aynen dondurur, cagrilari kaydeder."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[tuple[str, str | None, int | None]] = []

    def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
        self.calls.append((prompt, image_path, max_tokens))
        return self.response_text


def _run_agent(response_text: str) -> PipelineState:
    return VisionAgent(FakeLLMProvider(response_text)).run(PipelineState(image_path="data/golden/images/x.png"))


def test_valid_json_fills_raw_extraction():
    state = _run_agent(VALID_JSON)
    assert state.raw_extraction["invoice_no"] == "2025864544"
    assert state.raw_extraction["grand_total"] == 20789.03
    assert state.validation_errors == []


def test_fenced_json_block_is_unwrapped():
    state = _run_agent(f"Here is the extracted data:\n```json\n{VALID_JSON}\n```\nLet me know if you need more.")
    assert state.raw_extraction["invoice_no"] == "2025864544"
    assert state.validation_errors == []


def test_json_embedded_in_prose_is_extracted():
    state = _run_agent(f"Sure! The invoice contains: {VALID_JSON} Hope this helps.")
    assert state.raw_extraction["items"][0]["description"] == "Klavye"


def test_broken_json_is_reported_not_raised():
    state = _run_agent('{"invoice_no": "2025", "grand_total": ')
    assert state.raw_extraction is None
    assert len(state.validation_errors) == 1
    assert state.validation_errors[0].startswith("VisionAgent:")


def test_json_that_is_not_an_object_is_rejected():
    state = _run_agent('["invoice_no", "grand_total"]')
    assert state.raw_extraction is None
    assert "dict" in state.validation_errors[0]


def test_agent_passes_image_path_and_does_not_mutate_input():
    provider = FakeLLMProvider(VALID_JSON)
    original = PipelineState(image_path="data/golden/images/invoice_0003_aug04.png", validation_errors=["onceki hata"])
    result = VisionAgent(provider).run(original)

    prompt, image_path, max_tokens = provider.calls[0]
    assert image_path == "data/golden/images/invoice_0003_aug04.png"
    assert "invoice_no" in prompt and "grand_total" in prompt  # prompt semayi iceriyor
    assert max_tokens == VisionAgent.MAX_OUTPUT_TOKENS  # donguye giren model dakikalarca beklenmesin
    assert original.raw_extraction is None  # model_copy: girdi nesnesi degismedi
    assert result.validation_errors == ["onceki hata"]  # basarili parse eski hatalari silmez


def test_provider_error_is_not_swallowed():
    """Saglayici hatasi (baglanti yok, bozuk govde...) VisionAgent tarafindan yutulmaz;
    bu bir parse hatasi degil, cagirana (script/orchestrator) bildirilmesi gereken altyapi hatasidir."""

    class FailingProvider(LLMProvider):
        def generate(self, prompt: str, image_path: str | None = None, max_tokens: int | None = None) -> str:
            raise ProviderResponseError("Ollama JSON olmayan govde dondu")

    with pytest.raises(ProviderResponseError):
        VisionAgent(FailingProvider()).run(PipelineState(image_path="data/golden/images/x.png"))


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Cevap: {"a": 1}.',
        '  \n{"a": 1}\n  ',
    ],
)
def test_parser_handles_common_wrappings(text):
    assert JsonResponseParser().parse(text) == {"a": 1}


def test_parser_raises_on_no_json():
    with pytest.raises(ValueError, match="JSON"):
        JsonResponseParser().parse("Uzgunum, gorseli okuyamadim.")

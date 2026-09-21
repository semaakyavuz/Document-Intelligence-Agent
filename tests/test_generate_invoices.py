"""
generate_invoices.py'nin InvoiceRenderer sablonlarini test eder.

Piksel icerigini degil, "gorsel gercekten olusuyor mu, dogru boyutta mi ve
ground-truth semasi sablondan bagimsiz mi kaliyor mu" seviyesini dogrular.
"""

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_invoices.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_invoices", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen_module():
    return _load_module()


@pytest.fixture(scope="module")
def renderer(gen_module):
    return gen_module.InvoiceRenderer()


@pytest.fixture
def invoice(gen_module):
    return gen_module.InvoiceFactory().generate()


# --- Render sablonlari -------------------------------------------------------

@pytest.mark.parametrize("method_name", ["render_classic", "render_compact", "render_letterhead"])
def test_render_produces_correct_size_rgb_image(renderer, invoice, method_name):
    img = getattr(renderer, method_name)(invoice)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (renderer.WIDTH, renderer.HEIGHT)


@pytest.mark.parametrize("method_name", ["render_compact", "render_letterhead"])
def test_render_does_not_mutate_ground_truth(renderer, invoice, method_name):
    """Sablon degisikligi yalnizca gorseli etkilemeli; ground truth semasi ayni kalmali."""
    before = invoice.to_ground_truth()
    getattr(renderer, method_name)(invoice)
    after = invoice.to_ground_truth()
    assert before == after


def test_all_templates_yield_identical_ground_truth_schema(renderer, invoice):
    schemas = {}
    for name in ("render_classic", "render_compact", "render_letterhead"):
        getattr(renderer, name)(invoice)
        schemas[name] = set(invoice.to_ground_truth().keys())
    assert schemas["render_classic"] == schemas["render_compact"]
    assert schemas["render_compact"] == schemas["render_letterhead"]


# --- --template CLI parametresi ----------------------------------------------

@pytest.mark.parametrize("value", ["classic", "compact", "letterhead", "random"])
def test_template_cli_accepts_valid_choices(gen_module, value):
    args = gen_module.build_arg_parser().parse_args(["--template", value])
    assert args.template == value


def test_template_cli_rejects_invalid_choice(gen_module):
    parser = gen_module.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--template", "gecersiz"])


def test_template_cli_default_is_random(gen_module):
    args = gen_module.build_arg_parser().parse_args([])
    assert args.template == "random"


def test_templates_registry_matches_cli_choices(gen_module):
    """TEMPLATES sozlugu main()'in secebilecegi sablonlarla birebir eslesmeli."""
    parser_choices = set(gen_module.build_arg_parser()._option_string_actions["--template"].choices)
    assert set(gen_module.TEMPLATES) == parser_choices - {"random"}

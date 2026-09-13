"""
augment_invoices.py icin uctan uca ama dis servis gerektirmeyen testler.

generate_invoices.py'yi calistirmaya gerek kalmadan, kucuk sahte "temiz"
fatura goruntuleri ve minimal ground-truth JSON'lari uretip augment_invoices.py'yi
subprocess olarak bunlar uzerinde calistirir; goruntu piksellerini/kalitesini
degil, yalnizca uretilen dosya SAYISINI ve ISIMLERINI dogrular.
"""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "augment_invoices.py"


def _make_fake_dataset(base_dir: Path, count: int) -> list[str]:
    """count kadar kucuk sahte 'temiz' fatura goruntusu + JSON etiketi olusturur."""
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stems = []
    for i in range(1, count + 1):
        stem = f"invoice_{i:04d}"
        Image.new("RGB", (300, 400), "white").save(images_dir / f"{stem}.png")
        (labels_dir / f"{stem}.json").write_text(
            json.dumps({"invoice_no": stem, "grand_total": 100.0 * i}, ensure_ascii=False),
            encoding="utf-8",
        )
        stems.append(stem)
    return stems


def _run_augment_script(input_dir: Path, output_dir: Path, golden_dir: Path,
                         per_image: int, golden_count: int, seed: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--golden-dir", str(golden_dir),
            "--per-image", str(per_image),
            "--golden-count", str(golden_count),
            "--seed", str(seed),
        ],
        capture_output=True,
        text=True,
    )


def test_augment_invoices_generates_expected_files(tmp_path):
    input_dir = tmp_path / "synthetic"
    output_dir = tmp_path / "synthetic" / "augmented"
    golden_dir = tmp_path / "golden"

    clean_stems = _make_fake_dataset(input_dir, count=3)
    per_image = 2
    golden_count = 4

    result = _run_augment_script(input_dir, output_dir, golden_dir, per_image, golden_count, seed=123)
    assert result.returncode == 0, result.stderr

    aug_images = sorted((output_dir / "images").glob("*.png"))
    aug_labels = sorted((output_dir / "labels").glob("*.json"))

    expected_total = len(clean_stems) * per_image
    assert len(aug_images) == expected_total
    assert len(aug_labels) == expected_total

    # Her augment edilmis goruntunun tam olarak eslesen bir etiketi olmali.
    image_stems = {p.stem for p in aug_images}
    label_stems = {p.stem for p in aug_labels}
    assert image_stems == label_stems

    # Adlandirma deseni: invoice_000X_augNN
    for stem in image_stems:
        assert "_aug" in stem

    golden_images = list((golden_dir / "images").glob("*.png"))
    golden_labels = list((golden_dir / "labels").glob("*.json"))
    assert len(golden_images) == golden_count
    assert len(golden_labels) == golden_count
    assert {p.stem for p in golden_images} == {p.stem for p in golden_labels}


def test_augmented_label_matches_original(tmp_path):
    """Augment edilmis etiket, sadece goruntu bozuldugu icin orijinaliyle BIREBIR ayni olmali."""
    input_dir = tmp_path / "synthetic"
    output_dir = tmp_path / "synthetic" / "augmented"
    golden_dir = tmp_path / "golden"

    _make_fake_dataset(input_dir, count=1)
    result = _run_augment_script(input_dir, output_dir, golden_dir, per_image=1, golden_count=1, seed=42)
    assert result.returncode == 0, result.stderr

    original_text = (input_dir / "labels" / "invoice_0001.json").read_text(encoding="utf-8")
    augmented_text = (output_dir / "labels" / "invoice_0001_aug01.json").read_text(encoding="utf-8")
    assert original_text == augmented_text


def test_golden_set_is_stable_across_seeds(tmp_path):
    """Golden set, --seed farkli olsa da HER ZAMAN ayni dosyalari icermeli (CI/CD kosulu)."""
    input_dir = tmp_path / "synthetic"
    _make_fake_dataset(input_dir, count=5)

    def golden_stems_for_seed(seed: int, run_name: str) -> set[str]:
        output_dir = tmp_path / run_name / "augmented"
        golden_dir = tmp_path / run_name / "golden"
        result = _run_augment_script(input_dir, output_dir, golden_dir, per_image=2, golden_count=3, seed=seed)
        assert result.returncode == 0, result.stderr
        return {p.stem for p in (golden_dir / "images").glob("*.png")}

    golden_with_seed_1 = golden_stems_for_seed(1, "run_seed_1")
    golden_with_seed_2 = golden_stems_for_seed(999, "run_seed_999")

    assert golden_with_seed_1 == golden_with_seed_2

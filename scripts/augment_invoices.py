"""
augment_invoices.py

generate_invoices.py'nin urettigi temiz sentetik fatura goruntulerini
(data/synthetic/images) alip gercekci bozulmalarla cogaltir.
Amac: Vision Agent'i sadece "stüdyo kalitesinde" temiz görsellerle değil,
gercek hayattaki taranmis/fotokopi/telefonla cekilmis fatura goruntulerine
yakin kosullarda da test etmek.

Iki asamali bozulma uygulanir:
    1. augraphy  -> belge/kagit seviyesinde gercekci bozulmalar
                    (tarayici/yazici izi, murekkep solmasi, kagit dokusu, leke, golge)
    2. albumentations -> kamera/tarayici hizalama bozulmalari
                    (hafif dondurme, perspektif kaymasi, gaussian blur, parlaklik/kontrast)

Her temiz goruntuden --per-image kadar farkli augment edilmis versiyon uretilir.
Augment sadece goruntuyu bozar, faturanin verisi/matematigi degismedigi icin
her versiyon orijinaliyle AYNI ground-truth JSON'u miras alir.

Ayrica augment edilmis veri kumesinden sabit bir seed ile kucuk bir "golden set"
ayrilir (data/golden/). Bu set CI/CD regresyon testlerinde kullanilir ve
--seed parametresi ne olursa olsun HER ZAMAN ayni dosyalari icerir.

Kullanim:
    python augment_invoices.py --input-dir data/synthetic --output-dir data/synthetic/augmented \
        --per-image 5 --golden-count 15 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from albumentations import Compose, GaussianBlur, Perspective, RandomBrightnessContrast, Rotate
from augraphy import (
    AugraphyPipeline,
    BrightnessTexturize,
    DirtyRollers,
    LowInkRandomLines,
    NoiseTexturize,
    ShadowCast,
    Stains,
)

# Golden set, kullanicinin --seed'inden bagimsiz sabit bir seed ile secilir.
# Boylece augment parametreleri farkli bir seed ile yeniden uretilse bile
# CI/CD'nin karsilastirdigi golden dosyalar hic degismez.
GOLDEN_SET_SEED = 20260913


class DocumentDegrader:
    """Augraphy pipeline'ini sarmalayip tarayici/yazici/kagit kaynakli bozulmalari uygular."""

    def __init__(self, seed: int | None = None):
        self._pipeline = AugraphyPipeline(
            ink_phase=[LowInkRandomLines(p=0.4)],
            paper_phase=[NoiseTexturize(p=0.6), BrightnessTexturize(p=0.5)],
            post_phase=[DirtyRollers(p=0.35), Stains(p=0.35), ShadowCast(p=0.35)],
            random_seed=seed,
        )

    def degrade(self, image_bgr: np.ndarray) -> np.ndarray:
        return self._pipeline.augment(image_bgr)["output"]


class GeometricAugmenter:
    """Albumentations pipeline'ini sarmalayip hafif geometri/isik bozulmalari uygular."""

    WHITE = (255, 255, 255)

    def __init__(self, seed: int | None = None):
        self._transform = Compose(
            [
                Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, fill=self.WHITE, p=0.8),
                Perspective(scale=(0.02, 0.05), border_mode=cv2.BORDER_CONSTANT, fill=self.WHITE, p=0.5),
                GaussianBlur(blur_limit=(3, 5), p=0.4),
                RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            ],
            seed=seed,
        )

    def augment(self, image_bgr: np.ndarray) -> np.ndarray:
        return self._transform(image=image_bgr)["image"]


class InvoiceAugmentationPipeline:
    """Tek bir temiz fatura goruntusune once belge bozulmasini, sonra geometri bozulmasini uygular."""

    def __init__(self, degrader: DocumentDegrader, augmenter: GeometricAugmenter):
        self.degrader = degrader
        self.augmenter = augmenter

    def generate_variant(self, clean_image_bgr: np.ndarray) -> np.ndarray:
        degraded = self.degrader.degrade(clean_image_bgr)
        return self.augmenter.augment(degraded)


class GoldenSetSelector:
    """Augment edilmis veri kumesinden CI/CD icin degismeyen sabit bir alt kume secer."""

    def __init__(self, count: int, seed: int = GOLDEN_SET_SEED):
        self.count = count
        self._rng = random.Random(seed)

    def select(self, stems: list[str]) -> list[str]:
        k = min(self.count, len(stems))
        return self._rng.sample(sorted(stems), k)

    def copy_to_golden(
        self,
        stems: list[str],
        source_images_dir: Path,
        source_labels_dir: Path,
        golden_images_dir: Path,
        golden_labels_dir: Path,
    ) -> None:
        golden_images_dir.mkdir(parents=True, exist_ok=True)
        golden_labels_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            shutil.copy2(source_images_dir / f"{stem}.png", golden_images_dir / f"{stem}.png")
            shutil.copy2(source_labels_dir / f"{stem}.json", golden_labels_dir / f"{stem}.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=str, default="data/synthetic",
                         help="generate_invoices.py ciktisinin bulundugu klasor (images/ ve labels/ icerir)")
    parser.add_argument("--output-dir", type=str, default="data/synthetic/augmented",
                         help="Augment edilmis goruntu/etiketlerin yazilacagi klasor")
    parser.add_argument("--golden-dir", type=str, default="data/golden",
                         help="CI/CD golden set'inin yazilacagi klasor")
    parser.add_argument("--per-image", type=int, default=5,
                         help="Her temiz goruntuden uretilecek augment sayisi")
    parser.add_argument("--golden-count", type=int, default=15,
                         help="Golden set'e alinacak dosya sayisi")
    parser.add_argument("--seed", type=int, default=None,
                         help="Augraphy/albumentations rastgeleligi icin seed (golden set secimini etkilemez)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    clean_images_dir = input_dir / "images"
    clean_labels_dir = input_dir / "labels"

    output_dir = Path(args.output_dir)
    aug_images_dir = output_dir / "images"
    aug_labels_dir = output_dir / "labels"
    aug_images_dir.mkdir(parents=True, exist_ok=True)
    aug_labels_dir.mkdir(parents=True, exist_ok=True)

    clean_image_paths = sorted(clean_images_dir.glob("invoice_*.png"))
    if not clean_image_paths:
        raise SystemExit(
            f"{clean_images_dir} icinde temiz fatura goruntusu bulunamadi. "
            "Once generate_invoices.py calistirin."
        )

    degrader = DocumentDegrader(seed=args.seed)
    augmenter = GeometricAugmenter(seed=args.seed)
    pipeline = InvoiceAugmentationPipeline(degrader, augmenter)

    generated_stems: list[str] = []
    for clean_image_path in clean_image_paths:
        stem = clean_image_path.stem  # orn: invoice_0001
        clean_label_path = clean_labels_dir / f"{stem}.json"
        if not clean_label_path.exists():
            print(f"UYARI: {clean_label_path} bulunamadi, {clean_image_path.name} atlaniyor.")
            continue

        clean_image_bgr = cv2.imread(str(clean_image_path))
        label_text = clean_label_path.read_text(encoding="utf-8")

        for variant_no in range(1, args.per_image + 1):
            variant_stem = f"{stem}_aug{variant_no:02d}"
            augmented_bgr = pipeline.generate_variant(clean_image_bgr)

            cv2.imwrite(str(aug_images_dir / f"{variant_stem}.png"), augmented_bgr)
            (aug_labels_dir / f"{variant_stem}.json").write_text(label_text, encoding="utf-8")

            generated_stems.append(variant_stem)

    print(f"{len(clean_image_paths)} temiz goruntuden {len(generated_stems)} augment edilmis versiyon uretildi -> {output_dir}")

    golden_dir = Path(args.golden_dir)
    selector = GoldenSetSelector(count=args.golden_count)
    golden_stems = selector.select(generated_stems)
    selector.copy_to_golden(
        golden_stems,
        source_images_dir=aug_images_dir,
        source_labels_dir=aug_labels_dir,
        golden_images_dir=golden_dir / "images",
        golden_labels_dir=golden_dir / "labels",
    )
    print(f"Golden set: {len(golden_stems)} dosya -> {golden_dir} (sabit seed={GOLDEN_SET_SEED}, --seed'den etkilenmez)")


if __name__ == "__main__":
    main()

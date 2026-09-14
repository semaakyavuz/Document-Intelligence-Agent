"""
Proje capinda (app/, scripts/) tum metin dosyasi I/O'sunun (open/read_text/write_text)
acikca encoding="utf-8" belirttigini AST uzerinden statik olarak dogrular. Ayrica
scripts/*.py icinde Turkce ozel karakter basan print() cagrisi varsa, dosyanin bir
yerinde sys.stdout.reconfigure(encoding="utf-8") cagrisi oldugunu dogrular.

Amac: Faz 1'de generate_invoices.py'de yasanan turden bir mojibake hatasinin -
encoding parametresi verilmeyip Windows'un varsayilan sistem codepage'ine (cp1252
vb.) dusulmesi - projede baska bir dosyada sessizce tekrarlanmasini, calisma
zamanini beklemeden (kod inceleme asamasinda) yakalamak icin.

PIL.Image.open() bilinen tek istisna: bir goruntu dosyasi acar, metin degil;
encoding parametresi kabul etmez (verilirse TypeError firlatir).

Ikinci kontrolun ayri gerekcesi: run_pipeline_manual.py'de canlica yasanan farkli bir
hata - dosya I/O'su degil, print() cikisi. stdout gercek bir konsola bagli degilse
(yonlendirme/boru hatti) Windows'ta Turkce sistemlerde varsayilan ANSI kod sayfasi
(cp1254) kullanilir, UTF-8 degil; bu da Turkce karakterleri sessizce bozar.
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKED_DIRS = ("app", "scripts")
KNOWN_BINARY_RECEIVERS = {"Image"}  # PIL.Image.open(): goruntu acar, encoding almaz


def _python_files() -> list[Path]:
    files = []
    for sub in CHECKED_DIRS:
        files.extend(sorted((PROJECT_ROOT / sub).rglob("*.py")))
    return files


def _mode_is_binary(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and "b" in node.value


def _find_violations(path: Path) -> list[tuple[int, str]]:
    """encoding= icermeyen open()/.open()/.read_text()/.write_text() cagrilarini bulur."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            call_name, receiver_name = "open", None
        elif isinstance(func, ast.Attribute) and func.attr in ("open", "read_text", "write_text"):
            call_name = func.attr
            receiver_name = func.value.id if isinstance(func.value, ast.Name) else None
        else:
            continue

        if call_name == "open" and receiver_name in KNOWN_BINARY_RECEIVERS:
            continue

        if call_name == "open":
            mode_arg = node.args[1] if len(node.args) >= 2 else None
            mode_kw = next((kw.value for kw in node.keywords if kw.arg == "mode"), None)
            if _mode_is_binary(mode_arg) or _mode_is_binary(mode_kw):
                continue  # binary mod ("rb"/"wb"): encoding almaz/gerektirmez

        if not any(kw.arg == "encoding" for kw in node.keywords):
            violations.append((node.lineno, call_name))
    return violations


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_text_file_io_specifies_utf8_encoding(path):
    violations = _find_violations(path)
    assert not violations, (
        f"{path.relative_to(PROJECT_ROOT)}: encoding='utf-8' eksik -> "
        + ", ".join(f"L{line} ({name})" for line, name in violations)
    )


TURKISH_DIACRITICS = set("çğıöşüÇĞİÖŞÜ")


def _has_print_with_turkish_literal(tree: ast.AST) -> bool:
    """print() cagrilarinin literal metninde (f-string sabit kisimlari dahil, {degisken}
    kisimlari haric) Turkce ozel karakter var mi kontrol eder."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"):
            continue
        for arg in node.args:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and any(
                    ch in TURKISH_DIACRITICS for ch in sub.value
                ):
                    return True
    return False


def _has_stdout_reconfigure(tree: ast.AST) -> bool:
    """sys.stdout.reconfigure(...) cagrisi dosyanin herhangi bir yerinde var mi.
    Bir kere yeterli; her print()'ten once tekrarlanmasi gerekmez."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reconfigure"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "stdout"
        ):
            return True
    return False


@pytest.mark.parametrize(
    "path", sorted((PROJECT_ROOT / "scripts").glob("*.py")), ids=lambda p: str(p.relative_to(PROJECT_ROOT))
)
def test_scripts_printing_turkish_text_reconfigure_stdout_to_utf8(path):
    """Turkce ozel karakterli print() varsa, dosyada sys.stdout.reconfigure(encoding='utf-8')
    cagrisi olmali; yoksa stdout yonlendirildiginde/boru hattina baglandiginda Windows'ta
    sistemin varsayilan ANSI kod sayfasina dusup bu karakterleri sessizce bozabilir."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if not _has_print_with_turkish_literal(tree):
        pytest.skip("Turkce ozel karakterli print() yok, kontrol gerekmiyor.")
    assert _has_stdout_reconfigure(tree), (
        f"{path.relative_to(PROJECT_ROOT)}: Turkce karakterli print() var ama "
        "sys.stdout.reconfigure(encoding='utf-8') cagrisi bulunamadi."
    )

"""Garante que os pesos dos modelos estão em models/.

1. Confere cada arquivo listado em scripts/pesos.json (tamanho e SHA-256).
2. Se faltar algum, baixa as pastas-espelho do Google Drive (gdown) e confere de novo.

    python scripts/baixar_pesos.py            # confere e baixa o que faltar
    python scripts/baixar_pesos.py --gerar    # (autor) recalcula pesos.json a partir de models/
    python scripts/baixar_pesos.py --destino pasta_temporaria   # testa o download sem mexer em models/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "models"
MANIFESTO = Path(__file__).with_name("pesos.json")
EXTENSOES = {".pt", ".joblib"}


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def gerar() -> None:
    antigo = json.loads(MANIFESTO.read_text(encoding="utf-8")) if MANIFESTO.exists() else {}
    arquivos = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(MODELOS.glob("*")) if p.suffix in EXTENSOES}
    MANIFESTO.write_text(json.dumps({"pastas_drive": _pastas(antigo), "arquivos": arquivos},
                                    indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(arquivos)} arquivos registrados em {MANIFESTO.name}")


def _pastas(manifesto: dict) -> list[str]:
    """Endereços das pastas-espelho (aceita o formato antigo, com uma pasta só)."""
    pastas = manifesto.get("pastas_drive") or []
    if manifesto.get("pasta_drive"):
        pastas = [manifesto["pasta_drive"], *pastas]
    return list(dict.fromkeys(pastas))


def faltando(manifesto: dict, destino: Path = MODELOS) -> list[str]:
    ruins = []
    for nome, info in manifesto["arquivos"].items():
        p = destino / nome
        if not p.exists() or p.stat().st_size != info["bytes"] or sha256(p) != info["sha256"]:
            ruins.append(nome)
    return ruins


def baixar(url: str, destino: Path = MODELOS) -> None:
    try:
        import gdown
    except ImportError:
        sys.exit("Instale as dependências: pip install -r requirements.txt")
    with tempfile.TemporaryDirectory() as tmp:
        gdown.download_folder(url, output=tmp, quiet=False, use_cookies=False)
        destino.mkdir(parents=True, exist_ok=True)
        for p in Path(tmp).rglob("*"):
            if p.suffix in EXTENSOES:
                shutil.move(str(p), destino / p.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gerar", action="store_true")
    ap.add_argument("--destino", type=Path, default=MODELOS)
    a = ap.parse_args()
    if a.gerar:
        return gerar()
    destino = a.destino
    manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    ruins = faltando(manifesto, destino)
    if not ruins:
        print(f"Todos os {len(manifesto['arquivos'])} pesos estão em models/ e conferem.")
        return
    print(f"Faltam ou estão diferentes: {', '.join(ruins)}")
    if not _pastas(manifesto):
        sys.exit("O endereço do espelho no Google Drive não está configurado em scripts/pesos.json.")
    for url in _pastas(manifesto):
        baixar(url, destino)
    ruins = faltando(manifesto, destino)
    if ruins:
        sys.exit(f"Após o download ainda faltam: {', '.join(ruins)}")
    print("Pesos baixados e conferidos.")


if __name__ == "__main__":
    main()

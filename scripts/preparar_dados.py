"""Extrai as bases dos zips para data/ (fora do Git).

- TRUNK12: todas as 12 espécies -> data/trunk12/<especie>/
- Pl@ntNet-300K: só as 10 árvores escolhidas, até LIMITE fotos cada
  -> data/plantnet/<split>/<especie>/ + data/plantnet/manifesto.csv (autor e licença de cada foto)

Uso:
    python scripts/preparar_dados.py --zips "caminho/da/pasta/com/os/zips"
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
SEMENTE = 42
LIMITE = 500

# Nome de pasta -> ids da Pl@ntNet-300K (Schefflera actinophylla aparece com dois ids na base)
ESPECIES_PLANTNET: dict[str, list[str]] = {
    "punica_granatum": ["1359616"],
    "liriodendron_tulipifera": ["1393614"],
    "acacia_dealbata": ["1363764"],
    "schefflera_actinophylla": ["1362954", "1399783"],
    "hippophae_rhamnoides": ["1392777"],
    "metasequoia_glyptostroboides": ["1396708"],
    "acacia_retinodes": ["1363778"],
    "casuarina_cunninghamiana": ["1361847"],
    "morinda_citrifolia": ["1372016"],
    "couroupita_guianensis": ["1385701"],
}


def extrair_trunk12(zip_path: Path) -> None:
    destino = DATA / "trunk12"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(destino)
    total = sum(1 for p in destino.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"TRUNK12: {total} imagens em {destino}")


def selecionar(fotos: list[dict], limite: int, rng: random.Random) -> list[dict]:
    """Escolhe até `limite` fotos de UMA espécie.

    Cada item de `fotos` é um dict com: arquivo, split ('train'/'val'/'test'),
    organ ('leaf', 'flower', 'fruit', 'bark', 'habit'...), author, license.
    Devolve a sublista escolhida. Use `rng` (semente fixa) para que o resultado
    seja reprodutível.
    """
    # TODO(human)
    raise NotImplementedError


def extrair_plantnet(zip_path: Path) -> None:
    rng = random.Random(SEMENTE)
    id_para_pasta = {sid: pasta for pasta, ids in ESPECIES_PLANTNET.items() for sid in ids}
    destino = DATA / "plantnet"
    with zipfile.ZipFile(zip_path) as z:
        meta = json.loads(z.read("plantnet_300K/plantnet300K_metadata.json"))
        por_especie: dict[str, list[dict]] = defaultdict(list)
        for img_id, m in meta.items():
            pasta = id_para_pasta.get(m["species_id"])
            if pasta is None:
                continue
            split = m["split"]
            por_especie[pasta].append({
                "arquivo": f"plantnet_300K/images_{split}/{m['species_id']}/{img_id}.jpg",
                "split": split, "organ": m["organ"],
                "author": m["author"], "license": m["license"],
            })

        linhas = []
        for pasta in sorted(por_especie):
            escolhidas = selecionar(por_especie[pasta], LIMITE, rng)
            for f in escolhidas:
                saida = destino / f["split"] / pasta / Path(f["arquivo"]).name
                saida.parent.mkdir(parents=True, exist_ok=True)
                saida.write_bytes(z.read(f["arquivo"]))
                linhas.append({"especie": pasta, "caminho": saida.relative_to(DATA).as_posix(),
                               **{k: f[k] for k in ("split", "organ", "author", "license")}})
            print(f"{pasta}: {len(escolhidas)} de {len(por_especie[pasta])}")

    with open(destino / "manifesto.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["especie", "caminho", "split", "organ", "author", "license"])
        w.writeheader()
        w.writerows(linhas)
    print(f"Pl@ntNet: {len(linhas)} imagens; manifesto em {destino / 'manifesto.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", type=Path, required=True, help="pasta com TRUNK12.zip e plantnet_300K.zip")
    args = ap.parse_args()
    extrair_trunk12(args.zips / "TRUNK12.zip")
    extrair_plantnet(args.zips / "plantnet_300K.zip")


if __name__ == "__main__":
    main()

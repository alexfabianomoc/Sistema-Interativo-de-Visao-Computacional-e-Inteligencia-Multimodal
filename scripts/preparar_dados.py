"""Prepara as bases em data/ (fora do Git).

- Casca: TRUNK12 (12 espécies) extraído do zip -> data/trunk12/<especie>/
- Folha: fotos de folha das MESMAS 12 espécies, do conjunto "Pl@ntNet observations" no GBIF,
  até LIMITE por espécie -> data/folhas/<split>/<especie>/ + data/folhas/manifesto.csv
  (autor, licença e link de cada foto, para os créditos no README)

Etapas (cada uma pode ser rodada sozinha):
    python scripts/preparar_dados.py trunk12 --zips "pasta/com/TRUNK12.zip"
    python scripts/preparar_dados.py listar      # consulta o GBIF -> data/folhas/candidatos.json
    python scripts/preparar_dados.py baixar      # seleciona e baixa as fotos (600 px)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
FOLHAS = DATA / "folhas"
SEMENTE = 42
LIMITE = 500
MAX_OBS_POR_ESPECIE = 4000  # observações consultadas por espécie (folhas suficientes para o LIMITE)
GBIF = "https://api.gbif.org/v1"
DATASET_PLANTNET = "7a3679ef-5582-4aaa-81f0-8c2545cafc81"  # "Pl@ntNet observations"

# Pasta do TRUNK12 -> espécie usada na busca do GBIF (igual a src/metrics/dados.py: TAXON_GBIF).
# Tília, carvalho e pinheiro: espécie escolhida em 20/11 sem confirmação; ver docs/auditoria_taxonomica.md
ESPECIES: dict[str, str] = {
    "alder": "Alnus glutinosa",
    "beech": "Fagus sylvatica",
    "birch": "Betula pendula",
    "chestnut": "Castanea sativa",
    "ginkgo biloba": "Ginkgo biloba",
    "hornbeam": "Carpinus betulus",
    "horse chestnut": "Aesculus hippocastanum",
    "linden": "Tilia cordata",
    "oak": "Quercus robur",
    "oriental plane": "Platanus orientalis",
    "pine": "Pinus sylvestris",
    "spruce": "Picea abies",
}


def _get_json(url: str, tentativas: int = 4) -> dict:
    for i in range(tentativas):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if i == tentativas - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("inalcançável")


def extrair_trunk12(zip_path: Path) -> None:
    destino = DATA / "trunk12"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(destino)
    total = sum(1 for p in destino.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"TRUNK12: {total} imagens em {destino}")


def listar_candidatos() -> None:
    """Consulta o GBIF e grava todas as fotos de folha encontradas, por espécie."""
    candidatos: dict[str, list[dict]] = {}
    for pasta, nome in ESPECIES.items():
        chave = _get_json(f"{GBIF}/species/match?name={urllib.parse.quote(nome)}")["usageKey"]
        fotos, offset = [], 0
        while offset < MAX_OBS_POR_ESPECIE:
            r = _get_json(f"{GBIF}/occurrence/search?taxonKey={chave}&datasetKey={DATASET_PLANTNET}"
                          f"&mediaType=StillImage&limit=300&offset={offset}")
            for occ in r["results"]:
                for m in occ.get("media", []):
                    if (m.get("title") or "").lower().endswith(": leaf") and m.get("identifier"):
                        fotos.append({
                            "obs": str(occ["key"]),
                            "url": m["identifier"].replace("/image/o/", "/image/m/"),  # 600 px
                            "autor": m.get("creator") or m.get("rightsHolder") or "",
                            "licenca": m.get("license") or occ.get("license") or "",
                            "fonte": m.get("source") or "",
                        })
            if r["endOfRecords"]:
                break
            offset += 300
        candidatos[pasta] = fotos
        print(f"{pasta:16s} {nome:24s} {len(fotos):5d} fotos de folha "
              f"em {len({f['obs'] for f in fotos})} observações")
    FOLHAS.mkdir(parents=True, exist_ok=True)
    (FOLHAS / "candidatos.json").write_text(json.dumps(candidatos, ensure_ascii=False), encoding="utf-8")


def selecionar(fotos: list[dict], limite: int, rng: random.Random) -> list[dict]:
    """Escolhe até `limite` fotos de UMA espécie e define o split de cada uma.

    Cada item de `fotos` tem: obs (id da observação no GBIF; uma observação é uma
    planta fotografada por uma pessoa num dia e pode ter várias fotos), url, autor,
    licenca, fonte. Devolva a sublista escolhida, cada item com a chave extra
    "split" valendo "train", "val" ou "test". Use `rng` (semente fixa) para que o
    resultado seja reprodutível.
    """
    # Agrupa por observação e sorteia observações inteiras, para que fotos da
    # mesma planta nunca fiquem em splits diferentes (evita teste otimista).
    por_obs: dict[str, list[dict]] = defaultdict(list)
    for f in fotos:
        por_obs[f["obs"]].append(f)
    obs_ids = sorted(por_obs)  # ordem estável antes do sorteio -> reprodutível
    rng.shuffle(obs_ids)

    # Pega observações inteiras até o limite (espécies pequenas entram todas).
    grupos, total = [], 0
    for o in obs_ids:
        if total >= limite:
            break
        grupo = por_obs[o][: limite - total]
        grupos.append(grupo)
        total += len(grupo)

    # 10% teste, 10% validação, 80% treino, contando fotos; teste e validação
    # são preenchidos primeiro para não ficarem vazios nas espécies pequenas.
    escolhidas, acumulado = [], 0
    for grupo in grupos:
        split = "test" if acumulado < 0.1 * total else "val" if acumulado < 0.2 * total else "train"
        escolhidas += [{**f, "split": split} for f in grupo]
        acumulado += len(grupo)
    return escolhidas


def _baixar(item: tuple[str, Path]) -> bool:
    url, destino = item
    if destino.exists():
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "projeto-academico-visao-computacional"})
        with urllib.request.urlopen(req, timeout=60) as r:
            destino.write_bytes(r.read())
        return True
    except Exception as e:  # registra e segue; a foto fica de fora
        print(f"falhou: {url} ({e})", file=sys.stderr)
        return False


def baixar_folhas() -> None:
    candidatos = json.loads((FOLHAS / "candidatos.json").read_text(encoding="utf-8"))
    rng = random.Random(SEMENTE)
    linhas, tarefas = [], []
    for pasta in sorted(candidatos):
        escolhidas = selecionar(candidatos[pasta], LIMITE, rng)
        for f in escolhidas:
            nome = f["url"].rstrip("/").rsplit("/", 1)[-1] + ".jpg"
            destino = FOLHAS / f["split"] / pasta / nome
            destino.parent.mkdir(parents=True, exist_ok=True)
            tarefas.append((f["url"], destino))
            linhas.append({"especie": pasta, "caminho": destino.relative_to(DATA).as_posix(),
                           **{k: f[k] for k in ("split", "obs", "autor", "licenca", "fonte")}})
        print(f"{pasta}: {len(escolhidas)} de {len(candidatos[pasta])}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = list(ex.map(_baixar, tarefas))
    linhas = [l for l, o in zip(linhas, ok) if o]
    with open(FOLHAS / "manifesto.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["especie", "caminho", "split", "obs", "autor", "licenca", "fonte"])
        w.writeheader()
        w.writerows(linhas)
    print(f"Folhas: {len(linhas)} imagens baixadas ({ok.count(False)} falhas); manifesto em {FOLHAS / 'manifesto.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["trunk12", "listar", "baixar"])
    ap.add_argument("--zips", type=Path, help="pasta com TRUNK12.zip (etapa trunk12)")
    args = ap.parse_args()
    if args.etapa == "trunk12":
        if not args.zips:
            ap.error("a etapa trunk12 precisa de --zips")
        extrair_trunk12(args.zips / "TRUNK12.zip")
    elif args.etapa == "listar":
        listar_candidatos()
    else:
        baixar_folhas()


if __name__ == "__main__":
    main()

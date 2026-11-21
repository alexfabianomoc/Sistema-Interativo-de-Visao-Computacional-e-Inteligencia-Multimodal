"""Baixa fotos de casca de campo do GBIF para TREINO e VALIDAÇÃO (opção A, 21/11), excluindo as
observações já usadas no teste de campo (data/casca_campo). Divisão por observação: 85% treino, 15% val.

    python scripts/baixar_cascas_treino.py --por-especie 200
Saída: data/casca_gbif/<split>/<especie>/ + data/casca_gbif/manifesto.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from preparar_dados import DATASET_PLANTNET, ESPECIES, GBIF, SEMENTE, _baixar, _get_json  # noqa: E402

SAIDA = RAIZ / "data" / "casca_gbif"
TESTE = RAIZ / "data" / "casca_campo" / "manifesto.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-especie", type=int, default=200)
    n = ap.parse_args().por_especie
    with open(TESTE, encoding="utf-8") as fh:
        obs_teste = {l["obs"] for l in csv.DictReader(fh)}
    rng = random.Random(SEMENTE + 1)
    linhas, tarefas = [], []
    for pasta, nome in ESPECIES.items():
        chave = _get_json(f"{GBIF}/species/match?name={urllib.parse.quote(nome)}")["usageKey"]
        fotos, offset = [], 0
        while offset < 10000:
            r = _get_json(f"{GBIF}/occurrence/search?taxonKey={chave}&datasetKey={DATASET_PLANTNET}"
                          f"&mediaType=StillImage&limit=300&offset={offset}")
            for occ in r["results"]:
                if str(occ["key"]) in obs_teste:
                    continue
                for m in occ.get("media", []):
                    if (m.get("title") or "").lower().endswith(": bark") and m.get("identifier"):
                        fotos.append({"obs": str(occ["key"]), "url": m["identifier"].replace("/image/o/", "/image/m/"),
                                      "autor": m.get("creator") or "", "licenca": m.get("license") or "",
                                      "fonte": m.get("source") or ""})
                        break  # uma foto por observação
            if r["endOfRecords"]:
                break
            offset += 300
        rng.shuffle(fotos)
        fotos = fotos[:n]
        corte = round(len(fotos) * 0.15)
        for k, f in enumerate(fotos):
            split = "val" if k < corte else "train"
            destino = SAIDA / split / pasta / (f["url"].rsplit("/", 1)[-1] + ".jpg")
            destino.parent.mkdir(parents=True, exist_ok=True)
            tarefas.append((f["url"], destino))
            linhas.append({"especie": pasta, "split": split,
                           "caminho": destino.relative_to(RAIZ / "data").as_posix(), **f})
        print(f"{pasta}: {len(fotos)} fotos ({corte} validação)", flush=True)
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(_baixar, tarefas))
    linhas = [l for l, o in zip(linhas, ok) if o]
    with open(SAIDA / "manifesto.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"{len(linhas)} fotos baixadas ({ok.count(False)} falhas)")


if __name__ == "__main__":
    main()

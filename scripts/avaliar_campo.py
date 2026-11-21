"""Avalia os modelos de casca em fotos de campo do GBIF (data/casca_campo), que nunca entraram
no treino e não são padronizadas como as do TRUNK12. Grava docs/metricas/campo_casca.json."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.metrics.dados import CLASSES  # noqa: E402
from src.models import servico  # noqa: E402
from src.preprocessing.transforms import ler_arquivo  # noqa: E402


def main() -> None:
    with open(RAIZ / "data" / "casca_campo" / "manifesto.csv", encoding="utf-8") as fh:
        itens = [(RAIZ / "data" / l["caminho"], CLASSES.index(l["especie"])) for l in csv.DictReader(fh)]
    imgs = [ler_arquivo(p) for p, _ in itens]
    y = np.array([c for _, c in itens])
    rot = list(range(len(CLASSES)))
    saida = {"n": len(itens), "fonte": "GBIF, Pl@ntNet observations, fotos marcadas como bark", "metodos": {}}
    for metodo in ("mobilenet_v3_large", "resnet18", "classico"):
        m = servico.carregar(metodo, "casca")
        prob = np.stack([servico.classificar(m, metodo, "casca", im, 224, 3.0)["prob"] for im in imgs])
        pred = prob.argmax(1)
        saida["metodos"][metodo] = {
            "acuracia": round(accuracy_score(y, pred), 4),
            "acuracia_top3": round(top_k_accuracy_score(y, prob, k=3, labels=rot), 4),
            "f1_macro": round(f1_score(y, pred, average="macro", labels=rot, zero_division=0), 4),
            "confianca_media": round(float(prob.max(1).mean()), 4),
            "fracao_inconclusivo_limiar_0.6": round(float((prob.max(1) < 0.6).mean()), 4),
        }
        print(metodo, saida["metodos"][metodo])
    (RAIZ / "docs" / "metricas" / "campo_casca.json").write_text(
        json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

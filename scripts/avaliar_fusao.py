"""Avalia o uso combinado casca + folha (fusão tardia) com MobileNetV3.

As bases não têm fotos da mesma árvore, então cada foto de folha do teste é pareada com uma foto de
casca de teste da MESMA espécie, sorteada com semente fixa (limitação registrada no README).
Duas variantes de casca: teste do TRUNK12 e fotos de campo do GBIF. Grava docs/metricas/fusao.json.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.metrics import dados  # noqa: E402
from src.models import servico  # noqa: E402
from src.preprocessing.transforms import ler_arquivo  # noqa: E402

METODO = "mobilenet_v3_large"


def probs(modelo, modo, itens):
    return np.stack([servico.classificar(modelo, METODO, modo, ler_arquivo(p), 224,
                                         3.0 if modo == "casca" else 1.5)["prob"] for p, _ in itens])


def main() -> None:
    rng = random.Random(42)
    folha = dados.folhas()["test"]
    pf = probs(servico.carregar(METODO, "folha"), "folha", folha)
    yf = np.array([c for _, c in folha])
    mc = servico.carregar(METODO, "casca")
    variantes = {"TRUNK12 (teste)": dados.trunk12()["test"], "Fotos de campo (GBIF)": dados.casca_campo()}
    tabela = []
    for nome, casca in variantes.items():
        pc = probs(mc, "casca", casca)
        por_classe = defaultdict(list)
        for k, (_, c) in enumerate(casca):
            por_classe[c].append(k)
        idx = [i for i in range(len(folha)) if por_classe[yf[i]]]
        par = [rng.choice(por_classe[yf[i]]) for i in idx]
        y = yf[idx]
        a_c = (pc[par].argmax(1) == y).mean()
        a_f = (pf[idx].argmax(1) == y).mean()
        fus = np.stack([servico.fusao(pc[j], pf[i]) for i, j in zip(idx, par)])
        a_u = (fus.argmax(1) == y).mean()
        tabela.append({"Casca de": nome, "Pares": len(idx), "Só casca": round(float(a_c), 4),
                       "Só folha": round(float(a_f), 4), "Casca + folha": round(float(a_u), 4)})
        print(tabela[-1])
    nota = ("Pares montados com fotos de casca e de folha da mesma espécie, de árvores diferentes "
            "(as bases não têm a mesma árvore). Fusão: produto das probabilidades da MobileNetV3.")
    (RAIZ / "docs" / "metricas" / "fusao.json").write_text(
        json.dumps({"tabela": tabela, "nota": nota}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

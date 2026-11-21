"""Avalia o uso combinado casca + folha (fusão tardia) com ConvNeXt-Tiny (método padrão).

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

METODO = "convnext_tiny"


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
            "(as bases não têm a mesma árvore). Fusão: produto das probabilidades da ConvNeXt-Tiny.")
    (RAIZ / "docs" / "metricas" / "fusao.json").write_text(
        json.dumps({"tabela": tabela, "nota": nota}, ensure_ascii=False, indent=2), encoding="utf-8")


def main_br() -> None:
    """Base brasileira: tronco (3 classes) + folha (10 classes), fusão nas 3 espécies em comum.
    Pares da MESMA árvore quando a observação do iNaturalist tem foto de tronco e de folha no teste."""
    import csv
    linhas = list(csv.DictReader(open(dados.DATA / "inat_br" / "manifesto.csv", encoding="utf-8")))
    comuns = [c for c in dados.TRONCO_BR if c in dados.FOLHA_BR]
    teste = [l for l in linhas if l.get("split") == "test" and l["taxon"] in comuns]
    por_obs = defaultdict(lambda: {"tronco": [], "folha": []})
    for l in teste:
        if l["parte"] in ("tronco", "folha"):
            por_obs[l["obs"]][l["parte"]].append(l)
    pares = [(v["tronco"][0], v["folha"][0]) for v in por_obs.values() if v["tronco"] and v["folha"]]
    mt, mf = servico.carregar(METODO, "tronco_br"), servico.carregar(METODO, "folha_br")
    acertos = {"tronco": 0, "folha": 0, "fusao": 0}
    for lt, lf in pares:
        pt = servico.classificar(mt, METODO, "tronco_br", ler_arquivo(dados.DATA / lt["caminho"]), 224, 3.0)["prob"]
        pf = servico.classificar(mf, METODO, "folha_br", ler_arquivo(dados.DATA / lf["caminho"]), 224, 1.5)["prob"]
        cls, pu = servico.fusao_por_classe(pt, dados.TRONCO_BR, pf, dados.FOLHA_BR)
        real = lt["taxon"]
        acertos["tronco"] += dados.TRONCO_BR[int(pt.argmax())] == real
        acertos["folha"] += dados.FOLHA_BR[int(pf.argmax())] == real  # folha escolhe entre as 10
        acertos["fusao"] += cls[int(pu.argmax())] == real
    n = max(len(pares), 1)
    tabela = [{"Casca de": "Mesma árvore (observação do iNaturalist com tronco e folha)", "Pares": len(pares),
               "Só casca": round(acertos["tronco"] / n, 4), "Só folha": round(acertos["folha"] / n, 4),
               "Casca + folha": round(acertos["fusao"] / n, 4)}]
    print(tabela)
    nota = (f"Pares da mesma árvore, nas {len(comuns)} espécies com modelo de tronco "
            f"({', '.join(dados.POPULAR_BR[c] for c in comuns)}). 'Só folha' escolhe entre as 10 espécies de "
            "folha; a fusão considera só as espécies em comum. ConvNeXt-Tiny.")
    (RAIZ / "docs" / "metricas" / "fusao_br.json").write_text(
        json.dumps({"tabela": tabela, "nota": nota}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main_br() if "--br" in sys.argv else main()

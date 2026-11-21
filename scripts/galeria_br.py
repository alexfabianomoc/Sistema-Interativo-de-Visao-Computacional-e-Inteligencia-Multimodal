"""Galeria de acertos e erros da base brasileira (ConvNeXt-Tiny), com o mapa de calor Grad-CAM.

- Tronco e folha: 5 acertos e 5 erros sorteados no conjunto de teste.
- Tronco + folha: pares da mesma árvore (observação do iNaturalist com as duas fotos no teste), mostrando
  o que cada modelo previu sozinho e o resultado da fusão.
Saída: docs/galeria/br_*.jpg e a lista dos casos em docs/galeria_br.json
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.metrics import dados  # noqa: E402
from src.models import servico  # noqa: E402
from src.preprocessing.transforms import ler_arquivo  # noqa: E402
from src.visualization import desenho, gradcam  # noqa: E402

SAIDA = RAIZ / "docs" / "galeria"
METODO = "convnext_tiny"
LADO = 300
CLAHE = {"tronco_br": 3.0, "folha_br": 1.5}


def _texto(tela, linhas):
    for k, (txt, cor) in enumerate(linhas):
        cv2.putText(tela, txt, (6, LADO + 22 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA)


def _miniatura(img):
    h, w = img.shape[:2]
    e = LADO / max(h, w)
    vis = cv2.resize(img, (round(w * e), round(h * e)), interpolation=cv2.INTER_AREA)
    tela = np.full((LADO + 34 + 24 * 3, LADO, 3), 255, np.uint8)
    y0, x0 = (LADO - vis.shape[0]) // 2, (LADO - vis.shape[1]) // 2
    tela[y0:y0 + vis.shape[0], x0:x0 + vis.shape[1]] = vis
    return tela


def _popular(modo, classe):
    return dados.nome(modo, classe)[0]


def _analise(m, modo, p):
    img = ler_arquivo(p)
    r = servico.classificar(m, METODO, modo, img, 224, CLAHE[modo])
    prev = int(r["prob"].argmax())
    mapa = gradcam.mapa_completo(m, img, r, prev, 224, CLAHE[modo])
    return img, r, prev, desenho.mapa_de_calor(img, mapa, 0.4)


def _salvar(nome, cartoes):
    faixa = np.hstack(cartoes + [np.full((cartoes[0].shape[0], 1, 3), 255, np.uint8)])
    cv2.imencode(".jpg", cv2.cvtColor(faixa, cv2.COLOR_RGB2BGR))[1].tofile(str(SAIDA / nome))


def galeria_modo(modo, rng, registro):
    m = servico.carregar(METODO, modo)
    cls = dados.classes(modo)
    itens = dados.carregar(modo)["test"]
    rng.shuffle(itens)
    grupos = {"acertos": [], "erros": []}
    for p, real in itens:
        if all(len(g) >= 5 for g in grupos.values()):
            break
        img, r, prev, vis = _analise(m, modo, p)
        g = grupos["acertos" if prev == real else "erros"]
        if len(g) >= 5:
            continue
        conf = float(r["prob"][prev])
        t = _miniatura(vis)
        _texto(t, [(f"Real: {_popular(modo, cls[real])}", (20, 20, 20)),
                   (f"Previsto: {_popular(modo, cls[prev])} ({conf:.0%})", (0, 140, 0) if prev == real else (200, 0, 0)),
                   ("inconclusivo" if conf < 0.6 else "", (120, 120, 120))])
        g.append(t)
        registro.append({"modo": modo, "grupo": "acerto" if prev == real else "erro",
                         "foto": p.relative_to(dados.DATA).as_posix(), "real": cls[real], "previsto": cls[prev],
                         "confianca": round(conf, 3)})
    for nome, g in grupos.items():
        _salvar(f"br_{modo}_{nome}.jpg", g)
    print(modo, {k: len(v) for k, v in grupos.items()})


def galeria_fusao(rng, registro):
    linhas = list(csv.DictReader(open(dados.DATA / "inat_br" / "manifesto.csv", encoding="utf-8")))
    comuns = [c for c in dados.TRONCO_BR if c in dados.FOLHA_BR]
    por_obs = defaultdict(lambda: {"tronco": [], "folha": []})
    for l in linhas:
        if l.get("split") == "test" and l["taxon"] in comuns and l["parte"] in ("tronco", "folha"):
            por_obs[l["obs"]][l["parte"]].append(l)
    pares = [(v["tronco"][0], v["folha"][0]) for v in por_obs.values() if v["tronco"] and v["folha"]]
    rng.shuffle(pares)
    mt, mf = servico.carregar(METODO, "tronco_br"), servico.carregar(METODO, "folha_br")
    corrigidos, falhos = [], []
    for lt, lf in pares:
        it, rt, pt, vt = _analise(mt, "tronco_br", dados.DATA / lt["caminho"])
        iff, rf, pf, vf = _analise(mf, "folha_br", dados.DATA / lf["caminho"])
        cls_u, pu = servico.fusao_por_classe(rt["prob"], dados.TRONCO_BR, rf["prob"], dados.FOLHA_BR)
        real, fus = lt["taxon"], cls_u[int(pu.argmax())]
        t_ok, f_ok, u_ok = dados.TRONCO_BR[pt] == real, dados.FOLHA_BR[pf] == real, fus == real
        if u_ok and not (t_ok and f_ok) and len(corrigidos) < 3:
            alvo = corrigidos
        elif not u_ok and len(falhos) < 2:
            alvo = falhos
        else:
            continue
        a, b = _miniatura(vt), _miniatura(vf)
        _texto(a, [(f"Real: {_popular('tronco_br', real)}", (20, 20, 20)),
                   (f"Tronco: {_popular('tronco_br', dados.TRONCO_BR[pt])} ({rt['prob'][pt]:.0%})", (0, 140, 0) if t_ok else (200, 0, 0)),
                   ("", (0, 0, 0))])
        _texto(b, [("", (0, 0, 0)),
                   (f"Folha: {_popular('folha_br', dados.FOLHA_BR[pf])} ({rf['prob'][pf]:.0%})", (0, 140, 0) if f_ok else (200, 0, 0)),
                   (f"Juntos: {_popular('folha_br', fus)} ({pu.max():.0%})", (0, 140, 0) if u_ok else (200, 0, 0))])
        alvo.append(np.hstack([a, np.full((a.shape[0], 6, 3), 255, np.uint8), b, np.full((a.shape[0], 24, 3), 255, np.uint8)]))
        registro.append({"modo": "tronco+folha", "grupo": "corrigido pela fusão" if u_ok else "erro da fusão",
                         "foto_tronco": lt["caminho"], "foto_folha": lf["caminho"], "real": real,
                         "tronco": dados.TRONCO_BR[pt], "folha": dados.FOLHA_BR[pf], "fusao": fus})
        if len(corrigidos) >= 3 and len(falhos) >= 2:
            break
    if corrigidos:
        _salvar("br_fusao_corrigidos.jpg", corrigidos)
    if falhos:
        _salvar("br_fusao_erros.jpg", falhos)
    print("fusão", len(corrigidos), "corrigidos,", len(falhos), "erros, de", len(pares), "pares")


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    registro: list[dict] = []
    galeria_modo("tronco_br", rng, registro)
    galeria_modo("folha_br", rng, registro)
    galeria_fusao(rng, registro)
    (RAIZ / "docs" / "galeria_br.json").write_text(json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

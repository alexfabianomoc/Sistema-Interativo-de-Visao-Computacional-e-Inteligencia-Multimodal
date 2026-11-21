"""Galeria de acertos e erros com o mapa de calor (5 + 5 por modo), com o método padrão de cada modo.

Casca: fotos de campo do GBIF (o teste do TRUNK12 tem 100% e não teria erros para mostrar).
Folha: teste das folhas do GBIF.
Saída: docs/galeria/<modo>_acertos.jpg, <modo>_erros.jpg e docs/galeria.md
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.metrics import dados  # noqa: E402
from src.models import servico  # noqa: E402
from src.preprocessing.transforms import ler_arquivo  # noqa: E402
from src.visualization import desenho, gradcam  # noqa: E402

SAIDA = RAIZ / "docs" / "galeria"
LADO = 300


def cartao(img, mapa, real, prev, conf) -> np.ndarray:
    vis = desenho.mapa_de_calor(img, mapa, 0.4)
    h, w = vis.shape[:2]
    e = LADO / max(h, w)
    vis = cv2.resize(vis, (round(w * e), round(h * e)), interpolation=cv2.INTER_AREA)
    tela = np.full((LADO + 58, LADO, 3), 255, np.uint8)
    y0, x0 = (LADO - vis.shape[0]) // 2, (LADO - vis.shape[1]) // 2
    tela[y0:y0 + vis.shape[0], x0:x0 + vis.shape[1]] = vis
    ok = real == prev
    cv2.putText(tela, f"Real: {dados.ESPECIES[dados.CLASSES[real]][0]}", (6, LADO + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(tela, f"Previsto: {dados.ESPECIES[dados.CLASSES[prev]][0]} ({conf:.0%})", (6, LADO + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 0) if ok else (200, 0, 0), 1, cv2.LINE_AA)
    return tela


def main() -> None:
    cfg = yaml.safe_load(open(RAIZ / "config.yaml", encoding="utf-8"))
    rng = random.Random(7)
    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas = ["# Galeria de acertos e erros", "",
              "Cada cartão mostra a foto com o mapa de calor Grad-CAM da espécie prevista. "
              "Verde: acerto; vermelho: erro.", ""]
    for modo, itens, origem in (("casca", dados.casca_campo(), "fotos de campo do GBIF"),
                                ("folha", dados.folhas()["test"], "teste das folhas do GBIF")):
        metodo = cfg["metodo_padrao"][modo]
        clahe = cfg[f"clahe_{modo}"]
        m = servico.carregar(metodo, modo)
        rng.shuffle(itens)
        acertos, erros = [], []
        for p, real in itens:
            if len(acertos) >= 5 and len(erros) >= 5:
                break
            img = ler_arquivo(p)
            r = servico.classificar(m, metodo, modo, img, 224, clahe)
            prev = int(r["prob"].argmax())
            alvo = acertos if prev == real else erros
            if len(alvo) < 5:
                mapa = gradcam.mapa_completo(m, img, r, prev, 224, clahe)
                alvo.append((cartao(img, mapa, real, prev, float(r["prob"][prev])), p, real, prev))
        for nome, grupo in (("acertos", acertos), ("erros", erros)):
            faixa = np.hstack([c for c, *_ in grupo] + [np.full((LADO + 58, 1, 3), 255, np.uint8)])
            cv2.imencode(".jpg", cv2.cvtColor(faixa, cv2.COLOR_RGB2BGR))[1].tofile(str(SAIDA / f"{modo}_{nome}.jpg"))
        linhas += [f"## {modo.capitalize()} ({metodo}, {origem})", "",
                   f"![Acertos](galeria/{modo}_acertos.jpg)", "", f"![Erros](galeria/{modo}_erros.jpg)", "",
                   "| Foto | Real | Previsto |", "|---|---|---|"]
        linhas += [f"| `{p.relative_to(dados.DATA).as_posix()}` | {dados.ESPECIES[dados.CLASSES[real]][0]} | "
                   f"{dados.ESPECIES[dados.CLASSES[prev]][0]} |" for _, p, real, prev in erros]
        linhas += ["", "Comentário dos erros: <!-- preencher após ver as imagens -->", ""]
        print(modo, len(acertos), "acertos,", len(erros), "erros")
    (RAIZ / "docs" / "galeria.md").write_text("\n".join(linhas), encoding="utf-8")


if __name__ == "__main__":
    main()

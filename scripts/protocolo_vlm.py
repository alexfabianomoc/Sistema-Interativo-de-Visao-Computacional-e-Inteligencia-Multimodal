"""Protocolo de validação do VLM (seção 9 do enunciado): 20 casos fixos.

- 12 perguntas fechadas (sim/não) com resposta esperada conhecida: classificadas automaticamente.
- 8 perguntas abertas (descrição e perguntas livres): respostas gravadas para classificação manual
  em correta, incompleta, incorreta ou alucinada (coluna preenchida pelo autor após ver a imagem).

Saída: docs/vlm_casos.json e docs/vlm_protocolo.md (as respostas abertas ficam com "a classificar").
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.models.traducao import Tradutor  # noqa: E402
from src.models.vlm import PROMPTS, PaliGemma  # noqa: E402
from src.preprocessing.transforms import ler_arquivo  # noqa: E402

AM = RAIZ / "data_samples"
TRONCO, FOLHA = PROMPTS["tem_tronco"], PROMPTS["tem_folha"]


def imagem_sem_planta() -> np.ndarray:
    """Imagem sintética sem planta (formas geométricas e texto), para testar alucinação."""
    img = np.full((480, 640, 3), 235, np.uint8)
    cv2.rectangle(img, (60, 80), (300, 360), (40, 60, 200), -1)
    cv2.circle(img, (470, 220), 110, (200, 150, 30), -1)
    cv2.putText(img, "TESTE 123", (120, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    return img


CASOS = [
    # (id, imagem, prompt, esperado: "yes"/"no" para fechadas, None para abertas)
    ("F01", "casca/oak.jpg", TRONCO, "yes"),
    ("F02", "casca/beech.jpg", TRONCO, "yes"),
    ("F03", "casca/pine.jpg", TRONCO, "yes"),
    ("F04", "casca/birch.jpg", TRONCO, "yes"),
    ("F05", "folha/oak.jpg", TRONCO, "no"),
    ("F06", "folha/ginkgo_biloba.jpg", TRONCO, "no"),
    ("F07", "folha/linden.jpg", FOLHA, "yes"),
    ("F08", "folha/hornbeam.jpg", FOLHA, "yes"),
    ("F09", "folha/spruce.jpg", FOLHA, "yes"),
    ("F10", "SINTETICA", FOLHA, "no"),
    ("F11", "SINTETICA", TRONCO, "no"),
    ("F12", "casca/spruce.jpg", "answer en Is this a photo of a leaf?", "no"),
    ("A01", "casca/oak.jpg", PROMPTS["descricao"], None),
    ("A02", "folha/chestnut.jpg", PROMPTS["descricao"], None),
    ("A03", "folha/horse_chestnut.jpg", PROMPTS["descreve_folha"], None),
    ("A04", "folha/ginkgo_biloba.jpg", PROMPTS["descreve_folha"], None),
    ("A05", "casca/birch.jpg", "answer en What color is the bark?", None),
    ("A06", "folha/alder.jpg", "answer en How many leaves are visible?", None),
    ("A07", "casca/linden.jpg", "answer en What tree species is this?", None),
    ("A08", "SINTETICA", PROMPTS["descricao"], None),
]


def main() -> None:
    vlm, tr = PaliGemma(), Tradutor()
    resultados = []
    for ident, arq, prompt, esperado in CASOS:
        img = imagem_sem_planta() if arq == "SINTETICA" else ler_arquivo(AM / arq)
        r = vlm.perguntar(img, prompt, 5 if esperado else 60)
        texto = r.texto.strip().lower()
        if esperado:
            classe = "correta" if texto.startswith(esperado) else "incorreta"
        else:
            classe = "a classificar"
        resultados.append({"id": ident, "imagem": arq, "prompt": prompt, "esperado": esperado,
                           "resposta": r.texto, "traducao": tr.para_portugues(r.texto),
                           "ms": round(r.ms), "classificacao": classe})
        print(ident, classe, r.texto[:90], f"{r.ms:.0f} ms")
    (RAIZ / "docs" / "vlm_casos.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    lat = [x["ms"] for x in resultados]
    print(f"latência mediana {statistics.median(lat):.0f} ms, máx {max(lat)} ms")


if __name__ == "__main__":
    main()

"""Camada entre a interface e os modelos: carrega pesos, classifica, faz a fusão casca + folha."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from src.metrics.dados import CLASSES, ESPECIES
from src.models import classico, cnn
from src.preprocessing import transforms as T

RAIZ = Path(__file__).resolve().parents[2]
MODELOS = RAIZ / "models"
NOMES = [f"{ESPECIES[c][0]} ({ESPECIES[c][1]})" for c in CLASSES]


class PesosAusentes(FileNotFoundError):
    pass


def caminho_pesos(metodo: str, modo: str) -> Path:
    ext = "joblib" if metodo == "classico" else "pt"
    p = MODELOS / f"{metodo}_{modo}.{ext}"
    if not p.exists():
        raise PesosAusentes(f"Pesos não encontrados em {p.relative_to(RAIZ)}. "
                            "Rode `python scripts/baixar_pesos.py` (veja o README).")
    return p


def carregar(metodo: str, modo: str):
    p = caminho_pesos(metodo, modo)
    return joblib.load(p) if metodo == "classico" else cnn.Classificador.carregar(p)


def classificar(modelo, metodo: str, modo: str, img: np.ndarray, lado_patch: int, clahe: float) -> dict:
    if metodo == "classico":
        t0 = time.perf_counter()
        x = classico.caracteristicas_casca(img, clahe) if modo == "casca" else classico.caracteristicas_folha(img)
        prob = np.zeros(len(CLASSES))
        prob[modelo.classes_] = modelo.predict_proba(x[None])[0]
        return {"prob": prob, "prob_partes": None, "posicoes": None, "ms": (time.perf_counter() - t0) * 1000}
    return modelo.classificar(img, lado_patch, clahe)


def fusao(prob_casca: np.ndarray, prob_folha: np.ndarray) -> np.ndarray:
    """Fusão tardia: produto das probabilidades (independência entre os modos), renormalizado."""
    p = prob_casca * prob_folha
    return p / p.sum() if p.sum() > 0 else (prob_casca + prob_folha) / 2


def top_k(prob: np.ndarray, k: int) -> list[tuple[int, float]]:
    idx = np.argsort(prob)[::-1][:k]
    return [(int(i), float(prob[i])) for i in idx]


def preparar(img: np.ndarray, roi: tuple[float, float, float, float], graus: float) -> np.ndarray:
    """Rotação (item 3) e depois recorte da ROI em frações da imagem rotacionada (item 9)."""
    rot = T.rotacionar(img, graus)
    h, w = rot.shape[:2]
    fx, fy, fw, fh = roi
    return T.recortar(rot, (int(fx * w), int(fy * h), int(fw * w), int(fh * h)))

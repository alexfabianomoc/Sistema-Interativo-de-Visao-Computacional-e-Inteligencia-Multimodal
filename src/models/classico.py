"""Linhas de base clássicas (sem rede neural).

- Casca: textura LBP (duas escalas) + histograma de cor HSV + SVM (Ratajczak et al., 2019)
- Folha: forma do maior contorno (momentos de Hu, razão de aspecto, solidez, circularidade)
  + histograma de cor HSV dentro da máscara + SVM
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocessing import transforms as T

LADO_ANALISE = 512  # menor lado usado na extração (fotos do TRUNK12 têm 1000×1334)


def _reduzir(img: np.ndarray, lado: int = LADO_ANALISE) -> np.ndarray:
    h, w = img.shape[:2]
    escala = lado / min(h, w)
    if escala >= 1:
        return img
    return cv2.resize(img, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA)


def _hist_cor(img: np.ndarray, mascara: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1], mascara, [16, 8], [0, 180, 0, 256]).ravel()
    return h / max(h.sum(), 1)


def _hist_lbp(cinza: np.ndarray, p: int, r: int) -> np.ndarray:
    lbp = local_binary_pattern(cinza, p, r, method="uniform")
    h, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2))
    return h / max(h.sum(), 1)


def caracteristicas_casca(img: np.ndarray, clahe: float = 3.0) -> np.ndarray:
    img = T.clahe_l(_reduzir(img), clahe)
    cinza = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.concatenate([_hist_lbp(cinza, 8, 1), _hist_lbp(cinza, 16, 2), _hist_lbp(cinza, 24, 3),
                           _hist_cor(img)])


def caracteristicas_folha(img: np.ndarray) -> np.ndarray:
    img = _reduzir(img, 384)
    mascara, contorno = T.segmentar_folha(img)
    if contorno is None:  # sem folha detectada: forma neutra, só a cor da imagem toda
        return np.concatenate([np.zeros(10), _hist_cor(img)])
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    casco = cv2.convexHull(contorno)
    (_, _), (lw, lh), _ = cv2.minAreaRect(contorno)
    hu = cv2.HuMoments(cv2.moments(contorno)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)  # escala logarítmica usual dos momentos de Hu
    forma = [min(lw, lh) / max(lw, lh, 1e-6),                    # razão de aspecto
             area / max(cv2.contourArea(casco), 1e-6),           # solidez
             4 * np.pi * area / max(perimetro ** 2, 1e-6)]       # circularidade
    return np.concatenate([hu, forma, _hist_cor(img, mascara)])


def novo_svm(c: float = 10.0) -> object:
    """SVM RBF com probabilidades calibradas (necessárias para o top-3 e para a fusão)."""
    svm = SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced", random_state=42)
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(svm, ensemble=False))

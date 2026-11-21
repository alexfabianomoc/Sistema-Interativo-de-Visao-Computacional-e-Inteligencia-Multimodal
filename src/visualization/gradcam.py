"""Grad-CAM (Selvaraju et al., 2017) e montagem do mapa de calor na imagem original (item 1)."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from src.models import cnn
from src.preprocessing import transforms as T


def gradcam(clf: cnn.Classificador, img224: np.ndarray, classe: int) -> np.ndarray:
    """Mapa 224×224 em [0, 1] de onde a rede olhou para decidir `classe`."""
    camada = cnn.camada_gradcam(clf.modelo, clf.arquitetura)
    guard: dict = {}
    h1 = camada.register_forward_hook(lambda m, i, o: guard.__setitem__("a", o))
    h2 = camada.register_full_backward_hook(lambda m, gi, go: guard.__setitem__("g", go[0]))
    try:
        x = torch.from_numpy(T.normalizar(img224)[None]).to(clf.disp).requires_grad_(True)
        clf.modelo.zero_grad(set_to_none=True)
        clf.modelo(x)[0, classe].backward()
        pesos = guard["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((pesos * guard["a"]).sum(1))[0].detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()
    cam = cv2.resize(cam, (cnn.LADO, cnn.LADO), interpolation=cv2.INTER_LINEAR)
    return cam / cam.max() if cam.max() > 0 else cam


def mapa_completo(clf: cnn.Classificador, img: np.ndarray, resultado: dict, classe: int,
                  lado_patch: int = cnn.LADO, clahe: float | None = None) -> np.ndarray:
    """Mapa de calor no tamanho de `img`: cada recorte que a rede viu leva seu Grad-CAM de volta
    à posição de origem (mapeamento de coordenadas), com média onde os recortes se sobrepõem."""
    h, w = img.shape[:2]
    soma = np.zeros((h, w), np.float32)
    cont = np.zeros((h, w), np.float32)
    entradas, posicoes = clf.entradas(img, lado_patch, clahe)
    for ent, (x, y, s) in zip(entradas, posicoes):
        cam = gradcam(clf, ent, classe)
        if clf.modo == "casca":  # patch quadrado: escala direta
            reg = cv2.resize(cam, (s, s))
            x1, y1 = min(w, x + s), min(h, y + s)
            soma[y:y1, x:x1] += reg[: y1 - y, : x1 - x]
            cont[y:y1, x:x1] += 1
        else:  # recorte em letterbox: desfaz as bordas antes de devolver à imagem
            sub_h, sub_w = (h, w) if s == max(h, w) and (x, y) == (0, 0) else (s, s)
            _, lb = T.letterbox(np.zeros((sub_h, sub_w, 3), np.uint8), cnn.LADO)
            util = cam[lb.dy: lb.dy + round(sub_h * lb.escala), lb.dx: lb.dx + round(sub_w * lb.escala)]
            reg = cv2.resize(util, (sub_w, sub_h))
            soma[y:y + sub_h, x:x + sub_w] += reg
            cont[y:y + sub_h, x:x + sub_w] += 1
    mapa = soma / np.maximum(cont, 1)
    # suaviza as emendas entre recortes (sigma proporcional ao tamanho da imagem)
    mapa = cv2.GaussianBlur(mapa, (0, 0), sigmaX=max(h, w) / 60)
    return mapa / mapa.max() if mapa.max() > 0 else mapa

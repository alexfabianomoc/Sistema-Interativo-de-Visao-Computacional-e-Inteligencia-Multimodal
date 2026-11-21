"""CNNs pré-treinadas no ImageNet com ajuste fino (ConvNeXt-Tiny, MobileNetV3-Large e ResNet-18) e a inferência
de cada modo: casca = voto médio de patches; folha = imagem inteira + 5 recortes."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

from src.preprocessing import transforms as T

ARQUITETURAS = ("mobilenet_v3_large", "resnet18", "convnext_tiny")
LR_MAXIMA = {"mobilenet_v3_large": 1e-3, "resnet18": 1e-3, "convnext_tiny": 3e-4}  # ConvNeXt pede taxa menor
CLAHE_PADRAO = {"casca": 3.0, "folha": 1.5}
LADO = 224


def dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def criar(arquitetura: str, n_classes: int, pretreinada: bool = True) -> nn.Module:
    if arquitetura == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights="IMAGENET1K_V2" if pretreinada else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    elif arquitetura == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1" if pretreinada else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arquitetura == "convnext_tiny":
        m = models.convnext_tiny(weights="IMAGENET1K_V1" if pretreinada else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    else:
        raise ValueError(f"arquitetura desconhecida: {arquitetura}")
    return m


def camada_gradcam(m: nn.Module, arquitetura: str) -> nn.Module:
    """Última camada convolucional, usada pelo Grad-CAM."""
    if arquitetura == "convnext_tiny":
        return m.features[-1][-1]  # último bloco ConvNeXt do estágio 4 (saída 7×7)
    return m.features[-1] if arquitetura == "mobilenet_v3_large" else m.layer4[-1]


@dataclass
class Classificador:
    modelo: nn.Module
    arquitetura: str
    modo: str
    classes: list[str]
    clahe: float
    disp: torch.device = field(default_factory=dispositivo)

    @classmethod
    def carregar(cls, caminho: Path, disp: torch.device | None = None) -> "Classificador":
        disp = disp or dispositivo()
        ck = torch.load(caminho, map_location=disp, weights_only=False)
        m = criar(ck["arquitetura"], len(ck["classes"]), pretreinada=False)
        m.load_state_dict(ck["estado"])
        m.to(disp).eval()
        return cls(m, ck["arquitetura"], ck["modo"], ck["classes"], ck["clahe"], disp)

    def _lote(self, imgs: list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack([T.normalizar(i) for i in imgs])).to(self.disp)

    @torch.no_grad()
    def probabilidades(self, imgs: list[np.ndarray]) -> np.ndarray:
        """Probabilidades por imagem já em 224×224 (sem realce: aplique antes)."""
        return torch.softmax(self.modelo(self._lote(imgs)), 1).cpu().numpy()

    def entradas(self, img: np.ndarray, lado_patch: int = LADO,
                 clahe: float | None = None) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
        """Recortes que a rede vê e onde cada um fica na imagem (x, y, lado), para o desenho."""
        img = T.clahe_l(img, self.clahe if clahe is None else clahe)
        if self.modo == "casca":
            base = reduzir_menor_lado(img, 2 * lado_patch)
            esc = img.shape[0] / base.shape[0]
            ps = T.patches(base, lado_patch)
            redim = [p if lado_patch == LADO else T.letterbox(p, LADO)[0] for _, _, p in ps]
            return redim, [(round(x * esc), round(y * esc), round(lado_patch * esc)) for x, y, _ in ps]
        return cinco_recortes(img)

    def classificar(self, img: np.ndarray, lado_patch: int = LADO, clahe: float | None = None) -> dict:
        t0 = time.perf_counter()
        entradas, posicoes = self.entradas(img, lado_patch, clahe)
        prob_partes = self.probabilidades(entradas)
        prob = prob_partes.mean(0)
        if self.disp.type == "cuda":
            torch.cuda.synchronize()
        return {"prob": prob, "prob_partes": prob_partes, "posicoes": posicoes,
                "ms": (time.perf_counter() - t0) * 1000}


def reduzir_menor_lado(img: np.ndarray, lado: int) -> np.ndarray:
    h, w = img.shape[:2]
    e = lado / min(h, w)
    return cv2.resize(img, (round(w * e), round(h * e)), interpolation=cv2.INTER_AREA if e < 1 else cv2.INTER_LINEAR)


def cinco_recortes(img: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    """Imagem inteira (letterbox) + 4 cantos e centro, cada um com 80% do menor lado."""
    h, w = img.shape[:2]
    s = int(0.8 * min(h, w))
    cantos = [(0, 0), (w - s, 0), (0, h - s), (w - s, h - s), ((w - s) // 2, (h - s) // 2)]
    entradas = [T.letterbox(img, LADO)[0]] + [T.letterbox(img[y:y + s, x:x + s], LADO)[0] for x, y in cantos]
    return entradas, [(0, 0, max(h, w))] + [(x, y, s) for x, y in cantos]

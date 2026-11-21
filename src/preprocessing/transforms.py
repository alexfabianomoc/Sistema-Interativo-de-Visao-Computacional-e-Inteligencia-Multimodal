"""Operações de pré-processamento (seção 5 do enunciado) e itens de CG 1, 2, 3, 5, 6 e 9.

Todas as funções recebem e devolvem imagens NumPy em RGB, uint8, formato (altura, largura, 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TAMANHO_MINIMO = 224
MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImagemInvalida(ValueError):
    """Erro com mensagem pronta para mostrar ao usuário."""


def decodificar(dados: bytes) -> np.ndarray:
    """Abre bytes de JPG/PNG como RGB; mensagens claras para os casos de falha."""
    arr = np.frombuffer(dados, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImagemInvalida("Não foi possível abrir este arquivo. Envie um JPG ou PNG.")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    validar_tamanho(img)
    return img


def ler_arquivo(caminho) -> np.ndarray:
    """Lê imagem do disco como RGB. Não usa cv2.imread, que no Windows falha em caminhos
    com acentos (como "Computação Gráfica")."""
    bgr = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImagemInvalida(f"Não foi possível abrir {caminho}.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def validar_tamanho(img: np.ndarray, minimo: int = TAMANHO_MINIMO) -> None:
    h, w = img.shape[:2]
    if min(h, w) < minimo:
        raise ImagemInvalida(f"A imagem tem {w}×{h} px; o mínimo é {minimo} px no menor lado. "
                             "Tire uma foto mais próxima.")


# --- Item 9: região de interesse ------------------------------------------------------------

def roi_central(img: np.ndarray, fracao: float = 0.6) -> tuple[int, int, int, int]:
    """ROI padrão: faixa central (x, y, largura, altura) cobrindo `fracao` de cada lado."""
    h, w = img.shape[:2]
    rw, rh = int(w * fracao), int(h * fracao)
    return (w - rw) // 2, (h - rh) // 2, rw, rh


def recortar(img: np.ndarray, roi: tuple[int, int, int, int], minimo: int = 32) -> np.ndarray:
    x, y, rw, rh = roi
    h, w = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + rw), min(h, y + rh)
    if x1 - x0 < minimo or y1 - y0 < minimo:
        raise ImagemInvalida("O recorte ficou vazio ou pequeno demais. Refaça a seleção.")
    return img[y0:y1, x0:x1].copy()


# --- Item 3: transformação geométrica (rotação) ---------------------------------------------

def rotacionar(img: np.ndarray, graus: float) -> np.ndarray:
    """Gira em torno do centro sem cortar os cantos (a tela cresce para caber a imagem)."""
    if abs(graus) < 1e-6:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), graus, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# --- Itens 1 e 2: redimensionamento com proporção e mapeamento de coordenadas ---------------

@dataclass(frozen=True)
class Letterbox:
    """Guarda escala e deslocamento para levar coordenadas de volta à imagem original (item 1)."""
    escala: float
    dx: int
    dy: int
    largura_orig: int
    altura_orig: int

    def para_original(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.dx) / self.escala, (y - self.dy) / self.escala

    def para_letterbox(self, x: float, y: float) -> tuple[float, float]:
        return x * self.escala + self.dx, y * self.escala + self.dy


def letterbox(img: np.ndarray, lado: int = 224, cor: int = 0) -> tuple[np.ndarray, Letterbox]:
    """Redimensiona mantendo a proporção e completa com bordas até `lado`×`lado`."""
    h, w = img.shape[:2]
    escala = lado / max(h, w)
    nw, nh = round(w * escala), round(h * escala)
    interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    redim = cv2.resize(img, (nw, nh), interpolation=interp)
    dx, dy = (lado - nw) // 2, (lado - nh) // 2
    saida = np.full((lado, lado, 3), cor, dtype=np.uint8)
    saida[dy:dy + nh, dx:dx + nw] = redim
    return saida, Letterbox(escala, dx, dy, w, h)


# --- Realce: CLAHE no canal L ---------------------------------------------------------------

def clahe_l(img: np.ndarray, limite: float = 2.0, grade: int = 8) -> np.ndarray:
    """CLAHE só na luminância (LAB), para realçar textura sem distorcer a cor.
    Casca: limite ~3 (forte). Folha: limite ~1.5 (suave). limite <= 0 desliga."""
    if limite <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(limite), tileGridSize=(grade, grade))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def normalizar(img: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 (3, H, W) normalizado com média e desvio do ImageNet."""
    x = img.astype(np.float32) / 255.0
    x = (x - MEDIA_IMAGENET) / DESVIO_IMAGENET
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# --- Modo folha: segmentação por cor + morfologia (itens 5 e 6) -----------------------------

def segmentar_folha(img: np.ndarray, kernel: int = 7) -> tuple[np.ndarray, np.ndarray | None]:
    """Máscara de vegetação por cor em HSV, limpa com abertura e fechamento.
    Devolve (máscara 0/255 só com o maior contorno, contorno ou None se não achar folha)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Verde a amarelo-esverdeado; saturação e brilho mínimos para descartar céu, chão cinza e sombra.
    mascara = cv2.inRange(hsv, (25, 40, 30), (95, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k, iterations=2)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return np.zeros_like(mascara), None
    maior = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(maior) < 0.01 * mascara.size:
        return np.zeros_like(mascara), None
    limpa = np.zeros_like(mascara)
    cv2.drawContours(limpa, [maior], -1, 255, thickness=cv2.FILLED)
    return limpa, maior


def patches(img: np.ndarray, lado: int = 224, passo: int | None = None) -> list[tuple[int, int, np.ndarray]]:
    """Divide a imagem em patches lado×lado (modo casca). Devolve (x, y, patch) na imagem dada."""
    passo = passo or lado
    h, w = img.shape[:2]
    if min(h, w) < lado:
        return [(0, 0, letterbox(img, lado)[0])]
    xs = list(range(0, w - lado + 1, passo))
    ys = list(range(0, h - lado + 1, passo))
    if xs[-1] != w - lado:
        xs.append(w - lado)
    if ys[-1] != h - lado:
        ys.append(h - lado)
    return [(x, y, img[y:y + lado, x:x + lado]) for y in ys for x in xs]

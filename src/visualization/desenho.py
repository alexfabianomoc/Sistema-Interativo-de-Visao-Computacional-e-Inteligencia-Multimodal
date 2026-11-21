"""Camadas gráficas sobre a imagem (itens de CG 4, 5, 6 e 10): mapa de calor com transparência,
grade de patches colorida por espécie, contorno e máscara da folha, legenda e vista comparativa."""
from __future__ import annotations

import cv2
import numpy as np

# Paleta categórica de 12 cores distinguíveis (RGB), uma por espécie, na ordem de dados.CLASSES
PALETA = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
          (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207), (0, 0, 128), (128, 128, 0)]


def mapa_de_calor(img: np.ndarray, mapa: np.ndarray, opacidade: float = 0.45) -> np.ndarray:
    """Sobrepõe o mapa (0 a 1) com a escala JET e transparência (alpha blending)."""
    cor = cv2.applyColorMap((np.clip(mapa, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cor = cv2.cvtColor(cor, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - opacidade, cor, opacidade, 0)


def grade_patches(img: np.ndarray, posicoes, prob_partes: np.ndarray, opacidade: float = 0.35,
                  limiar: float = 0.0) -> np.ndarray:
    """Pinta cada patch com a cor da espécie que ele votou; borda mais grossa = mais confiança."""
    camada = img.copy()
    saida = img.copy()
    for (x, y, s), p in zip(posicoes, prob_partes):
        c = int(p.argmax())
        if p[c] < limiar:
            continue
        cv2.rectangle(camada, (x, y), (x + s, y + s), PALETA[c], -1)
    saida = cv2.addWeighted(camada, opacidade, saida, 1 - opacidade, 0)
    esp = max(2, img.shape[1] // 300)
    for (x, y, s), p in zip(posicoes, prob_partes):
        c = int(p.argmax())
        cv2.rectangle(saida, (x, y), (x + s - 1, y + s - 1), PALETA[c], max(1, int(esp * (0.5 + p[c]))))
        cv2.putText(saida, f"{p[c]:.0%}", (x + 6, y + 6 + 14 * esp), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * esp, (255, 255, 255), max(1, esp // 2) + 1, cv2.LINE_AA)
    return saida


def contorno_folha(img: np.ndarray, mascara: np.ndarray, contorno, opacidade: float = 0.3,
                   cor=(0, 255, 0)) -> np.ndarray:
    """Máscara semitransparente + contorno desenhado (itens 5 e 6)."""
    if contorno is None:
        return img
    camada = img.copy()
    camada[mascara > 0] = cor
    saida = cv2.addWeighted(camada, opacidade, img, 1 - opacidade, 0)
    cv2.drawContours(saida, [contorno], -1, (255, 255, 0), max(2, img.shape[1] // 250), cv2.LINE_AA)
    return saida


def legenda(nomes: list[str], indices: list[int], largura: int = 320) -> np.ndarray:
    """Quadro de legenda (cor -> espécie) para colar ao lado da imagem exportada."""
    linha = 28
    img = np.full((linha * len(indices) + 12, largura, 3), 255, np.uint8)
    for k, c in enumerate(indices):
        y = 8 + k * linha
        cv2.rectangle(img, (8, y), (30, y + 20), PALETA[c], -1)
        cv2.putText(img, nomes[c], (40, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def barra_escala_calor(altura: int = 200, largura: int = 24) -> np.ndarray:
    """Escala do mapa de calor (alto em cima, baixo embaixo)."""
    grad = np.linspace(255, 0, altura).astype(np.uint8)[:, None].repeat(largura, 1)
    return cv2.cvtColor(cv2.applyColorMap(grad, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def lado_a_lado(imagens: list[np.ndarray], titulos: list[str], altura: int = 360) -> np.ndarray:
    """Vista comparativa (item 10) com títulos, todas na mesma altura."""
    blocos = []
    for im, t in zip(imagens, titulos):
        e = altura / im.shape[0]
        r = cv2.resize(im, (round(im.shape[1] * e), altura), interpolation=cv2.INTER_AREA)
        topo = np.full((30, r.shape[1], 3), 255, np.uint8)
        cv2.putText(topo, t, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        blocos.append(np.vstack([topo, r]))
        blocos.append(np.full((altura + 30, 8, 3), 255, np.uint8))
    return np.hstack(blocos[:-1])


def compor_exportacao(img: np.ndarray, nomes: list[str], indices: list[int]) -> np.ndarray:
    """Imagem final com a legenda ao lado, pronta para baixar em PNG."""
    leg = legenda(nomes, indices)
    esc = barra_escala_calor(min(200, img.shape[0] - leg.shape[0] - 20) if img.shape[0] > leg.shape[0] + 60 else 120)
    painel = np.full((img.shape[0], leg.shape[1], 3), 255, np.uint8)
    painel[: leg.shape[0]] = leg[: img.shape[0]]
    y0 = leg.shape[0] + 10
    if y0 + esc.shape[0] <= img.shape[0]:
        painel[y0:y0 + esc.shape[0], 8:8 + esc.shape[1]] = esc
        cv2.putText(painel, "alta atencao", (40, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        cv2.putText(painel, "baixa", (40, y0 + esc.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    return np.hstack([img, painel])


def png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes()

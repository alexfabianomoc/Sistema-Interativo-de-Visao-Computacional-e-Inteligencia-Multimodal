"""Teste rápido do PaliGemma: tempo de carga, VRAM, latência e prompts em inglês e português."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.models.vlm import PaliGemma  # noqa: E402
from src.preprocessing.transforms import ler_arquivo  # noqa: E402

t0 = time.perf_counter()
vlm = PaliGemma()
print(f"carga: {time.perf_counter() - t0:.1f}s; VRAM: {torch.cuda.memory_allocated() / 2**30:.2f} GB")

casca = ler_arquivo(next((RAIZ / "data" / "trunk12" / "oak").glob("*.JPG")))
folha = ler_arquivo(next((RAIZ / "data" / "folhas" / "test" / "oak").glob("*.jpg")))
testes = [
    (casca, "caption en"), (casca, "caption pt"),
    (casca, "answer en Is there a tree trunk or tree bark in this image?"),
    (casca, "answer pt Há um tronco de árvore nesta imagem?"),
    (folha, "caption en"), (folha, "caption pt"),
    (folha, "answer en Describe the shape, edge and color of the leaf."),
    (folha, "answer pt Qual é a cor da folha?"),
    (folha, "answer en Is there a tree trunk or tree bark in this image?"),
]
for img, p in testes:
    r = vlm.perguntar(img, p)
    print(f"{r.ms:7.0f} ms | {p} -> {r.texto}")
print(f"pico de VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

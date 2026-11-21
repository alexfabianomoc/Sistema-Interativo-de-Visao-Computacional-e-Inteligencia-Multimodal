"""Interpretação multimodal local com PaliGemma 2 3B mix 224 (componente 7, opcional).

Cada chamada é uma rodada única imagem + texto. Recebe só o recorte (sem EXIF nem GPS).
A resposta é apresentada como interpretação do modelo, nunca como certeza.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

MODELO = "google/paligemma2-3b-mix-224"

# Prompts do formato da variante mix ("caption <idioma>", "answer <idioma> <pergunta>")
PROMPTS = {
    "descricao": "caption en",
    "tem_tronco": "answer en Is there a tree trunk or tree bark in this image?",
    "tem_folha": "answer en Is there a leaf or a plant in this image?",
    "descreve_folha": "answer en Describe the shape, edge and color of the leaf.",
}


class VLMIndisponivel(RuntimeError):
    """Modelo não baixado, sem GPU/memória ou bibliotecas ausentes: o painel fica desativado."""


@dataclass
class Resposta:
    texto: str
    prompt: str
    ms: float


class PaliGemma:
    def __init__(self, quatro_bits: bool = True):
        try:
            import torch
            from transformers import (AutoProcessor, BitsAndBytesConfig,
                                      PaliGemmaForConditionalGeneration)
        except ImportError as e:
            raise VLMIndisponivel(f"Bibliotecas do VLM ausentes: {e}") from e
        self._torch = torch
        usar_gpu = torch.cuda.is_available()
        kwargs: dict = {}
        if usar_gpu and quatro_bits:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
            kwargs["device_map"] = "cuda:0"
        else:
            kwargs["dtype"] = torch.bfloat16 if usar_gpu else torch.float32
            kwargs["device_map"] = "cuda:0" if usar_gpu else "cpu"
        try:
            self.processor = AutoProcessor.from_pretrained(MODELO)
            self.modelo = PaliGemmaForConditionalGeneration.from_pretrained(MODELO, **kwargs).eval()
        except OSError as e:
            raise VLMIndisponivel("PaliGemma não encontrado. Aceite os termos em huggingface.co/"
                                  f"{MODELO}, rode `hf auth login` e `hf download {MODELO}`.") from e
        self.dispositivo = "cuda" if usar_gpu else "cpu"

    def perguntar(self, img: np.ndarray, prompt: str, max_tokens: int = 60) -> Resposta:
        from PIL import Image
        torch = self._torch
        t0 = time.perf_counter()
        entradas = self.processor(text="<image>" + prompt, images=Image.fromarray(img),
                                  return_tensors="pt").to(self.modelo.device)
        n = entradas["input_ids"].shape[-1]
        with torch.inference_mode():
            saida = self.modelo.generate(**entradas, max_new_tokens=max_tokens, do_sample=False)
        texto = self.processor.decode(saida[0][n:], skip_special_tokens=True).strip()
        return Resposta(texto, prompt, (time.perf_counter() - t0) * 1000)

    def descrever(self, img: np.ndarray) -> Resposta:
        return self.perguntar(img, PROMPTS["descricao"])

    def verificar(self, img: np.ndarray, modo: str) -> tuple[bool | None, Resposta]:
        """True/False se a resposta for sim/não; None se ambígua."""
        r = self.perguntar(img, PROMPTS["tem_tronco" if modo in ("casca", "tronco_br") else "tem_folha"], 5)
        t = r.texto.lower()
        return (True if t.startswith("yes") else False if t.startswith("no") else None), r

    def pergunta_livre(self, img: np.ndarray, pergunta: str, idioma: str = "en") -> Resposta:
        return self.perguntar(img, f"answer {idioma} {pergunta.strip()}")

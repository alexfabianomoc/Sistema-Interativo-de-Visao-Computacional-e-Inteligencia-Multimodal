"""Tradução local pt <-> en para o painel do VLM (o PaliGemma funciona melhor em inglês).

- pt -> en: Helsinki-NLP/opus-mt-ROMANCE-en (Apache 2.0, ~310 MB)
- en -> pt: Helsinki-NLP/opus-mt-tc-big-en-pt (CC BY 4.0, ~470 MB)
Roda em CPU (0,5 a 3 s por frase), sem disputar a GPU com o PaliGemma.
"""
from __future__ import annotations

PT_EN = "Helsinki-NLP/opus-mt-ROMANCE-en"
EN_PT = "Helsinki-NLP/opus-mt-tc-big-en-pt"


class Tradutor:
    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        self._mods = {}
        for chave, nome in (("pt_en", PT_EN), ("en_pt", EN_PT)):
            self._mods[chave] = (MarianTokenizer.from_pretrained(nome),
                                 MarianMTModel.from_pretrained(nome).eval())

    def _traduzir(self, chave: str, texto: str, prefixo: str = "") -> str:
        import torch
        tok, mod = self._mods[chave]
        b = tok([prefixo + texto], return_tensors="pt", truncation=True)
        with torch.inference_mode():
            out = mod.generate(**b, max_new_tokens=200, num_beams=4)
        return tok.decode(out[0], skip_special_tokens=True)

    def para_ingles(self, texto: str) -> str:
        return self._traduzir("pt_en", texto)

    def para_portugues(self, texto: str) -> str:
        return self._traduzir("en_pt", texto, ">>por<< ")

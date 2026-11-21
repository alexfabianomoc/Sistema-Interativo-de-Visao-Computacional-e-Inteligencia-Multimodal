"""Comparação zero-shot com o BioCLIP (Stevens et al., 2024; imageomics/bioclip, licença MIT):
classifica as mesmas imagens de teste SEM nenhum treino nas nossas bases, comparando a imagem com
textos do tipo "a photo of <nome científico>, <nome comum>". Grava docs/metricas/bioclip.json.

Conjuntos: teste do TRUNK12 (casca), fotos de campo do GBIF (casca) e teste das folhas do GBIF.
A imagem passa pelo mesmo CLAHE do modo, mas inteira (o BioCLIP faz o próprio redimensionamento).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.metrics import dados  # noqa: E402
from src.preprocessing import transforms as T  # noqa: E402

MODELO = "hf-hub:imageomics/bioclip"
NOMES_EN = {"alder": "black alder", "beech": "European beech", "birch": "silver birch",
            "chestnut": "sweet chestnut", "ginkgo biloba": "ginkgo", "hornbeam": "European hornbeam",
            "horse chestnut": "horse chestnut", "linden": "small-leaved lime", "oak": "English oak",
            "oriental plane": "oriental plane", "pine": "Scots pine", "spruce": "Norway spruce"}


def main() -> None:
    import open_clip
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, _, prep = open_clip.create_model_and_transforms(MODELO)
    tok = open_clip.get_tokenizer(MODELO)
    modelo = modelo.to(disp).eval()
    rot = list(range(len(dados.CLASSES)))
    resultado = {"modelo": MODELO, "modo_uso": "zero-shot, sem treino", "conjuntos": {}}
    for orgao, modo_txt in (("casca", "bark"), ("folha", "leaf")):
        textos = [f"a photo of the {modo_txt} of {dados.ESPECIES[c][1]}, {NOMES_EN[c]}." for c in dados.CLASSES]
        with torch.no_grad():
            t = modelo.encode_text(tok(textos).to(disp))
            t = t / t.norm(dim=-1, keepdim=True)
        conjuntos = ({"TRUNK12 (teste)": dados.trunk12()["test"], "Casca de campo (GBIF)": dados.casca_campo()}
                     if orgao == "casca" else {"Folhas (teste GBIF)": dados.folhas()["test"]})
        for nome, itens in conjuntos.items():
            probs, tempos = [], []
            for p, _ in itens:
                img = T.clahe_l(T.ler_arquivo(p), 3.0 if orgao == "casca" else 1.5)
                t0 = time.perf_counter()
                x = prep(Image.fromarray(img)).unsqueeze(0).to(disp)
                with torch.no_grad():
                    v = modelo.encode_image(x)
                    v = v / v.norm(dim=-1, keepdim=True)
                    probs.append((100 * v @ t.T).softmax(-1)[0].cpu().numpy())
                tempos.append((time.perf_counter() - t0) * 1000)
            prob, y = np.stack(probs), np.array([c for _, c in itens])
            pred = prob.argmax(1)
            resultado["conjuntos"][nome] = {
                "n": len(itens), "acuracia": round(accuracy_score(y, pred), 4),
                "acuracia_top3": round(top_k_accuracy_score(y, prob, k=3, labels=rot), 4),
                "f1_macro": round(f1_score(y, pred, average="macro", labels=rot, zero_division=0), 4),
                "tempo_ms_mediana": round(float(np.median(tempos[1:])), 1)}
            print(nome, resultado["conjuntos"][nome])
    (RAIZ / "docs" / "metricas" / "bioclip.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")


if __name__ == "__main__":
    main()

"""Ajuste fino de uma CNN em um modo e avaliação no teste.

    python scripts/treinar_cnn.py casca mobilenet_v3_large
    python scripts/treinar_cnn.py folha resnet18 --epocas 12

Grava models/<arquitetura>_<modo>.pt (fora do Git) e docs/metricas/<arquitetura>_<modo>.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.metrics import dados  # noqa: E402
from src.models import cnn  # noqa: E402
from src.preprocessing import transforms as T  # noqa: E402

SEMENTE = 42


class Base(Dataset):
    """Carrega cada foto uma vez (reduzida), aplica o CLAHE do modo e sorteia aumentos a cada época."""

    def __init__(self, itens, modo: str, treino: bool):
        self.modo, self.treino = modo, treino
        lado_base = 2 * cnn.LADO if modo == "casca" else 320
        clahe = cnn.CLAHE_PADRAO[modo]
        self.imgs = [T.clahe_l(cnn.reduzir_menor_lado(T.ler_arquivo(p), lado_base), clahe) for p, _ in itens]
        self.rotulos = [c for _, c in itens]
        geo = [v2.RandomHorizontalFlip(), v2.RandomRotation(15)]
        cor = v2.ColorJitter(0.3, 0.3, 0.2, 0.02)
        if modo == "casca":  # textura: recorte de patch de 224 em qualquer ponto do tronco
            self.aum = v2.Compose([v2.ToImage(), v2.RandomCrop(cnn.LADO), *geo, cor])
        else:  # forma da folha: recorte com variação de escala (fotos de distâncias diferentes)
            self.aum = v2.Compose([v2.ToImage(), v2.RandomResizedCrop(cnn.LADO, scale=(0.4, 1.0)), *geo, cor])
        self.norm = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(T.MEDIA_IMAGENET.tolist(), T.DESVIO_IMAGENET.tolist())])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.norm(self.aum(self.imgs[i])), self.rotulos[i]


def avaliar(clf: cnn.Classificador, itens) -> dict:
    probs, tempos = [], []
    for p, _ in itens:
        r = clf.classificar(T.ler_arquivo(p))
        probs.append(r["prob"]); tempos.append(r["ms"])
    prob = np.stack(probs)
    y = np.array([c for _, c in itens])
    pred = prob.argmax(1)
    rot = list(range(len(dados.CLASSES)))
    return {"acuracia": round(accuracy_score(y, pred), 4),
            "acuracia_top3": round(top_k_accuracy_score(y, prob, k=3, labels=rot), 4),
            "f1_macro": round(f1_score(y, pred, average="macro", labels=rot, zero_division=0), 4),
            "tempo_ms_por_imagem_mediana": round(float(np.median(tempos[1:] or tempos)), 1),
            "matriz_confusao": confusion_matrix(y, pred, labels=rot).tolist(), "_prob": prob}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["casca", "folha"])
    ap.add_argument("arquitetura", choices=cnn.ARQUITETURAS)
    ap.add_argument("--epocas", type=int, default=None)
    ap.add_argument("--lote", type=int, default=32)
    a = ap.parse_args()
    epocas = a.epocas or (40 if a.modo == "casca" else 15)
    random.seed(SEMENTE); np.random.seed(SEMENTE); torch.manual_seed(SEMENTE)

    disp = cnn.dispositivo()
    splits = dados.carregar(a.modo)
    t0 = time.perf_counter()
    treino = Base(splits["train"], a.modo, True)
    print(f"{len(treino)} imagens de treino carregadas em {time.perf_counter() - t0:.0f}s; dispositivo {disp}")
    dl = DataLoader(treino, batch_size=a.lote, shuffle=True, num_workers=0, drop_last=True)

    modelo = cnn.criar(a.arquitetura, len(dados.CLASSES)).to(disp)
    contagem = np.bincount(treino.rotulos, minlength=len(dados.CLASSES))
    pesos = torch.tensor(contagem.sum() / (len(contagem) * np.maximum(contagem, 1)), dtype=torch.float32)
    perda = nn.CrossEntropyLoss(weight=pesos.to(disp), label_smoothing=0.1)
    opt = torch.optim.AdamW(modelo.parameters(), lr=3e-4, weight_decay=1e-4)
    passos = epocas * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=passos, pct_start=0.15)
    escala = torch.amp.GradScaler(enabled=disp.type == "cuda")

    clf = cnn.Classificador(modelo, a.arquitetura, a.modo, dados.CLASSES, cnn.CLAHE_PADRAO[a.modo], disp)
    saida_pt = RAIZ / "models" / f"{a.arquitetura}_{a.modo}.pt"
    saida_pt.parent.mkdir(exist_ok=True)
    melhor, historico = -1.0, []
    t_treino = time.perf_counter()
    for ep in range(1, epocas + 1):
        modelo.train()
        soma = 0.0
        for x, y in dl:
            x, y = x.to(disp, non_blocking=True), y.to(disp)
            with torch.autocast(disp.type, enabled=disp.type == "cuda"):
                l = perda(modelo(x), y)
            opt.zero_grad(set_to_none=True)
            escala.scale(l).backward()
            escala.step(opt); escala.update(); sched.step()
            soma += l.item()
        modelo.eval()
        val = avaliar(clf, splits["val"])
        historico.append({"epoca": ep, "perda": round(soma / len(dl), 4), "f1_val": val["f1_macro"],
                          "acc_val": val["acuracia"]})
        print(historico[-1], flush=True)
        if val["f1_macro"] > melhor:
            melhor = val["f1_macro"]
            torch.save({"arquitetura": a.arquitetura, "modo": a.modo, "classes": dados.CLASSES,
                        "clahe": cnn.CLAHE_PADRAO[a.modo], "estado": modelo.state_dict(),
                        "epoca": ep, "f1_val": melhor}, saida_pt)
    t_treino = time.perf_counter() - t_treino

    clf = cnn.Classificador.carregar(saida_pt, disp)
    teste = avaliar(clf, splits["test"])
    prob = teste.pop("_prob")
    np.save(RAIZ / "models" / f"{a.arquitetura}_{a.modo}_prob_teste.npy", prob)  # usado na fusão
    campo = dados.casca_campo() if a.modo == "casca" else []
    if campo:  # fotos de campo nunca vistas no treino
        teste_campo = avaliar(clf, campo)
        teste_campo.pop("_prob")
        teste["teste_campo"] = {k: v for k, v in teste_campo.items() if k != "matriz_confusao"} | {"n": len(campo)}
        print("teste de campo:", teste["teste_campo"])
    resultado = {"modo": a.modo, "metodo": a.arquitetura, "dispositivo": str(disp),
                 "dados_treino": "TRUNK12 + GBIF" if a.modo == "casca" and len(splits["train"]) > 300 else
                 ("TRUNK12" if a.modo == "casca" else "GBIF folhas"),
                 "n": {k: len(v) for k, v in splits.items()}, "epocas": epocas,
                 "melhor_epoca": torch.load(saida_pt, weights_only=False)["epoca"],
                 "f1_macro_val": melhor, **teste, "tempo_treino_s": round(t_treino, 1),
                 "classes": dados.CLASSES, "historico": historico}
    out = RAIZ / "docs" / "metricas"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.arquitetura}_{a.modo}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    print({k: v for k, v in resultado.items() if k not in ("matriz_confusao", "classes", "historico")})


if __name__ == "__main__":
    main()

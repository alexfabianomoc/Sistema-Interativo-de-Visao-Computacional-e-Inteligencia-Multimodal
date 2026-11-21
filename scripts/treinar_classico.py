"""Treina as linhas de base clássicas e grava pesos em models/ e métricas em docs/metricas/.

    python scripts/treinar_classico.py casca
    python scripts/treinar_classico.py folha
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.metrics import dados  # noqa: E402
from src.models import classico  # noqa: E402
from src.preprocessing.transforms import ler_arquivo  # noqa: E402


def _extrair(args: tuple[str, str]) -> np.ndarray:
    caminho, modo = args
    img = ler_arquivo(caminho)
    return classico.caracteristicas_casca(img) if dados.e_tronco(modo) else classico.caracteristicas_folha(img)


def matriz(itens, modo):
    with ProcessPoolExecutor() as ex:
        x = list(ex.map(_extrair, [(str(p), modo) for p, _ in itens], chunksize=8))
    return np.stack(x), np.array([c for _, c in itens])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=list(dados.TIPO))
    modo = ap.parse_args().modo
    splits = dados.carregar(modo)
    t0 = time.perf_counter()
    xtr, ytr = matriz(splits["train"], modo)
    xva, yva = matriz(splits["val"], modo)
    xte, yte = matriz(splits["test"], modo)
    t_extr = time.perf_counter() - t0

    # Escolhe C na validação
    melhor = max((f1_score(yva, classico.novo_svm(c).fit(xtr, ytr).predict(xva), average="macro"), c)
                 for c in (1, 3, 10, 30, 100))
    c = melhor[1]
    t0 = time.perf_counter()
    modelo = classico.novo_svm(c).fit(np.vstack([xtr, xva]), np.concatenate([ytr, yva]))
    t_treino = time.perf_counter() - t0

    t0 = time.perf_counter()
    prob = modelo.predict_proba(xte)
    t_pred = (time.perf_counter() - t0) / len(yte) * 1000
    pred = modelo.classes_[prob.argmax(1)]
    rotulos = list(range(len(dados.classes(modo))))
    resultado = {
        "modo": modo, "metodo": "LBP + cor + SVM" if dados.e_tronco(modo) else "Hu + forma + cor + SVM",
        "C": c, "f1_macro_val": round(melhor[0], 4),
        "n": {k: len(v) for k, v in splits.items()},
        "acuracia": round(accuracy_score(yte, pred), 4),
        "acuracia_top3": round(top_k_accuracy_score(yte, prob, k=3, labels=modelo.classes_), 4),
        "f1_macro": round(f1_score(yte, pred, average="macro", labels=rotulos, zero_division=0), 4),
        "tempo_extracao_s": round(t_extr, 1), "tempo_treino_s": round(t_treino, 2),
        "tempo_classificacao_ms_por_imagem_sem_extracao": round(t_pred, 3),
        "classes": dados.classes(modo),
        "matriz_confusao": confusion_matrix(yte, pred, labels=rotulos).tolist(),
    }
    (RAIZ / "models").mkdir(exist_ok=True)
    joblib.dump(modelo, RAIZ / "models" / f"classico_{modo}.joblib")
    saida = RAIZ / "docs" / "metricas"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / f"classico_{modo}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print({k: v for k, v in resultado.items() if k not in ("matriz_confusao", "classes")})


if __name__ == "__main__":
    main()

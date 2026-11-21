"""Página Métricas: resultados no conjunto de teste (docs/metricas/*.json)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.interface.comum import METODOS
from src.metrics import revisao
from src.metrics.dados import ESPECIES

PASTA = Path(__file__).resolve().parents[2] / "docs" / "metricas"


def _carregar() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PASTA.glob("*.json"))
            if not p.name.startswith("fusao")]


def pagina() -> None:
    st.title("Métricas")
    st.caption("Conjuntos de teste nunca usados no treino: TRUNK12 (casca, 59 fotos) e folhas do GBIF "
               "(518 fotos, separadas por observação).")
    res = _carregar()
    if not res:
        st.info("Nenhuma métrica em docs/metricas/. Rode os scripts de treino.")
        return
    nome = {"mobilenet_v3_large": METODOS["mobilenet_v3_large"], "resnet18": METODOS["resnet18"],
            "LBP + cor + SVM": "Clássico: LBP + cor + SVM", "Hu + forma + cor + SVM": "Clássico: forma + cor + SVM"}
    tabela = pd.DataFrame([{
        "Modo": r["modo"].capitalize(), "Método": nome.get(r["metodo"], r["metodo"]),
        "Acurácia": r["acuracia"], "Top-3": r["acuracia_top3"], "F1 macro": r["f1_macro"],
        "ms/imagem": r.get("tempo_ms_por_imagem_mediana", r.get("tempo_classificacao_ms_por_imagem_sem_extracao")),
        "Treino (s)": r.get("tempo_treino_s"),
    } for r in res]).sort_values(["Modo", "Acurácia"], ascending=[True, False])
    st.dataframe(tabela.style.format({"Acurácia": "{:.1%}", "Top-3": "{:.1%}", "F1 macro": "{:.3f}",
                                      "ms/imagem": "{:.1f}", "Treino (s)": "{:.1f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Tempo das CNNs: mediana por imagem na GPU, incluindo todos os patches/recortes. "
               "Tempo dos clássicos: só o SVM, sem a extração de características.")

    fusao = PASTA / "fusao.json"
    if fusao.exists():
        f = json.loads(fusao.read_text(encoding="utf-8"))
        st.subheader("Uso combinado casca + folha")
        st.dataframe(pd.DataFrame(f["tabela"]), hide_index=True)
        st.caption(f["nota"])

    st.subheader("Matriz de confusão")
    escolha = st.selectbox("Modelo", range(len(res)),
                           format_func=lambda i: f"{res[i]['modo']} · {nome.get(res[i]['metodo'], res[i]['metodo'])}")
    r = res[escolha]
    rot = [ESPECIES[c][0] for c in r["classes"]]
    mc = pd.DataFrame(r["matriz_confusao"], index=rot, columns=rot)
    st.dataframe(mc.style.background_gradient(cmap="Greens", axis=None), use_container_width=True)
    st.caption("Linhas: espécie real · colunas: espécie prevista.")

    c = revisao.concordancia()
    if c:
        st.metric("Concordância com profissionais", f"{c['concordancia']:.0%}", help=f"{c['revisoes']} revisões")

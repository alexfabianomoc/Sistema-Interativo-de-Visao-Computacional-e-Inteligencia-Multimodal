"""Sistema Interativo de Visão Computacional e Inteligência Multimodal — Alex Fabiano Silva (RA 031600).

Execução:  streamlit run app.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parent
(RAIZ / "logs").mkdir(exist_ok=True)
logging.basicConfig(filename=RAIZ / "logs" / "app.log", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Identificação de árvores", page_icon="🌳", layout="wide")

from src.interface import classificar, metricas, revisao  # noqa: E402

pg = st.navigation([
    st.Page(classificar.pagina, title="Classificar", icon="🔎", url_path="classificar", default=True),
    st.Page(revisao.pagina, title="Revisão", icon="🧑‍🔬", url_path="revisao"),
    st.Page(metricas.pagina, title="Métricas", icon="📊", url_path="metricas"),
])
pg.run()

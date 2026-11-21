"""Configuração, cache de modelos e barra lateral compartilhados pelas páginas."""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
import yaml

from src.models import servico

RAIZ = Path(__file__).resolve().parents[2]
log = logging.getLogger("app")

METODOS = {"mobilenet_v3_large": "MobileNetV3-Large (CNN)", "resnet18": "ResNet-18 (CNN)",
           "convnext_tiny": "ConvNeXt-Tiny (CNN)", "classico": "Clássico (LBP / forma + SVM)"}


def metodos_disponiveis(modo: str) -> list[str]:
    """Só oferece um método se houver pesos para o(s) modo(s) escolhido(s)."""
    modos = PARES.get(modo, [modo])
    return [m for m in METODOS if all(_tem_pesos(m, x) for x in modos)]


def _tem_pesos(metodo: str, modo: str) -> bool:
    try:
        servico.caminho_pesos(metodo, modo)
        return True
    except servico.PesosAusentes:
        return False


MODOS = {"tronco_br": "Tronco", "folha_br": "Folha", "tronco_folha_br": "Tronco + folha",
         "casca": "Tronco — exóticas (TRUNK12)", "folha": "Folha — exóticas europeias",
         "casca_folha": "Tronco + folha — exóticas"}
# Modo combinado -> (modo do tronco, modo da folha)
PARES = {"tronco_folha_br": ["tronco_br", "folha_br"], "casca_folha": ["casca", "folha"]}
# Modo brasileiro -> modo de reserva para árvores exóticas
RESERVA = {"tronco_br": "casca", "folha_br": "folha"}


def padrao() -> dict:
    with open(RAIZ / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def iniciar_estado() -> None:
    for k, v in padrao().items():
        st.session_state.setdefault(k, v)


def restaurar() -> None:
    for k, v in padrao().items():
        st.session_state[k] = v


@st.cache_resource(show_spinner="Carregando modelo...")
def modelo(metodo: str, modo: str):
    return servico.carregar(metodo, modo)


@st.cache_resource(show_spinner="Carregando PaliGemma 2 (cerca de 20 s na primeira vez)...")
def vlm():
    from src.models.vlm import PaliGemma
    return PaliGemma()


@st.cache_resource(show_spinner="Carregando tradutor pt ↔ en...")
def tradutor():
    from src.models.traducao import Tradutor
    return Tradutor()


def _metodo_do_modo() -> None:
    """Ao trocar de modo, seleciona o método padrão daquele modo (config.yaml: metodo_padrao)."""
    st.session_state["metodo"] = st.session_state["metodo_padrao"].get(st.session_state["modo"],
                                                                       st.session_state["metodo"])


def barra_lateral() -> dict:
    iniciar_estado()
    s = st.sidebar
    s.header("Parâmetros")
    s.radio("Modo", list(MODOS), format_func=MODOS.get, key="modo", on_change=_metodo_do_modo)
    opcoes = metodos_disponiveis(st.session_state["modo"]) or list(METODOS)
    if st.session_state["metodo"] not in opcoes:  # ex.: método sem pesos para o modo escolhido
        st.session_state["metodo"] = opcoes[0]
    s.selectbox("Método", opcoes, format_func=METODOS.get, key="metodo")
    s.slider("Limiar de confiança", 0.0, 1.0, step=0.05, key="limiar_confianca",
             help="Abaixo deste valor o resultado é 'inconclusivo' e pode ir para revisão manual.")
    s.slider("Top-k exibido", 1, 5, key="top_k")
    s.toggle("Consultar exóticas quando inconclusivo", key="reserva_exoticas",
             help="Se o modelo brasileiro ficar abaixo do limiar, roda também o modelo do TRUNK12 (tronco) "
                  "ou das folhas europeias e mostra o resultado como possível espécie exótica.")
    with s.expander("Pré-processamento", expanded=False):
        st.select_slider("Tamanho do patch (tronco)", [160, 192, 224, 288, 320], key="tamanho_patch",
                         help="Lado do patch na imagem reduzida; patches menores = grade mais fina.")
        st.slider("CLAHE no tronco", 0.0, 6.0, step=0.5, key="clahe_casca")
        st.slider("CLAHE na folha", 0.0, 6.0, step=0.5, key="clahe_folha")
    s.slider("Opacidade do mapa", 0.0, 1.0, step=0.05, key="opacidade_mapa")
    s.button("Restaurar padrão", on_click=restaurar, use_container_width=True)
    return {k: st.session_state[k] for k in padrao()}

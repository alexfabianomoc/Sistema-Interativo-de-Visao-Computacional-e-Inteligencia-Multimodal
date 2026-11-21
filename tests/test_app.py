"""Testes de ponta a ponta da interface com o AppTest do Streamlit (precisam dos pesos em models/)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
PRECISA_PESOS = pytest.mark.skipif(not (RAIZ / "models" / "mobilenet_v3_large_folha.pt").exists(),
                                   reason="pesos ausentes")


def _app(modo: str, metodo: str = "mobilenet_v3_large") -> AppTest:
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180)
    at.session_state["modo"] = modo
    at.session_state["metodo"] = metodo
    at.run()
    return at


def _usar_amostra(at: AppTest, chave: str, sub: str) -> None:
    at.radio(key=f"fonte_{chave}").set_value("Amostra").run()
    sel = at.selectbox(key=f"am_{chave}")
    escolha = next(o for o in sel.options if o.startswith(sub))
    sel.set_value(next(p for p in sel.options if p == escolha)).run()


def test_pagina_abre_sem_erro():
    at = _app("tronco_br")
    assert not at.exception
    assert "Identificação" in at.title[0].value


@PRECISA_PESOS
@pytest.mark.parametrize("metodo", ["mobilenet_v3_large", "classico"])
def test_classifica_folha_de_amostra(metodo):
    at = _app("folha", metodo)
    _usar_amostra(at, "folha", "folha")
    assert not at.exception, at.exception
    textos = " ".join(m.value for m in at.success) + " ".join(m.value for m in at.warning)
    assert "confiança" in textos or "Inconclusivo" in textos


@PRECISA_PESOS
def test_paginas_revisao_e_metricas():
    for pagina in ("src/interface/revisao.py", "src/interface/metricas.py"):
        at = AppTest.from_string(f"from src.interface import {Path(pagina).stem} as p\np.pagina()",
                                 default_timeout=60)
        at.run()
        assert not at.exception, at.exception


@pytest.mark.skipif(not (RAIZ / "models" / "convnext_tiny_tronco_br.pt").exists(), reason="pesos ausentes")
@pytest.mark.parametrize("modo,pasta", [("tronco_br", "tronco_br"), ("folha_br", "folha_br")])
def test_classifica_modos_brasileiros(modo, pasta):
    at = _app(modo, "convnext_tiny")
    _usar_amostra(at, modo, pasta)
    assert not at.exception, at.exception
    textos = " ".join(m.value for m in at.success) + " ".join(m.value for m in at.warning)
    assert "confiança" in textos or "Inconclusivo" in textos


def test_fusao_por_classe_usa_so_classes_comuns():
    import numpy as np
    from src.models import servico
    cls, p = servico.fusao_por_classe(np.array([0.7, 0.2, 0.1]), ["a", "b", "c"],
                                      np.array([0.1, 0.6, 0.2, 0.1]), ["b", "x", "a", "y"])
    assert cls == ["a", "b"] and abs(p.sum() - 1) < 1e-6 and p.argmax() == 0

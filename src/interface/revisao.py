"""Página Revisão: fila de casos inconclusivos para um profissional habilitado (CRBio ou CREA)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.metrics import revisao
from src.models import servico

OUTRA = "Outra espécie (digitar)"


def pagina() -> None:
    st.title("Revisão manual")
    st.caption("Casos inconclusivos ou enviados pelo usuário. O profissional confirma ou corrige a espécie; "
               "as respostas ficam em revisoes/revisoes.csv e viram a métrica de concordância.")
    fila = revisao.pendentes()
    if not fila:
        st.info("Nenhuma imagem aguardando revisão.")
    for item in fila:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 1, 1.3])
            c1.image(str(revisao.FILA / f"{item['id']}.jpg"), caption="Imagem")
            calor = revisao.FILA / f"{item['id']}_calor.jpg"
            if calor.exists():
                c2.image(str(calor), caption="Mapa de calor do modelo")
            with c3:
                st.markdown(f"**Modo:** {item['modo']} · **Método:** {item['metodo']}")
                st.markdown("**Top-3 do modelo:** " + ", ".join(
                    servico.NOMES[servico.CLASSES.index(c)] for c in item["top3"]))
                st.markdown(f"**Confiança:** {item['confianca']:.0%}")
                if "lat" in item:
                    st.markdown(f"**Local:** {item['lat']}, {item['lon']}")
                with st.form(f"f_{item['id']}"):
                    nome = st.text_input("Nome do profissional")
                    registro = st.text_input("Registro (CRBio ou CREA)", placeholder="CRBio 12345/04-D")
                    opcoes = servico.CLASSES + [OUTRA]
                    esc = st.selectbox("Espécie correta", opcoes, index=servico.CLASSES.index(item["previsto"]),
                                       format_func=lambda c: c if c == OUTRA else servico.NOMES[servico.CLASSES.index(c)])
                    outra = st.text_input("Se outra, qual (nome científico)")
                    obs = st.text_area("Observação", height=70)
                    if st.form_submit_button("Registrar revisão", type="primary"):
                        especie = outra.strip() if esc == OUTRA else esc
                        if not especie:
                            st.error("Digite o nome da outra espécie.")
                        else:
                            try:
                                revisao.registrar(item, especie, nome, registro, obs)
                                st.success("Revisão registrada.")
                                st.rerun()
                            except ValueError as e:
                                st.error(str(e))

    st.divider()
    st.subheader("Revisões registradas")
    c = revisao.concordancia()
    if c is None:
        st.caption("Ainda não há revisões.")
        return
    st.metric("Concordância modelo × profissional", f"{c['concordancia']:.0%}", help=f"{c['revisoes']} revisões")
    st.dataframe(pd.DataFrame(c["linhas"]), hide_index=True, use_container_width=True)

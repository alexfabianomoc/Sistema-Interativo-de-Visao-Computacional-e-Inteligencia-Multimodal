"""Página Classificar: entrada, ROI e rotação, classificação, camadas gráficas, fusão e VLM."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.acquisition import entrada
from src.interface import comum
from src.interface.comum import log
from src.metrics import revisao
from src.models import servico
from src.preprocessing import transforms as T
from src.visualization import desenho, gradcam

FONTES = ["Upload", "Amostra", "Câmera", "Foto de drone"]


# ------------------------------------------------------------------ entrada

def _obter_imagem(chave: str, rotulo: str) -> tuple[np.ndarray | None, dict | None]:
    """Mostra os controles de entrada e devolve (imagem RGB, coordenadas ou None)."""
    fonte = st.radio(f"Fonte da imagem {rotulo}", FONTES, horizontal=True, key=f"fonte_{chave}")
    try:
        if fonte == "Upload" or fonte == "Foto de drone":
            arq = st.file_uploader("Envie uma foto (JPG ou PNG)", key=f"up_{chave}_{fonte}",
                                   type=None)  # sem filtro: o próprio app explica formatos inválidos
            if arq is None:
                return None, None
            img, pil = entrada.abrir_upload(arq.getvalue(), arq.name)
            coords = _coordenadas_drone(pil, chave) if fonte == "Foto de drone" else None
            return img, coords
        if fonte == "Amostra":
            amostras = entrada.listar_amostras()
            if not amostras:
                st.info("Nenhuma amostra em data_samples/.")
                return None, None
            p = st.selectbox("Amostra", amostras, format_func=lambda p: str(p.relative_to(entrada.AMOSTRAS)),
                             key=f"am_{chave}")
            return T.ler_arquivo(p), None
        foto = st.camera_input("Tire a foto", key=f"cam_{chave}")
        st.caption("Se a câmera não aparecer, o navegador não encontrou uma câmera ou a permissão foi negada. "
                   "Use Upload ou Amostra.")
        if foto is None:
            return None, None
        return entrada.abrir_upload(foto.getvalue(), "camera.jpg")[0], None
    except T.ImagemInvalida as e:
        log.warning("entrada inválida: %s", e)
        st.error(str(e))
        return None, None


def _coordenadas_drone(pil, chave: str) -> dict | None:
    gps = entrada.gps_exif(pil)
    if gps:
        alt = f", altitude {gps['alt_m']} m" if "alt_m" in gps else ""
        st.success(f"GPS do EXIF: {gps['lat']}, {gps['lon']}{alt}")
        return gps
    st.warning("Esta foto não tem GPS no EXIF.")
    seguir = st.radio("Classificar sem coordenadas?", ["Sim", "Não, informar manualmente"],
                      horizontal=True, key=f"semgps_{chave}")
    if seguir == "Sim":
        return None
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", -90.0, 90.0, value=0.0, format="%.6f", key=f"lat_{chave}")
    lon = c2.number_input("Longitude", -180.0, 180.0, value=0.0, format="%.6f", key=f"lon_{chave}")
    erro = entrada.validar_coordenadas(lat, lon)
    if erro:
        st.error(erro)
        st.stop()
    return {"lat": lat, "lon": lon}


def _roi_e_rotacao(img: np.ndarray, chave: str, cfg: dict) -> np.ndarray:
    """Item 3 (rotação) e item 9 (ROI): controles + pré-visualização com o retângulo."""
    f = cfg["roi_fracao_padrao"]
    with st.expander("Recorte (região de interesse) e rotação", expanded=False):
        graus = st.slider("Rotação (graus)", -45, 45, 0, key=f"rot_{chave}",
                          help="Endireita troncos fotografados inclinados (comum em foto de drone).")
        c1, c2 = st.columns(2)
        x0, x1 = c1.slider("Faixa horizontal (%)", 0, 100, (round(50 - 50 * f), round(50 + 50 * f)), key=f"rx_{chave}")
        y0, y1 = c2.slider("Faixa vertical (%)", 0, 100, (round(50 - 50 * f), round(50 + 50 * f)), key=f"ry_{chave}")
        rot = T.rotacionar(img, graus)
        h, w = rot.shape[:2]
        prev = rot.copy()
        cv2.rectangle(prev, (int(x0 / 100 * w), int(y0 / 100 * h)), (int(x1 / 100 * w), int(y1 / 100 * h)),
                      (255, 220, 0), max(3, w // 200))
        st.image(prev, caption="Região que será analisada", width=360)
    return servico.preparar(img, (x0 / 100, y0 / 100, (x1 - x0) / 100, (y1 - y0) / 100), graus)


# ------------------------------------------------------------------ análise

def _analisar(img: np.ndarray, modo: str, cfg: dict) -> dict:
    metodo = cfg["metodo"]
    clahe = cfg["clahe_casca"] if modo == "casca" else cfg["clahe_folha"]
    m = comum.modelo(metodo, modo)
    lado = cfg["tamanho_patch"] if modo == "casca" else 224
    r = servico.classificar(m, metodo, modo, img, lado, clahe)
    r.update(modo=modo, metodo=metodo, img=img, clahe=clahe, lado=lado, modelo=m,
             realcada=T.clahe_l(img, clahe))
    if modo == "folha":
        r["mascara"], r["contorno"] = T.segmentar_folha(img)
    log.info("classificado modo=%s metodo=%s top=%s conf=%.3f ms=%.0f", modo, metodo,
             servico.CLASSES[int(r["prob"].argmax())], r["prob"].max(), r["ms"])
    return r


def _camadas(r: dict, cfg: dict, classe: int, chave: str) -> np.ndarray:
    """Item 4: cada camada liga e desliga; devolve a imagem final composta."""
    eh_cnn = r["metodo"] != "classico"
    c1, c2, c3 = st.columns(3)
    usar_calor = c1.toggle("Mapa de calor (Grad-CAM)", value=eh_cnn, disabled=not eh_cnn, key=f"lc_{chave}")
    usar_grade = c2.toggle("Grade de patches", value=eh_cnn and r["modo"] == "casca",
                           disabled=not (eh_cnn and r["modo"] == "casca"), key=f"lg_{chave}")
    usar_cont = c3.toggle("Contorno da folha", value=r["modo"] == "folha",
                          disabled=r["modo"] != "folha", key=f"lf_{chave}")
    final = r["img"].copy()
    if usar_calor:
        k = (chave, classe, r["lado"], r["clahe"], r["metodo"])
        if st.session_state.get(f"cam_key_{chave}") != k:
            with st.spinner("Calculando Grad-CAM..."):
                st.session_state[f"cam_{chave}"] = gradcam.mapa_completo(r["modelo"], r["img"], r, classe,
                                                                         r["lado"], r["clahe"])
            st.session_state[f"cam_key_{chave}"] = k
        final = desenho.mapa_de_calor(final, st.session_state[f"cam_{chave}"], cfg["opacidade_mapa"])
    if usar_grade:
        final = desenho.grade_patches(final, r["posicoes"], r["prob_partes"])
    if usar_cont:
        if r["contorno"] is None:
            st.caption("Nenhuma folha segmentada por cor nesta imagem.")
        final = desenho.contorno_folha(final, r["mascara"], r["contorno"])
    return final


def _resultado(r: dict, cfg: dict, chave: str, coords: dict | None) -> None:
    prob, k = r["prob"], cfg["top_k"]
    tops = servico.top_k(prob, k)
    melhor, conf = tops[0]
    inconclusivo = conf < cfg["limiar_confianca"]
    if inconclusivo:
        st.warning(f"**Inconclusivo**: a maior confiança ({conf:.0%}) está abaixo do limiar "
                   f"({cfg['limiar_confianca']:.0%}). Sugestão mais provável: {servico.NOMES[melhor]}.")
    else:
        st.success(f"**{servico.NOMES[melhor]}** com {conf:.0%} de confiança")
    st.caption(f"Tempo de classificação: {r['ms']:.0f} ms · método {comum.METODOS[r['metodo']]}")

    for i, p in tops:  # barras top-k
        st.progress(float(p), text=f"{servico.NOMES[i]}: {p:.1%}")

    classe = st.selectbox("Explicar a espécie", [i for i, _ in tops], format_func=lambda i: servico.NOMES[i],
                          key=f"exp_{chave}", help="O mapa de calor mostra a região que levou a esta espécie.")
    final = _camadas(r, cfg, classe, chave)
    indices = sorted({i for i, _ in tops} | ({int(p.argmax()) for p in r["prob_partes"]}
                                             if r["prob_partes"] is not None else set()))
    st.image(final, caption="Resultado (camadas ativas)", width=720)
    st.image(desenho.legenda(servico.NOMES, indices), caption="Legenda de cores por espécie", width=320)

    with st.expander("Vista comparativa: original, após CLAHE e resultado"):
        st.image(desenho.lado_a_lado([r["img"], r["realcada"], final],
                                     ["Original (recorte)", f"CLAHE {r['clahe']}", "Resultado"]))

    st.download_button("Baixar imagem final (PNG)",
                       desenho.png(desenho.compor_exportacao(final, servico.NOMES, indices)),
                       file_name=f"resultado_{r['modo']}.png", mime="image/png", key=f"dl_{chave}")

    if inconclusivo or st.toggle("Enviar mesmo assim para revisão manual", key=f"forcar_{chave}"):
        if st.button("Enviar para revisão por profissional", key=f"rev_{chave}", type="primary"):
            calor = st.session_state.get(f"cam_{chave}")
            calor_img = desenho.mapa_de_calor(r["img"], calor, 0.45) if calor is not None else None
            ident = revisao.enviar_para_fila(r["img"], calor_img, {
                "modo": r["modo"], "metodo": r["metodo"], "previsto": servico.CLASSES[melhor],
                "confianca": round(conf, 4), "top3": [servico.CLASSES[i] for i, _ in servico.top_k(prob, 3)],
                **(coords or {})})
            st.success(f"Enviado para a fila de revisão ({ident}). Veja a página Revisão.")


# ------------------------------------------------------------------ VLM

def _painel_vlm(img: np.ndarray, modo: str, chave: str) -> None:
    st.subheader("Pergunte sobre a imagem")
    if not st.toggle("Ativar o PaliGemma 2 (local; cerca de 20 s para carregar na primeira vez)",
                     key=f"vlm_on_{chave}"):
        return
    with st.container(border=True):
        st.caption("Interpretação de um modelo de linguagem visual: pode errar e não substitui a "
                   "classificação nem a revisão. Só o recorte é enviado, sem EXIF nem GPS.")
        traduzir = st.toggle("Traduzir pergunta e resposta (pt ↔ en)", value=True, key=f"tr_{chave}")
        try:
            vlm = comum.vlm()
            tr = comum.tradutor() if traduzir else None
        except Exception as e:  # modelo não baixado, sem memória, bibliotecas ausentes
            log.warning("VLM indisponível: %s", e)
            st.info(f"Painel desativado: {e}")
            return
        assinatura = (img.shape, float(img[::16, ::16].mean()))  # nova imagem = histórico limpo
        if st.session_state.get(f"vlm_img_{chave}") != assinatura:
            st.session_state[f"vlm_img_{chave}"] = assinatura
            st.session_state[f"vlm_hist_{chave}"] = []
        hist = st.session_state[f"vlm_hist_{chave}"]
        c1, c2 = st.columns(2)
        if c1.button("Descrever a imagem", key=f"vd_{chave}"):
            hist.append(_registro_vlm("Descrição", vlm.descrever(img), tr))
        if c2.button("Verificar a entrada", key=f"vv_{chave}"):
            ok, resp = vlm.verificar(img, modo)
            alvo = "tronco ou casca" if modo == "casca" else "folha ou planta"
            aviso = (f"O VLM não viu {alvo} nesta imagem. Confira antes de confiar na classificação."
                     if ok is False else f"O VLM confirma: há {alvo} na imagem." if ok else
                     "Resposta ambígua do VLM.")
            hist.append({"tipo": "Verificação", "texto": aviso, "original": resp.texto, "ms": resp.ms,
                         "alerta": ok is False})
        pergunta = st.text_input("Pergunta livre (uma por vez, curta e direta)", key=f"vq_{chave}")
        if st.button("Perguntar", key=f"vp_{chave}") and pergunta.strip():
            q = tr.para_ingles(pergunta) if tr else pergunta
            hist.append(_registro_vlm(f"Pergunta: {pergunta}", vlm.pergunta_livre(img, q), tr, q))
        for h in reversed(hist[-6:]):
            (st.warning if h.get("alerta") else st.info)(f"**{h['tipo']}** — {h['texto']}")
            extra = f" · pergunta em inglês: “{h['q_en']}”" if h.get("q_en") else ""
            st.caption(f"Original: “{h['original']}” · {h['ms']:.0f} ms{extra}")


def _registro_vlm(tipo: str, resp, tr, q_en: str | None = None) -> dict:
    log.info("vlm prompt=%r resposta=%r ms=%.0f", resp.prompt, resp.texto, resp.ms)
    texto = tr.para_portugues(resp.texto) if tr else resp.texto
    return {"tipo": tipo, "texto": texto, "original": resp.texto, "ms": resp.ms,
            "q_en": q_en if tr else None}


# ------------------------------------------------------------------ página

def _bloco(modo: str, cfg: dict, rotulo: str = "") -> tuple[dict | None, dict | None]:
    img, coords = _obter_imagem(modo, rotulo)
    if img is None:
        return None, None
    try:
        recorte = _roi_e_rotacao(img, modo, cfg)
    except T.ImagemInvalida as e:
        st.error(str(e))
        return None, None
    try:
        return _analisar(recorte, modo, cfg), coords
    except servico.PesosAusentes as e:
        st.error(str(e))
        return None, None
    except Exception as e:  # nunca fechar a aplicação
        log.exception("falha na análise")
        st.error(f"Erro inesperado na análise: {e}. Detalhes em logs/app.log.")
        return None, None


def pagina() -> None:
    cfg = comum.barra_lateral()
    st.title("Identificação de espécies de árvores")
    st.caption("Sugestão com grau de confiança, não identificação botânica definitiva. "
               "12 espécies europeias (TRUNK12).")
    modo = cfg["modo"]

    if modo in ("casca", "folha"):
        r, coords = _bloco(modo, cfg)
        if r:
            _resultado(r, cfg, modo, coords)
            _painel_vlm(r["img"], modo, modo)
        return

    st.info("Envie uma foto da casca e uma da folha da mesma árvore. A aplicação combina as duas "
            "classificações multiplicando as probabilidades (fusão tardia).")
    col_c, col_f = st.columns(2)
    with col_c:
        st.subheader("Casca")
        rc, coords = _bloco("casca", cfg, "da casca")
    with col_f:
        st.subheader("Folha")
        rf, _ = _bloco("folha", cfg, "da folha")
    if not (rc and rf):
        return
    prob = servico.fusao(rc["prob"], rf["prob"])
    st.subheader("Resultado conjunto")
    tabela = pd.DataFrame({
        "Espécie": [servico.NOMES[i] for i, _ in servico.top_k(prob, cfg["top_k"])],
        "Casca": [rc["prob"][i] for i, _ in servico.top_k(prob, cfg["top_k"])],
        "Folha": [rf["prob"][i] for i, _ in servico.top_k(prob, cfg["top_k"])],
        "Conjunto": [p for _, p in servico.top_k(prob, cfg["top_k"])],
    })
    st.dataframe(tabela.style.format({c: "{:.0%}" for c in ("Casca", "Folha", "Conjunto")}),
                 hide_index=True, use_container_width=True)
    melhor, conf = servico.top_k(prob, 1)[0]
    (st.warning if conf < cfg["limiar_confianca"] else st.success)(
        f"Conjunto: **{servico.NOMES[melhor]}** com {conf:.0%}"
        + (" (inconclusivo)" if conf < cfg["limiar_confianca"] else ""))
    if servico.CLASSES[int(rc["prob"].argmax())] != servico.CLASSES[int(rf["prob"].argmax())]:
        st.caption("Casca e folha discordam individualmente; confira se as fotos são da mesma árvore.")
    t1, t2 = st.tabs(["Detalhes da casca", "Detalhes da folha"])
    with t1:
        _resultado(rc, cfg, "casca", coords)
    with t2:
        _resultado(rf, cfg, "folha", None)

import cv2
import numpy as np
import pytest

from src.preprocessing import transforms as T


def _img(h=300, w=400):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_arquivo_que_nao_e_imagem():
    with pytest.raises(T.ImagemInvalida, match="JPG ou PNG"):
        T.decodificar(b"isto nao e uma imagem")


def test_imagem_pequena():
    ok, buf = cv2.imencode(".png", _img(100, 150))
    with pytest.raises(T.ImagemInvalida, match="mínimo"):
        T.decodificar(buf.tobytes())


def test_letterbox_mapeia_coordenadas_de_volta():
    img = _img(300, 600)
    out, lb = T.letterbox(img, 224)
    assert out.shape == (224, 224, 3)
    x, y = lb.para_letterbox(600, 300)
    assert lb.para_original(x, y) == pytest.approx((600, 300))
    assert lb.dx == 0 and lb.dy > 0  # imagem larga: bordas em cima e embaixo


def test_recorte_minusculo():
    with pytest.raises(T.ImagemInvalida, match="Refaça"):
        T.recortar(_img(), (10, 10, 5, 5))


def test_rotacao_nao_corta_cantos():
    out = T.rotacionar(_img(300, 400), 90)
    assert out.shape[:2] == (400, 300)


def test_clahe_preserva_formato_e_pode_desligar():
    img = _img()
    assert T.clahe_l(img, 3.0).shape == img.shape
    assert T.clahe_l(img, 0) is img


def test_normalizar():
    x = T.normalizar(_img(224, 224))
    assert x.shape == (3, 224, 224) and x.dtype == np.float32


def test_segmentar_folha_verde_em_fundo_cinza():
    img = np.full((300, 300, 3), 128, np.uint8)
    cv2.ellipse(img, (150, 150), (90, 50), 30, 0, 360, (40, 160, 40), -1)
    mascara, contorno = T.segmentar_folha(img)
    assert contorno is not None
    assert 0.1 < (mascara > 0).mean() < 0.3


def test_segmentar_sem_folha():
    _, contorno = T.segmentar_folha(np.full((300, 300, 3), 128, np.uint8))
    assert contorno is None


def test_patches_cobrem_a_imagem():
    ps = T.patches(_img(500, 700), 224)
    assert all(p.shape == (224, 224, 3) for _, _, p in ps)
    assert max(x for x, _, _ in ps) == 700 - 224

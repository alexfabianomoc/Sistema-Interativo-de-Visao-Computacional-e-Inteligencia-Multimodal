"""Aquisição: amostras, EXIF (GPS e altitude de drone), validação de coordenadas."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import ExifTags, Image, ImageOps
import numpy as np

from src.preprocessing.transforms import ImagemInvalida, validar_tamanho

RAIZ = Path(__file__).resolve().parents[2]
AMOSTRAS = RAIZ / "data_samples"
FORMATOS = {".jpg", ".jpeg", ".png"}


def abrir_upload(dados: bytes, nome: str = "") -> tuple[np.ndarray, Image.Image]:
    """Abre um upload respeitando a orientação do EXIF. Devolve (RGB, imagem PIL original)."""
    ext = Path(nome).suffix.lower()
    if ext and ext not in FORMATOS:
        raise ImagemInvalida(f"Formato {ext} não suportado. Envie JPG ou PNG.")
    try:
        pil = Image.open(io.BytesIO(dados))
        pil.load()
    except Exception as e:
        raise ImagemInvalida("Não foi possível abrir este arquivo. Envie um JPG ou PNG.") from e
    img = np.array(ImageOps.exif_transpose(pil).convert("RGB"))
    validar_tamanho(img)
    return img, pil


def listar_amostras() -> list[Path]:
    return sorted(p for p in AMOSTRAS.rglob("*") if p.suffix.lower() in FORMATOS)


def _graus(valor, ref) -> float:
    g, m, s = (float(v) for v in valor)
    dec = g + m / 60 + s / 3600
    return -dec if ref in ("S", "W") else dec


def gps_exif(pil: Image.Image) -> dict | None:
    """Latitude, longitude e altitude do EXIF; None se a foto não tiver GPS."""
    try:
        gps = pil.getexif().get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None
    out = {"lat": round(_graus(gps[2], gps.get(1, "N")), 6), "lon": round(_graus(gps[4], gps.get(3, "E")), 6)}
    if 6 in gps:
        alt = float(gps[6])
        out["alt_m"] = round(-alt if gps.get(5) == 1 else alt, 1)
    return out


def validar_coordenadas(lat: float, lon: float) -> str | None:
    """Mensagem de erro ou None se válidas."""
    if not -90 <= lat <= 90:
        return "Latitude deve estar entre -90 e 90."
    if not -180 <= lon <= 180:
        return "Longitude deve estar entre -180 e 180."
    if lat == 0 and lon == 0:
        return "Coordenadas (0, 0) parecem não preenchidas."
    return None

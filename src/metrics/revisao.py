"""Fila de revisão manual por profissional habilitado e registro em revisoes/revisoes.csv."""
from __future__ import annotations

import csv
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
PASTA = RAIZ / "revisoes"
FILA = PASTA / "fila"
CSV = PASTA / "revisoes.csv"
CAMPOS = ["data_hora", "id", "modo", "previsto", "confianca", "top3", "especie_profissional",
          "concorda", "profissional", "registro", "observacao", "lat", "lon"]
REGISTRO_RE = re.compile(r"^(CRBio|CREA)[\s-]*[A-Z]{0,2}[\s-]*\d{3,8}[\s/-]*[A-Z]{0,2}$", re.IGNORECASE)


def enviar_para_fila(img: np.ndarray, calor: np.ndarray | None, info: dict) -> str:
    """Guarda imagem, mapa de calor e top-3 para o profissional revisar."""
    FILA.mkdir(parents=True, exist_ok=True)
    ident = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(str(FILA / f"{ident}.jpg"))
    if calor is not None:
        cv2.imencode(".jpg", cv2.cvtColor(calor, cv2.COLOR_RGB2BGR))[1].tofile(str(FILA / f"{ident}_calor.jpg"))
    (FILA / f"{ident}.json").write_text(json.dumps({"id": ident, **info}, ensure_ascii=False), encoding="utf-8")
    return ident


def pendentes() -> list[dict]:
    if not FILA.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(FILA.glob("*.json"))]


def validar_registro(registro: str) -> bool:
    return bool(REGISTRO_RE.match(registro.strip()))


def registrar(item: dict, especie: str, profissional: str, registro: str, observacao: str) -> None:
    if not profissional.strip():
        raise ValueError("Informe o nome do profissional.")
    if not validar_registro(registro):
        raise ValueError("Registro inválido. Use, por exemplo, CRBio 12345/04-D ou CREA-SP 1234567.")
    novo = not CSV.exists()
    PASTA.mkdir(exist_ok=True)
    with open(CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        w.writerow({"data_hora": datetime.now().isoformat(timespec="seconds"), "id": item["id"],
                    "modo": item["modo"], "previsto": item["previsto"], "confianca": item["confianca"],
                    "top3": "; ".join(item["top3"]), "especie_profissional": especie,
                    "concorda": especie == item["previsto"], "profissional": profissional.strip(),
                    "registro": registro.strip(), "observacao": observacao.strip(),
                    "lat": item.get("lat", ""), "lon": item.get("lon", "")})
    for p in FILA.glob(f"{item['id']}*"):
        p.unlink()


def concordancia() -> dict | None:
    if not CSV.exists():
        return None
    with open(CSV, encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh))
    if not linhas:
        return None
    acordo = sum(l["concorda"] == "True" for l in linhas)
    return {"revisoes": len(linhas), "concordancia": acordo / len(linhas), "linhas": linhas}

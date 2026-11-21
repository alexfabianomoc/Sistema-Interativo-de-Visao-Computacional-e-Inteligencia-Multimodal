"""Base brasileira (opção A, 21/11): fotos do iNaturalist no Brasil, separadas em tronco e folha pelo PaliGemma.

- Tronco: pau-brasil, ipês (gênero Handroanthus) e aroeira-pimenteira, com TODAS as observações.
- Folha: as 10 espécies com mais folhas úteis no teste de rendimento, até 300 observações cada
  (as demais candidatas são baixadas também, para a escolha ser feita pela contagem real).
Divisão por observação: 70% treino, 15% validação, 15% teste (semente fixa).

    python scripts/baixar_inat_brasil.py baixar
    python scripts/baixar_inat_brasil.py classificar
    python scripts/baixar_inat_brasil.py dividir
Saída: data/inat_br/<taxon>/*.jpg e data/inat_br/manifesto.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from teste_rendimento_inat import API, BRASIL, CANDIDATAS, LICENCAS, PROMPT, _baixar, _get, _parte  # noqa: E402

SAIDA = RAIZ / "data" / "inat_br"
TRONCO = ["Paubrasilia echinata", "Handroanthus", "Schinus terebinthifolia"]
MAX_OBS_FOLHA = 300
SEMENTE = 42


def baixar() -> None:
    linhas, tarefas = [], []
    for nome, popular in CANDIDATAS.items():
        t = _get(f"{API}/taxa?" + urllib.parse.urlencode({"q": nome, "is_active": "true", "per_page": 5}))
        taxon = next(x for x in t["results"] if x["name"] == nome)
        limite = 10**6 if nome in TRONCO else MAX_OBS_FOLHA
        obs, id_acima = [], 0
        while len(obs) < limite:
            q = {"taxon_id": taxon["id"], "place_id": BRASIL, "quality_grade": "research", "photos": "true",
                 "photo_license": LICENCAS, "per_page": 200, "order_by": "id", "order": "asc",
                 "id_above": id_acima}
            r = _get(f"{API}/observations?" + urllib.parse.urlencode(q))
            if not r["results"]:
                break
            obs += r["results"]
            id_acima = r["results"][-1]["id"]
            time.sleep(1.1)  # limite de ~60 requisições/minuto
        if len(obs) > limite:  # amostra reprodutível quando há mais observações que o limite
            obs = random.Random(SEMENTE).sample(obs, limite)
        pasta = SAIDA / nome.replace(" ", "_")
        pasta.mkdir(parents=True, exist_ok=True)
        for o in obs:
            for f in o.get("photos", []):
                if not f.get("license_code"):
                    continue
                destino = pasta / f"{o['id']}_{f['id']}.jpg"
                tarefas.append((f["url"].replace("square", "medium"), destino))
                linhas.append({"taxon": nome, "popular": popular, "obs": o["id"], "foto": f["id"],
                               "caminho": destino.relative_to(RAIZ / "data").as_posix(),
                               "licenca": f["license_code"], "autor": (f.get("attribution") or "").replace("\n", " "),
                               "fonte": f"https://www.inaturalist.org/observations/{o['id']}"})
        print(f"{popular:20s} {len(obs):5d} observações", flush=True)
    for tentativa in range(3):  # o S3 do iNaturalist às vezes fecha a conexão: repete as que falharam
        with ThreadPoolExecutor(6) as ex:
            ok = list(ex.map(_baixar, tarefas))
        if all(ok):
            break
    linhas = [l for l, (_, d) in zip(linhas, tarefas) if d.exists()]
    _gravar(linhas)
    print(f"{len(linhas)} fotos ({len(tarefas) - len(linhas)} falhas)")


def _ler() -> list[dict]:
    with open(SAIDA / "manifesto.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _gravar(linhas: list[dict]) -> None:
    campos = list(dict.fromkeys(k for l in linhas for k in l))
    with open(SAIDA / "manifesto.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


def classificar() -> None:
    sys.path.insert(0, str(RAIZ))
    from src.models.vlm import PaliGemma
    from src.preprocessing.transforms import ImagemInvalida, ler_arquivo
    vlm = PaliGemma()
    linhas = _ler()
    for k, l in enumerate(linhas):
        if l.get("parte"):
            continue
        try:
            resp = vlm.perguntar(ler_arquivo(RAIZ / "data" / l["caminho"]), PROMPT, 6)
            l["resposta_vlm"], l["parte"] = resp.texto, _parte(resp.texto)
        except ImagemInvalida:
            l["resposta_vlm"], l["parte"] = "", "invalida"
        if k % 500 == 0:
            print(k, "de", len(linhas), flush=True)
            _gravar(linhas)  # salva o progresso
    _gravar(linhas)
    cont = defaultdict(Counter)
    for l in linhas:
        cont[l["taxon"]][l["parte"]] += 1
    for nome in CANDIDATAS:
        print(f"{CANDIDATAS[nome]:20s} {dict(cont[nome])}")


def dividir() -> None:
    linhas = _ler()
    rng = random.Random(SEMENTE)
    por_taxon = defaultdict(list)
    for l in linhas:
        por_taxon[l["taxon"]].append(l["obs"])
    split_obs = {}
    for nome, obs in por_taxon.items():
        obs = sorted(set(obs))
        rng.shuffle(obs)
        n = len(obs)
        for i, o in enumerate(obs):
            split_obs[o] = "test" if i < round(0.15 * n) else "val" if i < round(0.30 * n) else "train"
    for l in linhas:
        l["split"] = split_obs[l["obs"]]
    _gravar(linhas)
    folhas = Counter(l["taxon"] for l in linhas if l["parte"] == "folha")
    escolhidas = [n for n, _ in folhas.most_common(10)]
    (SAIDA / "classes.json").write_text(json.dumps({"tronco": TRONCO, "folha": escolhidas,
                                                     "folhas_por_taxon": folhas}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print("folha (10 com mais fotos de folha):", [(CANDIDATAS[n], folhas[n]) for n in escolhidas])
    print("tronco:", [(CANDIDATAS[n], sum(1 for l in linhas if l["taxon"] == n and l["parte"] == "tronco"))
                      for n in TRONCO])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["baixar", "classificar", "dividir"])
    {"baixar": baixar, "classificar": classificar, "dividir": dividir}[ap.parse_args().etapa]()

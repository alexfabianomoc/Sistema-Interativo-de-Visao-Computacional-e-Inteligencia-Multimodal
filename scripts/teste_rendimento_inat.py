"""Teste de rendimento (21/11): quantas fotos de TRONCO e de FOLHA cada espécie brasileira candidata rende
no iNaturalist. Baixa até N observações por táxon (Brasil, grau de pesquisa, foto com licença CC) e
classifica cada foto com o PaliGemma em tronco, folha ou outras.

    python scripts/teste_rendimento_inat.py baixar --por-especie 50
    python scripts/teste_rendimento_inat.py classificar
Saída: data/inat_teste/<taxon>/*.jpg, manifesto.csv e docs/teste_rendimento_inat.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "data" / "inat_teste"
API = "https://api.inaturalist.org/v1"
BRASIL = 6878
LICENCAS = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa,cc-by-nd,cc-by-nc-nd"
PROMPT = "answer en Is this a photo of a tree trunk, a leaf, a flower, a fruit, or a whole tree?"

# Nome aceito -> nome popular (tabela "Bases de imagens de árvores brasileiras", 21/11)
CANDIDATAS = {
    "Schinus terebinthifolia": "aroeira-pimenteira", "Handroanthus": "ipês (gênero)",
    "Erythrina speciosa": "mulungu", "Spathodea campanulata": "espatódea",
    "Paubrasilia echinata": "pau-brasil", "Mangifera indica": "mangueira", "Delonix regia": "flamboyant",
    "Terminalia catappa": "chapéu-de-sol", "Pleroma granulosum": "quaresmeira", "Pachira aquatica": "munguba",
    "Cenostigma pluviosum": "sibipiruna", "Bauhinia variegata": "pata-de-vaca",
    "Lagerstroemia indica": "resedá", "Syzygium cumini": "jambolão", "Peltophorum dubium": "canafístula",
    "Moquilea tomentosa": "oiti", "Tipuana tipu": "tipuana", "Libidibia ferrea": "pau-ferro",
}


def _get(url: str) -> dict:
    for i in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if i == 3:
                raise
            time.sleep(2 ** i)
    raise RuntimeError


def _baixar(item) -> bool:
    url, destino = item
    if destino.exists():
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            destino.write_bytes(r.read())
        return True
    except Exception as e:
        print("falhou", url, e, file=sys.stderr)
        return False


def baixar(n: int) -> None:
    linhas, tarefas, totais = [], [], {}
    for nome, popular in CANDIDATAS.items():
        t = _get(f"{API}/taxa?" + urllib.parse.urlencode({"q": nome, "is_active": "true", "per_page": 5}))
        taxon = next(x for x in t["results"] if x["name"] == nome)
        q = {"taxon_id": taxon["id"], "place_id": BRASIL, "quality_grade": "research", "photos": "true",
             "photo_license": LICENCAS, "per_page": n, "order_by": "votes"}
        r = _get(f"{API}/observations?" + urllib.parse.urlencode(q))
        totais[nome] = r["total_results"]
        pasta = SAIDA / nome.replace(" ", "_")
        pasta.mkdir(parents=True, exist_ok=True)
        for obs in r["results"][:n]:
            for f in obs.get("photos", []):
                lic = f.get("license_code")
                if not lic:
                    continue
                url = f["url"].replace("square", "medium")
                destino = pasta / f"{obs['id']}_{f['id']}.jpg"
                tarefas.append((url, destino))
                linhas.append({"taxon": nome, "popular": popular, "obs": obs["id"], "foto": f["id"],
                               "caminho": destino.relative_to(RAIZ / "data").as_posix(), "licenca": lic,
                               "autor": (f.get("attribution") or "").replace("\n", " "),
                               "fonte": f"https://www.inaturalist.org/observations/{obs['id']}"})
        print(f"{popular:20s} {nome:26s} taxon {taxon['id']:>7}  obs no Brasil: {r['total_results']:5d}", flush=True)
        time.sleep(1.1)  # respeita o limite de ~60 requisições/minuto da API
    with ThreadPoolExecutor(6) as ex:
        ok = list(ex.map(_baixar, tarefas))
    linhas = [l for l, o in zip(linhas, ok) if o]
    with open(SAIDA / "manifesto.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    (SAIDA / "totais.json").write_text(json.dumps(totais, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(linhas)} fotos baixadas ({ok.count(False)} falhas)")


def _parte(resposta: str) -> str:
    r = resposta.lower()
    if "trunk" in r or "bark" in r:
        return "tronco"
    if "leaf" in r or "leaves" in r:
        return "folha"
    return "outras"


def classificar() -> None:
    sys.path.insert(0, str(RAIZ))
    from src.models.vlm import PaliGemma
    from src.preprocessing.transforms import ler_arquivo
    vlm = PaliGemma()
    with open(SAIDA / "manifesto.csv", encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh))
    for k, l in enumerate(linhas):
        resp = vlm.perguntar(ler_arquivo(RAIZ / "data" / l["caminho"]), PROMPT, 6)
        l["resposta_vlm"], l["parte"] = resp.texto, _parte(resp.texto)
        if k % 200 == 0:
            print(k, "de", len(linhas), flush=True)
    with open(SAIDA / "manifesto.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    totais = json.loads((SAIDA / "totais.json").read_text(encoding="utf-8"))
    por = defaultdict(Counter)
    obs = defaultdict(set)
    for l in linhas:
        por[l["taxon"]][l["parte"]] += 1
        obs[l["taxon"]].add(l["obs"])
    resultado = []
    for nome, popular in CANDIDATAS.items():
        c, n_obs = por[nome], len(obs[nome]) or 1
        alvo = min(totais.get(nome, 0), 300)  # download completo: até 300 observações
        resultado.append({"popular": popular, "taxon": nome, "obs_brasil": totais.get(nome, 0),
                          "obs_amostra": len(obs[nome]), "fotos": sum(c.values()), "tronco": c["tronco"],
                          "folha": c["folha"], "outras": c["outras"],
                          "tronco_estimado_completo": round(c["tronco"] / n_obs * alvo),
                          "folha_estimada_completo": round(c["folha"] / n_obs * alvo)})
    resultado.sort(key=lambda x: -x["tronco_estimado_completo"])
    (RAIZ / "docs" / "teste_rendimento_inat.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1),
                                                              encoding="utf-8")
    for r in resultado:
        print(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["baixar", "classificar"])
    ap.add_argument("--por-especie", type=int, default=50)
    a = ap.parse_args()
    baixar(a.por_especie) if a.etapa == "baixar" else classificar()


if __name__ == "__main__":
    main()

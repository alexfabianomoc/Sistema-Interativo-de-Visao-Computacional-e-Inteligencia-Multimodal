"""Listas de imagens com rótulo e split para os dois modos."""
from __future__ import annotations

import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
SEMENTE = 42

# Pasta do TRUNK12 -> (nome popular em português, nome científico EXIBIDO). A mesma ordem define o índice
# da classe. Nomes revisados em 21/11/2026 (docs/auditoria_taxonomica.md): o TRUNK12 só traz nomes
# populares em inglês, por isso amieiro, bétula, tília, carvalho, plátano e pinheiro ficam no nível
# de gênero ("sp."), por decisão do autor em 21/11.
ESPECIES = {
    "alder": ("Amieiro", "Alnus sp."),
    "beech": ("Faia", "Fagus sylvatica"),
    "birch": ("Bétula", "Betula sp."),
    "chestnut": ("Castanheiro", "Castanea sativa"),
    "ginkgo biloba": ("Ginkgo", "Ginkgo biloba"),
    "hornbeam": ("Carpino", "Carpinus betulus"),
    "horse chestnut": ("Castanheiro-da-índia", "Aesculus hippocastanum"),
    "linden": ("Tília", "Tilia sp."),
    "oak": ("Carvalho", "Quercus sp."),
    "oriental plane": ("Plátano", "Platanus sp."),
    "pine": ("Pinheiro", "Pinus sp."),
    "spruce": ("Abeto-da-noruega", "Picea abies"),
}
CLASSES = list(ESPECIES)

# Espécie usada na BUSCA do GBIF para baixar as fotos de folha e de casca de campo (não muda os
# rótulos exibidos). Nas seis classes exibidas como gênero, as fotos do GBIF são de UMA espécie só, a da
# busca, escolhida em 20/11 sem confirmação de que seja a espécie do TRUNK12.
TAXON_GBIF = {
    "alder": "Alnus glutinosa", "beech": "Fagus sylvatica", "birch": "Betula pendula",
    "chestnut": "Castanea sativa", "ginkgo biloba": "Ginkgo biloba", "hornbeam": "Carpinus betulus",
    "horse chestnut": "Aesculus hippocastanum", "linden": "Tilia cordata", "oak": "Quercus robur",
    "oriental plane": "Platanus orientalis", "pine": "Pinus sylvestris", "spruce": "Picea abies",
}


def trunk12(val: float = 0.15, teste: float = 0.15) -> dict[str, list[tuple[Path, int]]]:
    """Divisão estratificada e reprodutível do TRUNK12 (cerca de 33 fotos por espécie)."""
    itens = [(p, CLASSES.index(p.parent.name))
             for p in sorted((DATA / "trunk12").glob("*/*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not itens:
        raise FileNotFoundError("TRUNK12 não encontrado em data/trunk12. Rode scripts/preparar_dados.py trunk12.")
    rotulos = [c for _, c in itens]
    treino, resto = train_test_split(itens, test_size=val + teste, stratify=rotulos, random_state=SEMENTE)
    v, t = train_test_split(resto, test_size=teste / (val + teste), stratify=[c for _, c in resto],
                            random_state=SEMENTE)
    return {"train": treino, "val": v, "test": t}


def folhas() -> dict[str, list[tuple[Path, int]]]:
    """Divisão das folhas já feita por observação em scripts/preparar_dados.py (manifesto.csv)."""
    manifesto = DATA / "folhas" / "manifesto.csv"
    if not manifesto.exists():
        raise FileNotFoundError("Folhas não encontradas. Rode scripts/preparar_dados.py listar e baixar.")
    out: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    with open(manifesto, encoding="utf-8") as fh:
        for linha in csv.DictReader(fh):
            out[linha["split"]].append((DATA / linha["caminho"], CLASSES.index(linha["especie"])))
    return out


def _manifesto(caminho: Path, por_split: bool = True) -> dict[str, list[tuple[Path, int]]]:
    out: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    with open(caminho, encoding="utf-8") as fh:
        for linha in csv.DictReader(fh):
            out[linha.get("split", "test") if por_split else "test"].append(
                (DATA / linha["caminho"], CLASSES.index(linha["especie"])))
    return out


def casca_campo() -> list[tuple[Path, int]]:
    """Teste de campo: fotos de casca do GBIF que nunca entram em treino nem validação."""
    m = DATA / "casca_campo" / "manifesto.csv"
    return _manifesto(m, por_split=False)["test"] if m.exists() else []


def casca_mista() -> dict[str, list[tuple[Path, int]]]:
    """TRUNK12 + fotos de casca do GBIF (treino e validação). O teste continua sendo o do TRUNK12;
    o teste de campo fica separado em casca_campo()."""
    t = trunk12()
    g = _manifesto(DATA / "casca_gbif" / "manifesto.csv")
    return {"train": t["train"] + g["train"], "val": t["val"] + g["val"], "test": t["test"]}


# ------------------------------------------------------------------ base brasileira (iNaturalist, 21/11)

# Nome aceito (classe) -> nome popular. Gênero inteiro quando o nome aceito não tem epíteto.
POPULAR_BR = {
    "Schinus terebinthifolia": "Aroeira-pimenteira", "Handroanthus": "Ipês",
    "Erythrina speciosa": "Mulungu", "Spathodea campanulata": "Espatódea",
    "Paubrasilia echinata": "Pau-brasil", "Mangifera indica": "Mangueira", "Delonix regia": "Flamboyant",
    "Terminalia catappa": "Chapéu-de-sol", "Pleroma granulosum": "Quaresmeira", "Pachira aquatica": "Munguba",
    "Cenostigma pluviosum": "Sibipiruna", "Bauhinia variegata": "Pata-de-vaca",
    "Lagerstroemia indica": "Resedá", "Syzygium cumini": "Jambolão", "Peltophorum dubium": "Canafístula",
    "Moquilea tomentosa": "Oiti", "Tipuana tipu": "Tipuana", "Libidibia ferrea": "Pau-ferro",
}
TRONCO_BR = ["Paubrasilia echinata", "Handroanthus", "Schinus terebinthifolia"]
# As 10 espécies com mais fotos de folha na base baixada (scripts/baixar_inat_brasil.py dividir, 21/11)
FOLHA_BR = ["Schinus terebinthifolia", "Handroanthus", "Paubrasilia echinata", "Terminalia catappa",
            "Pachira aquatica", "Syzygium cumini", "Mangifera indica", "Pleroma granulosum",
            "Cenostigma pluviosum", "Erythrina speciosa"]

# Modo -> tipo de imagem ("tronco" usa patches; "folha" usa imagem inteira + 5 recortes)
TIPO = {"casca": "tronco", "folha": "folha", "tronco_br": "tronco", "folha_br": "folha"}


def e_tronco(modo: str) -> bool:
    return TIPO[modo] == "tronco"


def classes(modo: str) -> list[str]:
    return {"casca": CLASSES, "folha": CLASSES, "tronco_br": TRONCO_BR, "folha_br": FOLHA_BR}[modo]


def nome(modo: str, classe: str) -> tuple[str, str]:
    """(nome popular, nome científico exibido) de uma classe."""
    if modo in ("casca", "folha"):
        return ESPECIES[classe]
    cientifico = classe if " " in classe else f"{classe} sp."
    return POPULAR_BR[classe], cientifico


def inat_br(modo: str) -> dict[str, list[tuple[Path, int]]]:
    """Fotos do iNaturalist no Brasil que o PaliGemma marcou como tronco ou folha, divididas por observação."""
    manifesto = DATA / "inat_br" / "manifesto.csv"
    if not manifesto.exists():
        raise FileNotFoundError("Base brasileira não encontrada. Rode scripts/baixar_inat_brasil.py.")
    cls, parte = classes(modo), TIPO[modo]
    out: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    with open(manifesto, encoding="utf-8") as fh:
        for l in csv.DictReader(fh):
            if l.get("parte") == parte and l["taxon"] in cls and l.get("split"):
                out[l["split"]].append((DATA / l["caminho"], cls.index(l["taxon"])))
    return out


def carregar(modo: str) -> dict[str, list[tuple[Path, int]]]:
    if modo == "casca":
        return casca_mista() if (DATA / "casca_gbif" / "manifesto.csv").exists() else trunk12()
    if modo == "folha":
        return folhas()
    return inat_br(modo)

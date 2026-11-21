"""Listas de imagens com rótulo e split para os dois modos."""
from __future__ import annotations

import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
SEMENTE = 42

# Pasta do TRUNK12 -> (nome popular em português, nome científico). A mesma ordem define o índice da classe.
ESPECIES = {
    "alder": ("Amieiro", "Alnus glutinosa"),
    "beech": ("Faia", "Fagus sylvatica"),
    "birch": ("Bétula", "Betula pendula"),
    "chestnut": ("Castanheiro", "Castanea sativa"),
    "ginkgo biloba": ("Ginkgo", "Ginkgo biloba"),
    "hornbeam": ("Carpino", "Carpinus betulus"),
    "horse chestnut": ("Castanheiro-da-índia", "Aesculus hippocastanum"),
    "linden": ("Tília", "Tilia cordata"),
    "oak": ("Carvalho", "Quercus robur"),
    "oriental plane": ("Plátano-oriental", "Platanus orientalis"),
    "pine": ("Pinheiro", "Pinus sylvestris"),
    "spruce": ("Abeto", "Picea abies"),
}
CLASSES = list(ESPECIES)


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


def carregar(modo: str) -> dict[str, list[tuple[Path, int]]]:
    if modo == "casca":
        return casca_mista() if (DATA / "casca_gbif" / "manifesto.csv").exists() else trunk12()
    return folhas()

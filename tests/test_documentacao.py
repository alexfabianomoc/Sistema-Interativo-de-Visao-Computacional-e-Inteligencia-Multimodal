"""Conferência automática da documentação: o README e os slides batem com os valores reais do projeto?

Verifica número de testes, fotos de demonstração, métricas principais, taxonomia, pesos, número de slides
e o uso de "casca" como nome de modo (o modo se chama "tronco"; "casca" só para a superfície do tronco
e para nomes de arquivos já existentes).
"""
import json
import re
import zipfile
from pathlib import Path

import pytest

from src.metrics import dados

RAIZ = Path(__file__).resolve().parents[1]
README = (RAIZ / "README.md").read_text(encoding="utf-8")
PPTX = RAIZ / "docs" / "apresentacao" / "apresentacao.pptx"
METRICAS = RAIZ / "docs" / "metricas"


def _texto_slides() -> tuple[int, str]:
    """(número de slides, texto de slides e notas), lido direto do XML do PPTX."""
    if not PPTX.exists():
        return 0, ""
    with zipfile.ZipFile(PPTX) as z:
        nomes = z.namelist()
        slides = [n for n in nomes if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        partes = slides + [n for n in nomes if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n)]
        texto = " ".join(" ".join(re.findall(r"<a:t>([^<]*)</a:t>", z.read(n).decode("utf-8"))) for n in partes)
    return len(slides), texto


N_SLIDES, SLIDES = _texto_slides()


def _pct(valor: float) -> str:
    if round(valor, 4) == 1:
        return "100%"
    return f"{valor * 100:.1f}".replace(".", ",") + "%"


def _numero(padrao: str, texto: str = README) -> int:
    m = re.search(padrao, texto)
    assert m, f"padrão não encontrado no README: {padrao}"
    return int(m.group(1).replace(".", ""))


def test_numero_de_testes_no_readme(request):
    arquivos = {Path(str(i.fspath)).name for i in request.session.items}
    if arquivos != {p.name for p in (RAIZ / "tests").glob("test_*.py")}:
        pytest.skip("só confere quando a suíte inteira é executada")
    assert _numero(r"Hoje são (\d+) testes") == len(request.session.items)


def test_fotos_de_demonstracao():
    fotos = [p for p in (RAIZ / "data_samples").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    assert _numero(r"\((\d+) fotos em `data_samples/`") == len(fotos)
    assert _numero(r"`data_samples/` \| (\d+) fotos de demonstração") == len(fotos)


@pytest.mark.parametrize("arquivo,chave", [("convnext_tiny_tronco_br.json", "acuracia"),
                                          ("convnext_tiny_folha_br.json", "acuracia"),
                                          ("convnext_tiny_folha_br.json", "acuracia_top3"),
                                          ("convnext_tiny_casca.json", "acuracia"),
                                          ("convnext_tiny_folha.json", "acuracia")])
def test_metricas_principais_no_readme(arquivo, chave):
    valor = json.loads((METRICAS / arquivo).read_text(encoding="utf-8"))[chave]
    assert _pct(valor) in README, f"{arquivo}:{chave} = {_pct(valor)} não aparece no README"


def test_fusao_da_mesma_arvore():
    f = json.loads((METRICAS / "fusao_br.json").read_text(encoding="utf-8"))["tabela"][0]
    for chave in ("Só casca", "Só folha", "Casca + folha"):
        assert _pct(f[chave]) in README
    assert f"| {f['Pares']} |" in README


@pytest.mark.skipif(not PPTX.exists(), reason="apresentação ausente")
def test_slides_com_as_metricas_e_contagem():
    for arquivo, chave in (("convnext_tiny_tronco_br.json", "acuracia"), ("convnext_tiny_folha_br.json", "acuracia")):
        valor = json.loads((METRICAS / arquivo).read_text(encoding="utf-8"))[chave]
        assert _pct(valor) in SLIDES, f"{_pct(valor)} não aparece nos slides"
    f = json.loads((METRICAS / "fusao_br.json").read_text(encoding="utf-8"))["tabela"][0]
    assert _pct(f["Casca + folha"]) in SLIDES
    assert _numero(r"apresentação de (\d+) slides") == N_SLIDES


def test_taxonomia_exoticas():
    genero = sum(c.endswith(" sp.") for _, c in dados.ESPECIES.values())
    especie = len(dados.ESPECIES) - genero
    m = re.search(r"Das (\d+) classes, (\d+) ficam no nível de espécie e (\d+) são exibidas", README)
    assert m and tuple(map(int, m.groups())) == (len(dados.ESPECIES), especie, genero)


def test_quantidade_de_pesos():
    pesos = json.loads((RAIZ / "scripts" / "pesos.json").read_text(encoding="utf-8"))
    assert _numero(r"Os (\d+) pesos") == len(pesos["arquivos"])
    assert len(pesos.get("pastas_drive", [])) >= 1


# "casca" é permitido para a superfície do tronco e para nomes de arquivos existentes
CASCA_PERMITIDA = [
    r"`data_samples/casca/`", r"`casca/`", r"baixar_cascas_\w+\.py", r"casca_campo", r"casca_gbif",
    r"[Aa] casca é textura", r"casca muito de perto", r"confundid[oa]s pela casca", r"textura da casca",
    r"casca acinzentada", r"casca rugosa", r"sobre a casca", r"placas da casca", r"casca do tronco",
    r"\"bark\"|\*bark\*", r"_casca\.", r"modo casca \(interno\)",
]
CASCA_COMO_MODO = [r"\bcasca \+ folha", r"\bSó casca\b", r"\bmodo casca\b", r"\bFusão casca\b", r"\bCasca \+ folha",
                   r"\bCasca de campo\b", r"\bcasca de campo\b", r"\bmodos? (?:de )?casca\b", r"\bCasca:"]


@pytest.mark.parametrize("nome,texto", [("README", README), ("slides", SLIDES)])
def test_casca_nao_e_nome_de_modo(nome, texto):
    limpo = texto
    for padrao in CASCA_PERMITIDA:
        limpo = re.sub(padrao, "", limpo)
    achados = [m.group(0) for p in CASCA_COMO_MODO for m in re.finditer(p, limpo)]
    assert not achados, f"{nome} usa 'casca' como nome de modo: {achados}"

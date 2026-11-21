"""Gera docs/apresentacao/apresentacao.pptx (python-pptx)."""
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

RAIZ = Path(r"F:\Curso Engenharia de Computação\2026-2\Computação Gráfica, Realidade Virtual e Metaverso\Atividades\Sistema Interativo de Visão Computacional e Inteligência Multimodal")
IMG = RAIZ / "docs" / "apresentacao" / "img"
GAL = RAIZ / "docs" / "galeria"

VERDE = RGBColor(0x1F, 0x4D, 0x2B)      # dominante (mata)
MUSGO = RGBColor(0x5E, 0x9A, 0x5B)
CLARO = RGBColor(0xEE, 0xF5, 0xEC)
IPE = RGBColor(0xE8, 0xA9, 0x00)        # acento (ipê-amarelo)
TEXTO = RGBColor(0x1E, 0x29, 0x22)
CINZA = RGBColor(0x5B, 0x66, 0x5E)
BRANCO = RGBColor(0xFF, 0xFF, 0xFF)
TIT, CORPO = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333


def fundo(s, cor):
    f = s.background.fill
    f.solid()
    f.fore_color.rgb = cor


def caixa(s, x, y, w, h, texto="", tam=16, cor=TEXTO, negrito=False, fonte=CORPO, alin=PP_ALIGN.LEFT,
          ancora=MSO_ANCHOR.TOP, italico=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = ancora
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    linhas = texto if isinstance(texto, list) else [texto]
    for i, l in enumerate(linhas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = alin
        r = p.add_run()
        r.text = l
        r.font.size, r.font.bold, r.font.italic = Pt(tam), negrito, italico
        r.font.color.rgb, r.font.name = cor, fonte
    return tb


def lista(s, x, y, w, h, itens, tam=16, cor=TEXTO, espaco=8):
    """itens: str ou (negrito, resto)."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    for i, it in enumerate(itens):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(espaco)
        forte, resto = it if isinstance(it, tuple) else ("", it)
        if forte:
            r = p.add_run(); r.text = forte
            r.font.bold, r.font.size, r.font.color.rgb, r.font.name = True, Pt(tam), cor, CORPO
        r = p.add_run(); r.text = resto
        r.font.size, r.font.color.rgb, r.font.name = Pt(tam), cor, CORPO
    return tb


def cartao(s, x, y, w, h, cor=CLARO):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.08
    sh.fill.solid(); sh.fill.fore_color.rgb = cor
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def circulo(s, x, y, d, texto, cor=VERDE, tam=16):
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    sh.fill.solid(); sh.fill.fore_color.rgb = cor
    sh.line.fill.background()
    tf = sh.text_frame
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = texto
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(tam), True, BRANCO, CORPO
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


def titulo(s, texto, sub=None):
    caixa(s, 0.6, 0.45, W - 1.2, 0.8, texto, 34, VERDE, True, TIT)
    if sub:
        caixa(s, 0.6, 1.2, W - 1.2, 0.45, sub, 16, CINZA, italico=True)


def rodape(s, n):
    caixa(s, 0.6, 7.0, 8, 0.3, "Alex Fabiano Silva · RA 031600 · Projeto Final de Visão Computacional", 10, CINZA)
    caixa(s, W - 1.6, 7.0, 1.0, 0.3, str(n), 10, CINZA, alin=PP_ALIGN.RIGHT)


def imagem(s, caminho, x, y, w=None, h=None):
    return s.shapes.add_picture(str(caminho), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def novo(notas, escuro=False):
    s = prs.slides.add_slide(BLANK)
    fundo(s, VERDE if escuro else BRANCO)
    s.notes_slide.notes_text_frame.text = notas
    return s


def grafico_barras(s, x, y, w, h, cats, series, titulo_g, cores, maximo=1.0):
    cd = CategoryChartData()
    cd.categories = cats
    for nome, vals in series:
        cd.add_series(nome, vals)
    g = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    g.has_title = True
    g.chart_title.text_frame.text = titulo_g
    tp = g.chart_title.text_frame.paragraphs[0].runs[0].font
    tp.size, tp.bold, tp.color.rgb, tp.name = Pt(16), True, TEXTO, CORPO
    g.has_legend = len(series) > 1
    if g.has_legend:
        g.legend.position = XL_LEGEND_POSITION.BOTTOM
        g.legend.include_in_layout = False
        g.legend.font.size, g.legend.font.name, g.legend.font.color.rgb = Pt(12), CORPO, CINZA
    va = g.value_axis
    va.maximum_scale, va.minimum_scale = maximo, 0
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE3, 0xDD)
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0%", False
    for ax in (va, g.category_axis):
        ax.tick_labels.font.size, ax.tick_labels.font.name = Pt(12), CORPO
        ax.tick_labels.font.color.rgb = CINZA
        ax.format.line.color.rgb = RGBColor(0xC8, 0xD0, 0xC8)
    g.category_axis.has_major_gridlines = False
    pl = g.plots[0]
    pl.gap_width, pl.overlap = 60, -10
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.number_format, dl.number_format_is_linked = "0.0%", False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.name, dl.font.color.rgb = Pt(11), CORPO, TEXTO
    for serie, cor in zip(pl.series, cores):
        serie.format.fill.solid(); serie.format.fill.fore_color.rgb = cor
    return g


n = 0
# 1 ---------------------------------------------------------------- título
s = novo("Apresentar o tema em uma frase: uma aplicação que sugere a espécie de árvores urbanas brasileiras pela "
         "foto do tronco, da folha ou das duas, mostra onde a imagem justificou a resposta e encaminha os casos "
         "duvidosos para um profissional.", escuro=True)
caixa(s, 0.8, 1.7, 11.5, 1.8, "Identificação de árvores urbanas brasileiras por visão computacional", 44, BRANCO, True, TIT)
caixa(s, 0.8, 3.75, 11.5, 0.6, "Tronco, folha e as duas juntas, com mapa de calor, revisão por profissional e um VLM local", 20, RGBColor(0xCF, 0xE3, 0xCB))
caixa(s, 0.8, 5.3, 11.5, 1.2, ["Alex Fabiano Silva · RA 031600",
                                "Computação Gráfica, Realidade Virtual e Metaverso · 2026-2 · Prof. Pedro Cândido"], 16, BRANCO)
circulo(s, 11.2, 5.25, 1.2, "27/11", IPE, 18)
n += 1

# 2 ---------------------------------------------------------------- problema
s = novo("Quem usa: estudantes, equipes de arborização urbana e guardas-parque. A saída é sempre uma sugestão com "
         "confiança. Abaixo do limiar, o resultado é inconclusivo e vai para revisão por biólogo (CRBio) ou "
         "engenheiro (CREA).")
n += 1
titulo(s, "Problema e usuário", "Triagem rápida da espécie em campo, com a decisão final para um profissional habilitado")
lista(s, 0.6, 2.0, 6.3, 4.6, [
    ("Usuário: ", "estudantes, equipes de arborização urbana e guardas-parque"),
    ("Entrada: ", "foto do tronco, da folha ou das duas; upload, amostra, câmera ou drone (GPS do EXIF)"),
    ("Saída: ", "espécie mais provável com confiança, top-k, mapa de calor e imagem exportável"),
    ("Abaixo do limiar: ", "\"inconclusivo\" e envio para revisão por CRBio ou CREA"),
    ("Não é: ", "identificação botânica definitiva"),
], 17)
for i, (num, rot) in enumerate([("10", "espécies brasileiras\nno modo folha"), ("3", "espécies brasileiras\nno modo tronco"),
                                ("12", "exóticas europeias\ncomo reserva")]):
    y = 2.0 + i * 1.55
    cartao(s, 7.6, y, 5.1, 1.35)
    caixa(s, 7.9, y + 0.2, 1.4, 0.95, num, 48, VERDE, True, TIT, ancora=MSO_ANCHOR.MIDDLE)
    caixa(s, 9.4, y + 0.25, 3.1, 0.9, rot.split("\n"), 16, TEXTO, ancora=MSO_ANCHOR.MIDDLE)
rodape(s, n)

# 3 ---------------------------------------------------------------- evolução
s = novo("Contar a evolução como uma sequência de decisões guiadas por dados: cada mudança veio de um teste. "
         "Destaque a queda de 100% para 30,5% nas fotos de campo, que motivou a troca de base, e o teste de "
         "rendimento que definiu as espécies brasileiras.")
n += 1
titulo(s, "Como o projeto evoluiu", "Cada mudança de rumo veio de um teste, registrada no diário e nos commits")
etapas = [("1", "Casca europeia", "TRUNK12 (12 espécies). Pl@ntNet-300K não tem nenhuma espécie em comum"),
          ("2", "Folha via GBIF", "Fotos de folha das mesmas 12 espécies: a fusão casca + folha fica possível"),
          ("3", "Teste de campo", "100% no TRUNK12, mas 30,5% em fotos reais; com fotos de campo no treino, 70,6%"),
          ("4", "Foco no Brasil", "Teste de rendimento com 18 árvores urbanas do iNaturalist define as espécies"),
          ("5", "Reserva", "\"Casca\" vira \"tronco\"; o TRUNK12 passa a atender só árvores exóticas")]
for i, (num, cab, txt) in enumerate(etapas):
    x = 0.6 + i * 2.5
    circulo(s, x + 0.75, 2.2, 0.8, num, IPE if i == 3 else VERDE, 20)
    if i < 4:
        ln = s.shapes.add_connector(1, Inches(x + 1.6), Inches(2.6), Inches(x + 3.2), Inches(2.6))
        ln.line.color.rgb, ln.line.width = MUSGO, Pt(2)
    caixa(s, x, 3.25, 2.3, 0.5, cab, 18, VERDE, True, alin=PP_ALIGN.CENTER)
    caixa(s, x + 0.05, 3.85, 2.2, 2.4, txt, 14, TEXTO, alin=PP_ALIGN.CENTER)
caixa(s, 0.6, 6.3, 12, 0.45, "20/11 a 21/11/2026 · decisões do autor em cada etapa", 12, CINZA, italico=True)
rodape(s, n)

# 4 ---------------------------------------------------------------- dados
s = novo("Explicar de onde vêm as imagens e as licenças. A base brasileira é do iNaturalist, com grau de pesquisa e "
         "licença Creative Commons; as partes (tronco ou folha) foram separadas pelo PaliGemma. A divisão é por "
         "observação, para que fotos da mesma árvore não fiquem no treino e no teste.")
n += 1
titulo(s, "Dados e licenças", "Nenhuma base é versionada; autor e licença de cada foto ficam nos manifestos")
linhas = [("Base", "Uso", "Fotos", "Licença"),
          ("iNaturalist, Brasil", "Tronco (3) e folha (10), divisão 70/15/15 por observação", "12.976 (18 árvores)", "Creative Commons, por foto"),
          ("TRUNK12 (Švab, 2014)", "Reserva de exóticas: tronco", "393", "Sem licença explícita; citação pedida"),
          ("GBIF, Pl@ntNet observations", "Reserva de exóticas: folha e tronco de campo", "5.168 + 2.051", "CC BY-SA, por foto"),
          ("Amostras do projeto", "Demonstração na aplicação", "50", "Créditos em CREDITOS.csv")]
tab = s.shapes.add_table(len(linhas), 4, Inches(0.6), Inches(2.0), Inches(12.1), Inches(3.6)).table
for j, wdt in enumerate((3.0, 4.6, 2.0, 2.5)):
    tab.columns[j].width = Inches(wdt)
for i, linha in enumerate(linhas):
    for j, v in enumerate(linha):
        c = tab.cell(i, j)
        c.text = v
        c.fill.solid(); c.fill.fore_color.rgb = VERDE if i == 0 else (CLARO if i % 2 else BRANCO)
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                r.font.size, r.font.name = Pt(14), CORPO
                r.font.bold = i == 0
                r.font.color.rgb = BRANCO if i == 0 else TEXTO
caixa(s, 0.6, 5.95, 12, 0.8, "Por que não a Pl@ntNet-300K: nenhuma das 1.081 espécies coincide com as do TRUNK12. Por que não a "
      "base da UFPR: mostra o corte da madeira, não o tronco da árvore em pé.", 14, CINZA)
rodape(s, n)

# 5 ---------------------------------------------------------------- pipeline
s = novo("Percorrer a pipeline da esquerda para a direita. Ressaltar que o VLM é opcional e só interpreta: quem "
         "classifica é a CNN. A fusão combina as probabilidades do tronco e da folha.")
n += 1
titulo(s, "Arquitetura e pipeline", "Os sete componentes do enunciado, em módulos separados em src/")
passos = [("Aquisição", "upload, amostra,\ncâmera, drone"), ("Preparação", "validação, ROI,\nrotação, CLAHE"),
          ("Análise", "CNN por patches\nou 5 recortes"), ("Representação", "Grad-CAM, grade,\ncontorno, legenda"),
          ("Interação", "Streamlit,\nparâmetros, revisão"), ("Avaliação", "acurácia, F1,\nmatriz, tempo")]
for i, (cab, txt) in enumerate(passos):
    x = 0.6 + i * 2.1
    cartao(s, x, 2.3, 1.85, 2.0, CLARO if i != 2 else RGBColor(0xFF, 0xF4, 0xD6))
    caixa(s, x + 0.1, 2.45, 1.65, 0.45, cab, 17, VERDE, True, alin=PP_ALIGN.CENTER)
    caixa(s, x + 0.1, 3.0, 1.65, 1.2, txt.split("\n"), 14, TEXTO, alin=PP_ALIGN.CENTER)
    if i < 5:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 1.87), Inches(3.15), Inches(0.21), Inches(0.3))
        a.fill.solid(); a.fill.fore_color.rgb = MUSGO; a.line.fill.background()
cartao(s, 2.7, 4.85, 3.9, 1.5, CLARO)
caixa(s, 2.9, 5.0, 3.5, 1.2, ["Fusão tronco + folha", "produto das probabilidades nas espécies em comum"], 15, TEXTO)
cartao(s, 6.9, 4.85, 3.9, 1.5, CLARO)
caixa(s, 7.1, 5.0, 3.5, 1.2, ["VLM PaliGemma 2 (opcional)", "descreve, verifica a entrada e responde, sem classificar"], 15, TEXTO)
rodape(s, n)

# 6 ---------------------------------------------------------------- pré-processamento
s = novo("Mostrar a figura: original, realce de textura e letterbox. Explicar por que cada operação existe. O CLAHE "
         "é aplicado só no canal L do espaço LAB, para realçar textura sem mudar as cores.")
n += 1
titulo(s, "Pré-processamento", "Seis operações, cada uma com um motivo ligado ao problema")
imagem(s, IMG / "pre_comparativo.jpg", 0.6, 1.95, w=7.4)
lista(s, 8.4, 1.95, 4.4, 4.8, [
    ("Validação: ", "formato e tamanho mínimo de 224 px"),
    ("Recorte (ROI): ", "foca o tronco ou a folha"),
    ("Rotação: ", "endireita troncos inclinados"),
    ("Letterbox: ", "224×224 sem distorcer a textura"),
    ("CLAHE no canal L: ", "realça textura sem mudar a cor"),
    ("Normalização ImageNet: ", "mesma escala do pré-treino"),
], 15)
rodape(s, n)

# 7 ---------------------------------------------------------------- método
s = novo("Tronco é textura que se repete, por isso patches com voto médio. Folha depende de forma, por isso imagem "
         "inteira mais cinco recortes. Comparamos três CNNs com ajuste fino e uma linha de base clássica.")
n += 1
titulo(s, "Método de visão computacional", "Ajuste fino de CNNs pré-treinadas no ImageNet, comparadas com métodos clássicos")
for i, (cab, itens) in enumerate([
        ("Tronco", ["Imagem reduzida a 448 px no menor lado", "Patches de 224 px, voto médio das probabilidades",
                    "Aumento: recorte, espelho, rotação, cor"]),
        ("Folha", ["Imagem inteira + 4 cantos + centro", "Média das probabilidades dos 5 recortes",
                   "Aumento: recorte com variação de escala"]),
        ("Modelos", ["ConvNeXt-Tiny (padrão)", "MobileNetV3-Large e ResNet-18",
                     "Clássicos: LBP + cor e forma (Hu) + cor, com SVM"])]):
    x = 0.6 + i * 4.15
    cartao(s, x, 2.0, 3.9, 3.4)
    circulo(s, x + 0.25, 2.25, 0.6, str(i + 1), IPE if i == 2 else VERDE, 16)
    caixa(s, x + 1.0, 2.33, 2.7, 0.5, cab, 20, VERDE, True)
    lista(s, x + 0.3, 3.15, 3.35, 3.1, itens, 15)
rodape(s, n)

# 8 ---------------------------------------------------------------- CG
s = novo("Mostrar o mapa de calor levado de volta à imagem original, a grade de patches colorida pela espécie votada "
         "e o contorno da folha. São oito itens da seção 7, quando o mínimo é quatro.")
n += 1
titulo(s, "Representação gráfica: 8 itens de Computação Gráfica", "Mapeamento de coordenadas, camadas com transparência, cores, contornos e ROI")
imagem(s, IMG / "tronco_gradcam.jpg", 0.6, 1.85, h=2.45)
imagem(s, IMG / "folha_gradcam.jpg", 4.4, 4.2, h=2.45)
lista(s, 8.3, 1.95, 4.5, 4.9, ["1. Mapeamento modelo → imagem → tela", "2. Redimensionamento com proporção",
                               "3. Rotação do recorte", "4. Camadas e transparência", "5. Cores, mapa de calor e legenda",
                               "6. Contornos e grade de patches", "9. Região de interesse", "10. Vista comparativa"], 15, espaco=6)
rodape(s, n)

# 9 ---------------------------------------------------------------- interface
s = novo("Demonstrar ao vivo: página Classificar, parâmetros na barra lateral, falhas tratadas com mensagens claras "
         "e a página de revisão por profissional.")
n += 1
titulo(s, "Interface, parâmetros e falhas", "Streamlit com três páginas; nada fecha a aplicação")
for i, (cab, itens) in enumerate([
        ("Páginas", ["Classificar: tronco, folha ou os dois", "Revisão: CRBio/CREA, grava em CSV", "Métricas: tabelas e matrizes"]),
        ("Parâmetros", ["Modo e método", "Limiar e top-k", "Patch, CLAHE e opacidade", "Consulta às exóticas", "Restaurar padrão"]),
        ("Falhas tratadas", ["Arquivo inválido ou HEIC", "Foto menor que 224 px", "Sem câmera", "Drone sem GPS",
                             "Pesos ausentes", "Confiança baixa"])]):
    x = 0.6 + i * 4.15
    cartao(s, x, 2.0, 3.9, 3.7)
    caixa(s, x + 0.3, 2.2, 3.3, 0.5, cab, 20, VERDE, True)
    lista(s, x + 0.3, 2.9, 3.35, 3.5, itens, 15, espaco=6)
rodape(s, n)

# 10 --------------------------------------------------------------- resultados BR
s = novo("Resultados no conjunto de teste brasileiro, nunca usado no treino. A ConvNeXt-Tiny vence nos dois modos. "
         "O clássico de folha quase só acerta a aroeira, que tem metade das fotos (F1 macro 0,10).")
n += 1
titulo(s, "Resultados no Brasil", "Acurácia no teste: tronco com 100 fotos, folha com 574")
grafico_barras(s, 0.6, 1.9, 8.2, 4.9, ["ConvNeXt-Tiny", "MobileNetV3", "ResNet-18", "Clássico"],
               [("Tronco (3 espécies)", (0.81, 0.77, 0.61, 0.55)), ("Folha (10 espécies)", (0.887, 0.831, 0.749, 0.439))],
               "Acurácia por método", [VERDE, IPE])
cartao(s, 9.2, 1.95, 3.5, 2.2)
caixa(s, 9.45, 2.1, 3.0, 0.9, "88,7%", 44, VERDE, True, TIT)
caixa(s, 9.45, 3.05, 3.0, 0.9, "folha, ConvNeXt-Tiny; top-3 de 97,4%", 14, TEXTO)
cartao(s, 9.2, 4.45, 3.5, 2.2)
caixa(s, 9.45, 4.6, 3.0, 0.9, "81,0%", 44, VERDE, True, TIT)
caixa(s, 9.45, 5.55, 3.0, 0.9, "tronco, ConvNeXt-Tiny; F1 macro 0,76", 14, TEXTO)
rodape(s, n)

# 11 --------------------------------------------------------------- fusão
s = novo("O resultado mais forte. São 35 observações do iNaturalist com foto do tronco e da folha da mesma árvore, "
         "no teste. Juntas, as duas fotos acertam 94,3%. Ressalva: a folha escolhe entre 10 espécies e a fusão entre 3.")
n += 1
titulo(s, "Tronco + folha da mesma árvore", "35 pares de teste: uma foto do tronco e uma da folha de cada árvore")
for i, (v, r, cor) in enumerate([("71,4%", "só tronco", MUSGO), ("77,1%", "só folha", MUSGO), ("94,3%", "tronco + folha", IPE)]):
    x = 0.9 + i * 4.1
    cartao(s, x, 2.3, 3.6, 3.0, CLARO if i < 2 else RGBColor(0xFF, 0xF4, 0xD6))
    caixa(s, x, 2.75, 3.6, 1.3, v, 60, VERDE, True, TIT, PP_ALIGN.CENTER)
    caixa(s, x, 4.2, 3.6, 0.6, r, 20, TEXTO, alin=PP_ALIGN.CENTER)
caixa(s, 0.9, 5.75, 11.6, 0.9, "Fusão tardia: produto das probabilidades, nas 3 espécies que existem nos dois modos "
      "(pau-brasil, ipês e aroeira). A folha sozinha escolhe entre 10 espécies.", 15, CINZA)
rodape(s, n)

# 12 --------------------------------------------------------------- acertos e erros
s = novo("Mostrar acertos e erros com o mapa de calor. Os erros mostram árvore inteira em vez de folha e espécies "
         "parecidas. Quatro de cinco erros ficaram abaixo do limiar e seriam marcados como inconclusivos.")
n += 1
titulo(s, "Acertos e erros", "Galeria com Grad-CAM (modelos de reserva, folhas europeias); verde acerta, vermelho erra")
imagem(s, GAL / "folha_acertos.jpg", 1.67, 1.85, w=10.0)
imagem(s, GAL / "folha_erros.jpg", 1.67, 4.4, w=10.0)
rodape(s, n)

# 13 --------------------------------------------------------------- exóticas e comparações
s = novo("A lição do TRUNK12: 100% no laboratório e 30,5% no campo. Com fotos de campo no treino, 70,6%. O BioCLIP, "
         "sem treino, fica atrás do ajuste fino nos três conjuntos.")
n += 1
titulo(s, "Reserva de exóticas e comparações", "Fotos padronizadas enganam; fotos de campo e ajuste fino resolvem parte do problema")
grafico_barras(s, 0.6, 1.9, 6.3, 4.9, ["TRUNK12", "Fotos de campo"],
               [("Só TRUNK12 no treino", (1.0, 0.305)), ("TRUNK12 + campo", (1.0, 0.706))],
               "Tronco europeu: antes e depois", [MUSGO, VERDE])
grafico_barras(s, 7.1, 1.9, 5.7, 4.9, ["TRUNK12", "Tronco campo", "Folha"],
               [("BioCLIP sem treino", (0.288, 0.478, 0.94)), ("ConvNeXt ajustada", (1.0, 0.706, 0.971))],
               "BioCLIP zero-shot × ajuste fino", [IPE, VERDE])
rodape(s, n)

# 14 --------------------------------------------------------------- VLM
s = novo("PaliGemma 2 3B roda local em 4 bits. Protocolo de 20 casos: nenhuma alucinação clara, perguntas fechadas "
         "12 de 12. Ele erra ao nomear a espécie, por isso não classifica. Em dois casos o gabarito estava errado e "
         "o VLM certo.")
n += 1
titulo(s, "VLM local: PaliGemma 2", "Interpreta e verifica; nunca substitui a CNN. 2,5 GB de VRAM, sem internet")
for i, (v, r) in enumerate([("15", "corretas"), ("3", "incompletas"), ("2", "incorretas"), ("0", "alucinadas")]):
    x = 0.6 + i * 2.05
    cartao(s, x, 2.0, 1.85, 1.9)
    caixa(s, x, 2.15, 1.85, 1.0, v, 48, VERDE if i != 3 else IPE, True, TIT, PP_ALIGN.CENTER)
    caixa(s, x, 3.2, 1.85, 0.5, r, 15, TEXTO, alin=PP_ALIGN.CENTER)
lista(s, 0.6, 4.3, 8.0, 2.5, [
    ("Uso: ", "descrição, verificação da entrada e pergunta livre"),
    ("Tradução local: ", "Opus-MT pt ↔ en; a resposta original aparece junto"),
    ("Latência: ", "mediana de 247 ms; descrições longas até 1,1 s"),
    ("Erro típico: ", "chamou tília de carvalho; a espécie fica com a CNN"),
], 15)
cartao(s, 9.0, 2.0, 3.7, 3.4)
caixa(s, 9.25, 2.2, 3.2, 0.5, "Também separa as partes", 18, VERDE, True)
lista(s, 9.25, 2.85, 3.2, 2.5, ["O PaliGemma classificou as 12.976 fotos do iNaturalist em tronco, folha ou outras.",
                               "Conferência à mão: 58% dos troncos e 67% das folhas eram úteis."], 15)
rodape(s, n)

# 15 --------------------------------------------------------------- limitações
s = novo("Ser transparente: as limitações também valem pontos na avaliação. Destacar a revisão taxonômica: o TRUNK12 "
         "não informa as espécies, então seis classes europeias ficam no nível de gênero.")
n += 1
titulo(s, "Limitações e revisão taxonômica", "O que os números não mostram")
lista(s, 0.6, 1.95, 6.2, 4.9, [
    ("Ruído de parte: ", "tronco e folha separados pelo VLM, com 58% e 67% de fotos úteis"),
    ("Rótulos da comunidade: ", "grau de pesquisa no iNaturalist não é parecer de especialista"),
    ("Tronco com 3 espécies: ", "as outras 15 candidatas não têm fotos de tronco suficientes"),
    ("Desbalanceio: ", "aroeira tem 1.528 folhas e mulungu 101"),
    ("Amostras próprias: ", "3 de 6 troncos de demonstração acertados com confiança"),
], 15)
cartao(s, 7.2, 1.95, 5.5, 3.7)
caixa(s, 7.5, 2.15, 4.9, 0.5, "Revisão taxonômica (GBIF e EPPO)", 18, VERDE, True)
lista(s, 7.5, 2.8, 4.9, 3.8, [
    "TRUNK12 só traz nomes populares em inglês",
    "6 classes europeias no nível de gênero: Alnus, Betula, Tilia, Quercus, Platanus e Pinus sp.",
    "Spruce exibido como abeto-da-noruega (Picea abies)",
    "Classes brasileiras usam o nome aceito; ipês como Handroanthus sp.",
], 15)
rodape(s, n)

# 16 --------------------------------------------------------------- futuro, ética, créditos, IA
s = novo("Trabalho futuro principal: guardar as fotos validadas por profissionais numa base própria para treinar de "
         "novo, com consentimento. Ética: tudo roda localmente e cada foto tem licença e autor. Uso de IA declarado.")
n += 1
titulo(s, "Trabalho futuro, ética, créditos e uso de IA")
blocos = [("Trabalho futuro", ["Guardar a foto após a revisão", "Base própria validada com manifesto",
                               "Treinar com ela, fora do teste", "Consentimento de uso no envio"]),
          ("Ética e privacidade", ["Tudo roda no computador do usuário", "Só o recorte vai ao VLM, sem EXIF",
                                   "Licença e autor de cada foto", "Bases não redistribuídas"]),
          ("Créditos e IA", ["iNaturalist, GBIF/Pl@ntNet, TRUNK12", "torchvision, PaliGemma 2, Opus-MT, BioCLIP",
                             "Claude Code: código e documentação, com revisão e decisões do autor"])]
for i, (cab, itens) in enumerate(blocos):
    x = 0.6 + i * 4.15
    cartao(s, x, 1.65, 3.9, 3.6)
    caixa(s, x + 0.3, 1.85, 3.3, 0.5, cab, 20, VERDE, True)
    lista(s, x + 0.3, 2.55, 3.35, 4.0, itens, 15, espaco=8)
rodape(s, n)

# 17 --------------------------------------------------------------- encerramento
s = novo("Passar para a demonstração ao vivo: classificar um tronco e uma folha, mostrar o mapa de calor, um caso "
         "inconclusivo indo para a revisão e uma pergunta ao VLM. Se a aplicação não abrir, usar o vídeo de segurança.",
         escuro=True)
n += 1
caixa(s, 0.8, 1.6, 11.5, 1.2, "Demonstração ao vivo", 44, BRANCO, True, TIT)
lista(s, 0.8, 3.0, 7.5, 3.0, ["Classificar um tronco e uma folha brasileiros", "Mapa de calor e grade de patches",
                              "Caso inconclusivo indo para a revisão", "Pergunta ao PaliGemma, com tradução"], 18, BRANCO)
caixa(s, 0.8, 6.2, 11.5, 0.5, "github.com/alexfabianomoc/Sistema-Interativo-de-Visao-Computacional-e-Inteligencia-Multimodal",
      14, RGBColor(0xCF, 0xE3, 0xCB))
circulo(s, 9.6, 2.9, 2.4, "Obrigado", IPE, 24)

saida = RAIZ / "docs" / "apresentacao" / "apresentacao.pptx"
prs.save(saida)
print("ok", n, saida)

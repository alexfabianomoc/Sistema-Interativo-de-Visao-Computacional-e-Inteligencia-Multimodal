# Sistema Interativo de Visão Computacional e Inteligência Multimodal

Identificação de espécies de árvores pela **foto da casca**, pela **foto da folha** ou pelas **duas juntas**, com mapa de calor sobre a imagem, revisão manual por profissional habilitado e um assistente visual local (PaliGemma 2) para perguntas sobre a imagem.

**Autor:** Alex Fabiano Silva — RA 031600 (trabalho individual)
**Disciplina:** Computação Gráfica, Realidade Virtual e Metaverso — 2026-2 — Prof. Pedro Cândido

---

## 1. Problema e usuário

Estudantes, guardas-parque e equipes de arborização urbana precisam de uma triagem rápida da espécie de uma árvore em campo. A aplicação recebe uma foto e devolve a espécie mais provável **com grau de confiança** e a região da imagem que justificou a resposta. É uma sugestão, não uma identificação botânica definitiva: abaixo de um limiar de confiança o resultado é **"inconclusivo"** e a imagem pode ir para revisão por um biólogo (CRBio) ou engenheiro florestal/agrônomo (CREA).

**Espécies (12, europeias, as do TRUNK12):** amieiro (*Alnus glutinosa*), faia (*Fagus sylvatica*), bétula (*Betula pendula*), castanheiro (*Castanea sativa*), ginkgo (*Ginkgo biloba*), carpino (*Carpinus betulus*), castanheiro-da-índia (*Aesculus hippocastanum*), tília (*Tilia cordata*), carvalho (*Quercus robur*), plátano-oriental (*Platanus orientalis*), pinheiro (*Pinus sylvestris*), abeto (*Picea abies*).

## 2. Funcionalidades

| Página | O que faz |
|---|---|
| **Classificar** | Entrada por upload, amostra, câmera do navegador ou foto de drone (GPS e altitude lidos do EXIF; sem GPS, pergunta se segue ou abre latitude/longitude validadas). Recorte da região de interesse e rotação com pré-visualização. Modos **casca**, **folha** e **casca + folha** (fusão). Top-k com barras, mapa de calor Grad-CAM, grade de patches colorida por espécie, contorno e máscara da folha (cada camada liga e desliga), legenda, vista comparativa e download do PNG final. Painel do PaliGemma 2 com tradução pt ↔ en. |
| **Revisão** | Fila de imagens inconclusivas (ou enviadas pelo usuário) com top-3 e mapa de calor. O profissional informa nome e registro (CRBio/CREA, validado), escolhe ou digita a espécie e deixa observação. Tudo vai para `revisoes/revisoes.csv` e vira a métrica de concordância modelo × especialista. |
| **Métricas** | Tabela de acurácia, top-3, F1 macro e tempo por imagem de todos os métodos; antes × depois do treino com fotos de campo; fusão casca + folha; matriz de confusão por modelo. |

### Parâmetros (barra lateral, sem editar código; valores iniciais em `config.yaml`)

| Parâmetro | Efeito |
|---|---|
| Modo | Casca, folha ou casca + folha |
| Método | ConvNeXt-Tiny (padrão), MobileNetV3-Large, ResNet-18 ou clássico (LBP/forma + SVM) |
| Limiar de confiança | Abaixo dele o resultado é "inconclusivo" |
| Top-k | Quantas espécies mostrar |
| Tamanho do patch | Grade mais fina ou mais grossa na casca |
| CLAHE (casca e folha) | Intensidade do realce de textura (0 desliga) |
| Opacidade do mapa | Transparência do mapa de calor |
| Restaurar padrão | Volta aos valores do `config.yaml` |

## 3. Instalação

Requisitos: **Python 3.11** (testado com 3.11.9), Windows 10/11 ou Linux. GPU NVIDIA opcional (testado em RTX 4050 Laptop 6 GB, CUDA 12.4); sem GPU tudo roda em CPU, mais devagar.

```bash
git clone https://github.com/alexfabianomoc/Sistema-Interativo-de-Visao-Computacional-e-Inteligencia-Multimodal.git
cd Sistema-Interativo-de-Visao-Computacional-e-Inteligencia-Multimodal
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

Sem GPU NVIDIA: troque `+cu124` por nada nas linhas do `torch` e `torchvision` do `requirements.txt` e use `--extra-index-url https://download.pytorch.org/whl/cpu`.

### Pesos dos modelos

Os pesos não ficam no Git. O script procura em `models/` e, se não achar, baixa do espelho no Google Drive:

```bash
python scripts/baixar_pesos.py
```

### PaliGemma 2 (opcional)

O modelo `google/paligemma2-3b-mix-224` exige aceitar a licença Gemma em <https://huggingface.co/google/paligemma2-3b-mix-224> e fazer login:

```bash
hf auth login          # cole um token do tipo Read
hf download google/paligemma2-3b-mix-224
```

Sem o modelo, o painel fica desativado com aviso e o resto da aplicação funciona normalmente. Os tradutores (Opus-MT) são baixados automaticamente na primeira pergunta.

## 4. Execução

```bash
streamlit run app.py
```

A aplicação abre em <http://localhost:8501>. Para testar sem câmera, use a fonte **Amostra** (24 fotos em `data_samples/`).

Testes automáticos: `python -m pytest -q tests`.

Instalação testada do zero em 21/11/2026 (clone novo, ambiente novo, pesos baixados do Drive): 14 testes passaram.

## 5. Arquitetura e pipeline

```
Aquisição ──> Preparação ──> Análise visual ──> Representação ──> Interação ──> Avaliação
(upload,      (validação,    (CNN por patches   (Grad-CAM,         (Streamlit,   (acurácia, F1,
 amostra,      ROI, rotação,  ou 5 recortes;     grade, contorno,   parâmetros,   matriz, tempo,
 câmera,       letterbox,     clássico LBP/      legenda, PNG)      revisão)      campo, fusão,
 drone/EXIF)   CLAHE, norm.)  forma + SVM)                                         concordância)
                                   │
                                   └──> Fusão casca + folha (produto das probabilidades)
                                   └──> VLM PaliGemma 2 (opcional): descrição, verificação, perguntas
```

| Pasta | Conteúdo |
|---|---|
| `app.py` | Ponto de entrada (navegação entre as 3 páginas, log em `logs/app.log`) |
| `src/acquisition/` | Abertura de uploads com orientação EXIF, amostras, GPS/altitude, validação de coordenadas |
| `src/preprocessing/` | Leitura segura (caminhos com acento), ROI, rotação, letterbox com mapeamento de coordenadas, CLAHE, normalização, segmentação da folha, patches |
| `src/models/` | CNNs e inferência (`cnn.py`), linhas de base clássicas (`classico.py`), serviço e fusão (`servico.py`), VLM (`vlm.py`), tradução (`traducao.py`) |
| `src/visualization/` | Grad-CAM levado à imagem original (`gradcam.py`) e camadas gráficas (`desenho.py`) |
| `src/metrics/` | Conjuntos e divisões (`dados.py`), fila de revisão e concordância (`revisao.py`) |
| `src/interface/` | Páginas Streamlit |
| `scripts/` | Preparação dos dados, treino, avaliação, pesos, teste do VLM |
| `tests/` | Testes de pré-processamento, casos de falha e da interface (AppTest) |
| `docs/` | Proposta, diário, métricas (`docs/metricas/*.json`), protocolo do VLM, galeria |
| `data_samples/` | 12 fotos de folha e 12 de casca para demonstração, com créditos em `CREDITOS.csv` |
| `models/` | Pesos (fora do Git) |

### Pré-processamento (seção 5 do enunciado)

| Operação | Por quê |
|---|---|
| Validação de formato e tamanho mínimo (224 px) | Evita falhas e entradas que a rede não consegue analisar |
| Recorte da região de interesse | Concentra a análise no tronco ou na folha e descarta céu, chão e prédios |
| Rotação | Endireita troncos fotografados inclinados (comum em drone) |
| Redimensionamento com proporção (letterbox) | A rede espera 224×224; distorcer a imagem mudaria a textura e a forma |
| CLAHE no canal L (LAB) | Realça a textura da casca (forte) e as nervuras da folha (suave) sem alterar as cores |
| Normalização ImageNet | Mesma distribuição das imagens de pré-treino das CNNs |
| Segmentação por cor (HSV) + abertura e fechamento | Isola a folha do fundo para a linha de base de forma e para o contorno desenhado |

### Método de visão

- **Casca:** a imagem reduzida (menor lado 448 px) é dividida em patches de 224 px; cada patch passa pela CNN e a espécie final é a média das probabilidades. A casca é textura que se repete pelo tronco, por isso patches funcionam bem.
- **Folha:** imagem inteira + 4 cantos + centro (cada recorte com 80% do menor lado), média das probabilidades. A folha depende de forma e contorno, por isso a imagem é vista inteira.
- **CNNs** (ConvNeXt-Tiny como padrão; MobileNetV3-Large e ResNet-18 para comparação) pré-treinadas no ImageNet com ajuste fino completo (AdamW, OneCycle, label smoothing 0,1, pesos por classe, precisão mista na GPU), melhor época escolhida pelo F1 macro de validação.
- **Linhas de base clássicas:** casca = histogramas LBP em três escalas + histograma de cor HSV + SVM RBF; folha = momentos de Hu, razão de aspecto, solidez e circularidade do maior contorno + histograma de cor + SVM.
- **Fusão casca + folha:** produto das probabilidades dos dois modelos, renormalizado (fusão tardia).

### Computação Gráfica (seção 7): 8 itens

| Item | Onde |
|---|---|
| 1. Mapeamento de coordenadas | Cada patch/recorte e seu Grad-CAM (224×224) voltam à posição na imagem original (`Letterbox.para_original`, `gradcam.mapa_completo`) |
| 2. Redimensionamento com proporção | `transforms.letterbox` |
| 3. Transformação geométrica | Rotação do recorte (`transforms.rotacionar`, sem cortar os cantos) |
| 4. Camadas e transparência | Mapa de calor, grade e máscara com opacidade ajustável, cada camada liga e desliga |
| 5. Cores, mapas de calor e legendas | Paleta de 12 cores por espécie, escala JET com legenda na imagem exportada |
| 6. Contornos | Contorno da folha segmentada e retângulos da grade de patches |
| 9. Região de interesse | Seleção da faixa horizontal e vertical com pré-visualização |
| 10. Visualização comparativa | Original, após CLAHE e resultado lado a lado |

## 6. Dados

| Conjunto | Uso | Tamanho | Licença |
|---|---|---|---|
| **TRUNK12** (Švab, 2014) | Casca: treino, validação e teste (70/15/15 estratificado, semente 42) | 393 fotos, 12 espécies | A confirmar com os autores; não versionado nem exibido em `data_samples/` |
| **GBIF — Pl@ntNet observations**, fotos marcadas como *leaf* | Folha: 5.168 fotos (até 500 por espécie), 80/10/10 **por observação** | 600 px | Cada foto CC BY-SA; autor e link em `data/folhas/manifesto.csv` |
| **GBIF — Pl@ntNet observations**, fotos marcadas como *bark* | Casca de campo: 1.779 para treino/validação + 272 só para teste | 600 px | CC BY-SA; autor e link nos manifestos |

Por que não a Pl@ntNet-300K: nenhuma das suas 1.081 espécies coincide com as do TRUNK12. A Forest Species Database (UFPR) mostra o corte da madeira, não a casca da árvore em pé. Detalhes em `docs/proposta.md`.

Preparação: `python scripts/preparar_dados.py trunk12 --zips <pasta dos zips>`, `... listar`, `... baixar`; `python scripts/baixar_cascas_campo.py`; `python scripts/baixar_cascas_treino.py`.

## 7. Resultados (conjuntos de teste nunca usados no treino)

Valores completos em `docs/metricas/` e na página Métricas.

### Casca

| Método | TRUNK12 (59 fotos) | Top-3 | F1 macro | Fotos de campo (272) | Top-3 campo | Tempo/imagem (GPU) |
|---|---|---|---|---|---|---|
| **ConvNeXt-Tiny (padrão)** | **100%** | 100% | 1,000 | **70,6%** | 87,9% | 124 ms |
| MobileNetV3-Large | 100% | 100% | 1,000 | 66,9% | 84,6% | 94 ms |
| ResNet-18 | 98,3% | 100% | 0,982 | 61,4% | 84,6% | 114 ms |
| LBP + cor + SVM | 72,9% | 93,2% | 0,690 | 27,6% | 51,5% | <1 ms (SVM) |

Antes de incluir fotos de campo no treino (só TRUNK12): MobileNetV3 100% no TRUNK12 e **30,5%** no campo; LBP 88,1% e 11,4%. O tempo da casca inclui todos os patches da foto.

### Folha (518 fotos de teste)

| Método | Acurácia | Top-3 | F1 macro | Tempo/imagem (GPU) |
|---|---|---|---|---|
| **ConvNeXt-Tiny (padrão)** | **97,1%** | 99,6% | 0,970 | 27 ms |
| MobileNetV3-Large | 96,1% | 98,8% | 0,942 | 23 ms |
| ResNet-18 | 93,4% | 98,8% | 0,914 | 26 ms |
| Forma (Hu) + cor + SVM | 26,5% | 53,3% | 0,213 | <1 ms (SVM) |

### Casca + folha (fusão, ConvNeXt-Tiny, 518 pares)

| Casca de | Só casca | Só folha | Casca + folha |
|---|---|---|---|
| TRUNK12 | 100% | 97,1% | 100% |
| Fotos de campo | 70,1% | 97,1% | **99,0%** |

### Limiar "inconclusivo" (0,6) nas fotos de campo

A ConvNeXt marca 54% das fotos de campo como inconclusivas; na galeria, 4 dos 5 erros de casca e 4 dos 5 de folha ficaram abaixo do limiar. O SVM clássico fica abaixo do limiar em 100% das fotos de campo, ou seja, não decide nunca. Galeria comentada de acertos e erros: `docs/galeria.md`.

### VLM

20 casos: 15 corretas, 3 incompletas, 2 incorretas, 0 alucinadas; perguntas fechadas 12/12; latência mediana 247 ms. Detalhes em `docs/vlm_protocolo.md`.

Treino (GPU RTX 4050): de 10 a 15 min por CNN.

## 8. Limitações

- **TRUNK12 é padronizado demais:** treinada só nele, a MobileNetV3 acerta 100% no teste do TRUNK12 mas só 30,5% em fotos de campo. O treino com fotos do GBIF elevou o campo para 66,9% (MobileNetV3) e 70,6% (ConvNeXt-Tiny), ainda longe do laboratório.
- **Ginkgo e plátano-oriental** têm poucas fotos no GBIF (66 e 102 folhas; 5 e 20 cascas), então seus resultados são menos confiáveis.
- **Fusão avaliada com pares de árvores diferentes** da mesma espécie, porque as bases não têm casca e folha da mesma árvore.
- **12 espécies europeias:** espécies fora da lista recebem a mais parecida; o limiar "inconclusivo" e a revisão manual existem para isso. Espécies brasileiras de arborização ficam como trabalho futuro.
- **Teste do TRUNK12 tem só 59 fotos**, por isso 100% não significa acerto garantido.
- **PaliGemma** descreve sempre em inglês (a tradução local resolve em parte) e pode alucinar; ver `docs/vlm_protocolo.md`.

## 9. Ética e privacidade

As imagens do GBIF são públicas, com licença e autor registrados por foto. A aplicação não envia imagens para serviços externos: CNNs, PaliGemma e tradução rodam no computador do usuário. Só o recorte vai ao VLM, sem EXIF nem GPS. Fotos de drone podem ter GPS; as coordenadas ficam só na fila de revisão local. Nenhuma base é redistribuída no repositório.

## 10. Fontes e créditos

**Dados:** TRUNK12 — M. Švab, *Tree bark dataset TRUNK12*, Univ. de Ljubljana, 2014. Affouard A, Joly A, Lombardo J, Champ J, Goëau H, Chouet M, Gresse H, Bonnet P (2025). *Pl@ntNet observations*, versão 1.9, conjunto publicado em CC BY 4.0, https://doi.org/10.15468/gtebaa, acessado via GBIF.org em 20 e 21/11/2026; fotos de voluntários da Pl@ntNet, em geral CC BY-SA; créditos por foto nos manifestos e em `data_samples/CREDITOS.csv`.

**Modelos e pesos:** MobileNetV3-Large, ResNet-18 e ConvNeXt-Tiny pré-treinados no ImageNet (torchvision, BSD-3). PaliGemma 2 3B mix 224 (Google, Gemma Terms of Use). Opus-MT `opus-mt-ROMANCE-en` (Apache 2.0) e `opus-mt-tc-big-en-pt` (CC BY 4.0), Helsinki-NLP.

**Bibliotecas:** PyTorch, torchvision, OpenCV, NumPy, Pillow, scikit-image, scikit-learn, pandas, Streamlit, Hugging Face Transformers, bitsandbytes, accelerate, PyYAML, pytest.

**Artigos:** Cotrim e Pedrini, *Identificação de Espécies de Árvores por Imagens de Tronco Utilizando Aprendizado de Máquina Profundo*, IC-Unicamp, 2019. Ratajczak et al., *Efficient Bark Recognition in the Wild*, VISAPP 2019. Selvaraju et al., *Grad-CAM*, ICCV 2017. Belfort et al., *Classificação de espécies de árvores em imagens de cenário urbano usando CNNs com Triplet Loss*, SEMISH 2026.

## 11. Uso de inteligência artificial

| Ferramenta | Finalidade | Partes afetadas | Revisão e validação |
|---|---|---|---|
| Claude (Anthropic), via Claude Code | Planejamento, pesquisa de bases, escrita e depuração de código, documentação | Todo o repositório | Cada etapa foi decidida pelo autor (escolha de bases, espécies, divisões, modelos, padrões); código validado por testes automáticos, execução na interface e métricas em conjuntos de teste separados |
| PaliGemma 2 (Google) | Componente opcional da aplicação: descrição e perguntas sobre a imagem | Painel do VLM | Protocolo de 20 casos em `docs/vlm_protocolo.md` |

## 12. Diário de desenvolvimento

Ver `docs/diario.md` (uma entrada por etapa, com os quatro marcos).

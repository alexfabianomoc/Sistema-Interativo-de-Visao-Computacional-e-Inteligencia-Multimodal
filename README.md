# Sistema Interativo de Visão Computacional e Inteligência Multimodal

Identificação de espécies de **árvores urbanas brasileiras** pela **foto do tronco**, pela **foto da folha** ou pelas **duas juntas**, com mapa de calor sobre a imagem, revisão manual por profissional habilitado e um assistente visual local (PaliGemma 2) para perguntas sobre a imagem.

**Autor:** Alex Fabiano Silva — RA 031600 (trabalho individual)
**Disciplina:** Computação Gráfica, Realidade Virtual e Metaverso — 2026-2 — Prof. Pedro Cândido

---

## 1. Problema e usuário

Estudantes, guardas-parque e equipes de arborização urbana precisam de uma triagem rápida da espécie de uma árvore em campo. A aplicação recebe uma foto e devolve a espécie mais provável **com grau de confiança** e a região da imagem que justificou a resposta. É uma sugestão, não uma identificação botânica definitiva: abaixo de um limiar de confiança o resultado é **"inconclusivo"** e a imagem pode ir para revisão por um biólogo (CRBio) ou engenheiro florestal/agrônomo (CREA).

**Foco no Brasil (decisão do autor em 21/11/2026).** Os modos principais usam fotos do iNaturalist tiradas no Brasil:

| Modo | Classes | Fotos (treino / validação / teste) |
|---|---|---|
| **Tronco** | 3: pau-brasil (*Paubrasilia echinata*), ipês (*Handroanthus* sp.), aroeira-pimenteira (*Schinus terebinthifolia*) | 380 / 81 / 100 |
| **Folha** | 10: aroeira-pimenteira, ipês, pau-brasil, chapéu-de-sol (*Terminalia catappa*), munguba (*Pachira aquatica*), jambolão (*Syzygium cumini*), mangueira (*Mangifera indica*), quaresmeira (*Pleroma granulosum*), sibipiruna (*Cenostigma pluviosum*), mulungu (*Erythrina speciosa*) | 2.726 / 539 / 574 |
| **Tronco + folha** | fusão nas 3 espécies que existem nos dois modos | pares da mesma árvore |

As espécies foram escolhidas por um teste de rendimento (`docs/teste_rendimento_inat.md`): entre 18 árvores urbanas candidatas, só essas três rendem fotos de tronco suficientes; as 10 de folha são as com mais fotos de folha.

**Reserva para árvores exóticas:** quando o modelo brasileiro fica abaixo do limiar, a aplicação consulta automaticamente os modelos treinados com o TRUNK12 (tronco) e com folhas europeias do GBIF, com 12 espécies europeias, e mostra o resultado como "possível espécie exótica". Esses modelos também podem ser escolhidos diretamente. Nomes conferidos no GBIF e no EPPO: [docs/auditoria_taxonomica.md](docs/auditoria_taxonomica.md) (seis das 12 classes europeias ficam no nível de gênero).

## 2. Funcionalidades

| Página | O que faz |
|---|---|
| **Classificar** | Entrada por upload, amostra, câmera do navegador ou foto de drone (GPS e altitude lidos do EXIF; sem GPS, pergunta se segue ou abre latitude/longitude validadas). Recorte da região de interesse e rotação com pré-visualização. Modos **tronco**, **folha** e **tronco + folha** (fusão), brasileiros, e os mesmos três modos para exóticas; abaixo do limiar, consulta automática do modelo de exóticas. Top-k com barras, mapa de calor Grad-CAM, grade de patches colorida por espécie, contorno e máscara da folha (cada camada liga e desliga), legenda, vista comparativa e download do PNG final. Painel do PaliGemma 2 com tradução pt ↔ en. |
| **Revisão** | Fila de imagens inconclusivas (ou enviadas pelo usuário) com top-3 e mapa de calor. O profissional informa nome e registro (CRBio/CREA, validado), escolhe ou digita a espécie e deixa observação. Tudo vai para `revisoes/revisoes.csv` e vira a métrica de concordância modelo × especialista. |
| **Métricas** | Tabela de acurácia, top-3, F1 macro e tempo por imagem de todos os métodos; antes × depois do treino com fotos de campo (exóticas); fusão tronco + folha; matriz de confusão por modelo. |

### Parâmetros (barra lateral, sem editar código; valores iniciais em `config.yaml`)

| Parâmetro | Efeito |
|---|---|
| Modo | Tronco, folha ou tronco + folha (Brasil); os mesmos para exóticas |
| Consultar exóticas | Liga ou desliga a consulta automática ao modelo de exóticas quando o resultado é inconclusivo |
| Método | ConvNeXt-Tiny (padrão), MobileNetV3-Large, ResNet-18 ou clássico (LBP/forma + SVM) |
| Limiar de confiança | Abaixo dele o resultado é "inconclusivo" |
| Top-k | Quantas espécies mostrar |
| Tamanho do patch | Grade mais fina ou mais grossa no tronco |
| CLAHE (tronco e folha) | Intensidade do realce de textura (0 desliga) |
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

A aplicação abre em <http://localhost:8501>. Para testar sem câmera, use a fonte **Amostra** (50 fotos em `data_samples/`, das bases brasileira e de exóticas).

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
| **iNaturalist, Brasil** (`place_id=6878`, grau de pesquisa) | Modos brasileiros: 12.976 fotos de 18 árvores urbanas; o PaliGemma separou cada foto em tronco, folha ou outras; divisão 70/15/15 **por observação** | 500 px (*medium*) | Só fotos com licença Creative Commons (CC0, BY, BY-SA, BY-NC e variantes); autor e link em `data/inat_br/manifesto.csv` |
| **TRUNK12** (Švab, 2014; ViCoS Lab, Univ. de Ljubljana) | Reserva de exóticas, tronco: treino, validação e teste (70/15/15 estratificado, semente 42) | 393 fotos JPEG 3000×4000 (Nikon COOLPIX S3000, a 20 cm do tronco), 12 espécies, 30 a 45 por espécie | Sem licença explícita; os autores pedem citação. Por cautela, não versionado nem exibido em `data_samples/` |
| **GBIF — Pl@ntNet observations**, fotos marcadas como *leaf* | Folha: 5.168 fotos (até 500 por espécie), 80/10/10 **por observação** | 600 px | Cada foto CC BY-SA; autor e link em `data/folhas/manifesto.csv` |
| **GBIF — Pl@ntNet observations**, fotos marcadas como *bark* | Casca de campo: 1.779 para treino/validação + 272 só para teste | 600 px | CC BY-SA; autor e link nos manifestos |

Por que não a Pl@ntNet-300K: nenhuma das suas 1.081 espécies coincide com as do TRUNK12. A Forest Species Database (UFPR) mostra o corte da madeira, não a casca da árvore em pé. Detalhes em `docs/proposta.md`.

Preparação da base brasileira: `python scripts/baixar_inat_brasil.py baixar`, `... classificar`, `... dividir`. Reserva de exóticas: `python scripts/preparar_dados.py trunk12 --zips <pasta dos zips>`, `... listar`, `... baixar`; `python scripts/baixar_cascas_campo.py`; `python scripts/baixar_cascas_treino.py`.

## 7. Resultados (conjuntos de teste nunca usados no treino)

Valores completos em `docs/metricas/` e na página Métricas.

### Tronco (Brasil, 100 fotos de teste)

| Método | Acurácia | F1 macro | Tempo/imagem (GPU) |
|---|---|---|---|
| **ConvNeXt-Tiny (padrão)** | **81,0%** | 0,761 | 22 ms |
| MobileNetV3-Large | 77,0% | 0,697 | 15 ms |
| ResNet-18 | 61,0% | 0,555 | 12 ms |
| LBP + cor + SVM | 55,0% | 0,382 | <1 ms (SVM) |

Com 3 classes, o top-3 é sempre 100% e não é informativo.

### Folha (Brasil, 574 fotos de teste)

| Método | Acurácia | Top-3 | F1 macro | Tempo/imagem (GPU) |
|---|---|---|---|---|
| **ConvNeXt-Tiny (padrão)** | **88,7%** | 97,4% | 0,831 | 27 ms |
| MobileNetV3-Large | 83,1% | 95,6% | 0,765 | 25 ms |
| ResNet-18 | 74,9% | 91,6% | 0,696 | 20 ms |
| Forma (Hu) + cor + SVM | 43,9% | 66,2% | 0,102 | <1 ms (SVM) |

O SVM de forma quase só acerta a aroeira, que tem metade das fotos; o F1 macro de 0,10 mostra isso.

### Tronco + folha (Brasil, pares da mesma árvore)

| Pares | Só tronco | Só folha | Tronco + folha |
|---|---|---|---|
| 35 | 71,4% | 77,1% | **94,3%** |

Cada par é uma observação do iNaturalist com foto do tronco e da folha da **mesma árvore**, no conjunto de teste. "Só folha" escolhe entre as 10 espécies; a fusão, entre as 3 em comum.

### Reserva de exóticas: tronco (TRUNK12 e fotos de campo do GBIF)

| Método | TRUNK12 (59 fotos) | Top-3 | F1 macro | Fotos de campo (272) | Top-3 campo | Tempo/imagem (GPU) |
|---|---|---|---|---|---|---|
| **ConvNeXt-Tiny (padrão)** | **100%** | 100% | 1,000 | **70,6%** | 87,9% | 124 ms |
| MobileNetV3-Large | 100% | 100% | 1,000 | 66,9% | 84,6% | 94 ms |
| ResNet-18 | 98,3% | 100% | 0,982 | 61,4% | 84,6% | 114 ms |
| LBP + cor + SVM | 72,9% | 93,2% | 0,690 | 27,6% | 51,5% | <1 ms (SVM) |

Antes de incluir fotos de campo no treino (só TRUNK12): MobileNetV3 100% no TRUNK12 e **30,5%** no campo; LBP 88,1% e 11,4%. O tempo da casca inclui todos os patches da foto.

### Reserva de exóticas: folha (518 fotos de teste)

| Método | Acurácia | Top-3 | F1 macro | Tempo/imagem (GPU) |
|---|---|---|---|---|
| **ConvNeXt-Tiny (padrão)** | **97,1%** | 99,6% | 0,970 | 27 ms |
| MobileNetV3-Large | 96,1% | 98,8% | 0,942 | 23 ms |
| ResNet-18 | 93,4% | 98,8% | 0,914 | 26 ms |
| Forma (Hu) + cor + SVM | 26,5% | 53,3% | 0,213 | <1 ms (SVM) |

### Reserva de exóticas: tronco + folha (ConvNeXt-Tiny, 518 pares de árvores diferentes)

| Casca de | Só casca | Só folha | Casca + folha |
|---|---|---|---|
| TRUNK12 | 100% | 97,1% | 100% |
| Fotos de campo | 70,1% | 97,1% | **99,0%** |

### Comparação zero-shot: BioCLIP sem treino

O BioCLIP (`imageomics/bioclip`, MIT) classifica comparando a foto com textos "a photo of the bark/leaf of *espécie*", sem nenhum ajuste nas nossas bases (`scripts/avaliar_bioclip.py`).

| Conjunto | BioCLIP (sem treino) | ConvNeXt-Tiny ajustada |
|---|---|---|
| TRUNK12 (59) | 28,8% | 100% |
| Casca de campo (272) | 47,8% | 70,6% |
| Folhas (518) | 94,0% | 97,1% |

O ajuste fino vence nos três conjuntos, com folga na casca. O BioCLIP vai mal nas fotos padronizadas de casca muito de perto (TRUNK12), que fogem das fotos comuns de biodiversidade, e quase empata nas folhas. Ressalva: ele foi treinado com milhões de fotos de biodiversidade, possivelmente incluindo fotos da Pl@ntNet/GBIF, o que pode favorecê-lo nos conjuntos do GBIF.

### Limiar "inconclusivo" (0,6) nas fotos de campo

A ConvNeXt marca 54% das fotos de campo como inconclusivas; na galeria, 4 dos 5 erros de casca e 4 dos 5 de folha ficaram abaixo do limiar. O SVM clássico fica abaixo do limiar em 100% das fotos de campo, ou seja, não decide nunca. Galeria comentada de acertos e erros: `docs/galeria.md`.

### VLM

20 casos: 15 corretas, 3 incompletas, 2 incorretas, 0 alucinadas; perguntas fechadas 12/12; latência mediana 247 ms. Detalhes em `docs/vlm_protocolo.md`.

Treino (GPU RTX 4050): de 10 a 15 min por CNN.

## 8. Limitações

- **Partes separadas automaticamente:** as fotos do iNaturalist não dizem se mostram tronco ou folha; o PaliGemma separou. Na conferência à mão, 58% das fotos marcadas como tronco e 67% das marcadas como folha eram realmente úteis. O treino e o teste brasileiros têm, portanto, ruído de rótulo de parte.
- **Rótulos de espécie da comunidade:** grau de pesquisa no iNaturalist significa concordância da comunidade, não identificação por especialista; a revisão manual existe para isso.
- **Tronco brasileiro com só 3 espécies**, porque as outras 15 candidatas não têm fotos de tronco suficientes; a aroeira tem só 76.
- **Folha desbalanceada:** aroeira tem 1.528 fotos e mulungu 101; o treino usa pesos por classe, mas o F1 macro fica abaixo da acurácia.
- **Ipês tratados como gênero** (*Handroanthus* sp.), juntando várias espécies.
- **TRUNK12 é padronizado demais:** treinada só nele, a MobileNetV3 acerta 100% no teste do TRUNK12 mas só 30,5% em fotos de campo. O treino com fotos do GBIF elevou o campo para 66,9% (MobileNetV3) e 70,6% (ConvNeXt-Tiny), ainda longe do laboratório.
- **Ginkgo e plátano** têm poucas fotos no GBIF (66 e 102 folhas; 5 e 20 cascas), então seus resultados são menos confiáveis.
- **Fusão avaliada com pares de árvores diferentes** da mesma espécie, porque as bases não têm casca e folha da mesma árvore.
- **12 espécies europeias:** espécies fora da lista recebem a mais parecida; o limiar "inconclusivo" e a revisão manual existem para isso. Espécies brasileiras de arborização ficam como trabalho futuro.
- **Nomes científicos:** o TRUNK12 não informa as espécies. Seis classes são exibidas só pelo gênero (*Alnus*, *Betula*, *Tilia*, *Quercus*, *Platanus*, *Pinus*), mas as fotos do GBIF de cada uma foram baixadas com uma única espécie (*A. glutinosa*, *B. pendula*, *T. cordata*, *Q. robur*, *P. orientalis*, *P. sylvestris*); outras espécies do mesmo gênero podem ser reconhecidas pior. Detalhes em `docs/auditoria_taxonomica.md`.
- **Teste do TRUNK12 tem só 59 fotos**, por isso 100% não significa acerto garantido.
- **PaliGemma** descreve sempre em inglês (a tradução local resolve em parte) e pode alucinar; ver `docs/vlm_protocolo.md`.

## 9. Trabalhos futuros

**Base própria de fotos validadas por profissionais.** A revisão manual já registra a espécie confirmada, o nome e o registro (CRBio/CREA) do profissional em `revisoes/revisoes.csv`, mas ainda não forma uma base de treino. Falta:

1. **Guardar a foto após a revisão.** Hoje a imagem é apagada da fila (`revisoes/fila/`) quando a revisão é registrada; sobra só a linha do CSV.
2. **Organizar uma base própria**, por exemplo `revisoes/base/<modo>/<espécie>/`, com um manifesto por foto: espécie validada, profissional, registro, data, origem e autorização de uso.
3. **Incluir essa base no treino**, sempre fora dos conjuntos de teste, para não inflar as métricas.
4. **Consentimento:** antes de enviar uma foto para a fila, o usuário deve concordar com o uso da imagem para treino e pesquisa; fotos sem esse consentimento ficam só na revisão.

## 10. Ética e privacidade

As imagens do GBIF são públicas, com licença e autor registrados por foto. A aplicação não envia imagens para serviços externos: CNNs, PaliGemma e tradução rodam no computador do usuário. Só o recorte vai ao VLM, sem EXIF nem GPS. Fotos de drone podem ter GPS; as coordenadas ficam só na fila de revisão local. Nenhuma base é redistribuída no repositório.

## 11. Fontes e créditos

**Dados:** iNaturalist (https://www.inaturalist.org), observações de grau de pesquisa no Brasil consultadas pela API em 21/11/2026; cada foto tem sua licença Creative Commons e autor registrados em `data/inat_br/manifesto.csv` e, para as amostras, em `data_samples/CREDITOS.csv`. TRUNK12 — Švab, M. *Computer-vision-based tree trunk recognition*. Trabalho de conclusão de bacharelado (orientador: Matej Kristan), Fakulteta za računalništvo in informatiko, Univerza v Ljubljani, 2014. Disponível em https://www.vicos.si/resources/trunk12/. Affouard A, Joly A, Lombardo J, Champ J, Goëau H, Chouet M, Gresse H, Bonnet P (2025). *Pl@ntNet observations*, versão 1.9, conjunto publicado em CC BY 4.0, https://doi.org/10.15468/gtebaa, acessado via GBIF.org em 20 e 21/11/2026; fotos de voluntários da Pl@ntNet, em geral CC BY-SA; créditos por foto nos manifestos e em `data_samples/CREDITOS.csv`.

**Modelos e pesos:** MobileNetV3-Large, ResNet-18 e ConvNeXt-Tiny pré-treinados no ImageNet (torchvision, BSD-3). BioCLIP (Stevens et al., *BioCLIP: A Vision Foundation Model for the Tree of Life*, CVPR 2024; imageomics/bioclip, MIT), só na comparação zero-shot. PaliGemma 2 3B mix 224 (Google, Gemma Terms of Use). Opus-MT `opus-mt-ROMANCE-en` (Apache 2.0) e `opus-mt-tc-big-en-pt` (CC BY 4.0), Helsinki-NLP.

**Bibliotecas:** PyTorch, torchvision, OpenCV, NumPy, Pillow, scikit-image, scikit-learn, pandas, Streamlit, Hugging Face Transformers, bitsandbytes, accelerate, PyYAML, pytest.

**Artigos:** Cotrim e Pedrini, *Identificação de Espécies de Árvores por Imagens de Tronco Utilizando Aprendizado de Máquina Profundo*, IC-Unicamp, 2019. Ratajczak et al., *Efficient Bark Recognition in the Wild*, VISAPP 2019. Selvaraju et al., *Grad-CAM*, ICCV 2017. Belfort et al., *Classificação de espécies de árvores em imagens de cenário urbano usando CNNs com Triplet Loss*, SEMISH 2026.

## 12. Uso de inteligência artificial

| Ferramenta | Finalidade | Partes afetadas | Revisão e validação |
|---|---|---|---|
| Claude (Anthropic), via Claude Code | Planejamento, pesquisa de bases, escrita e depuração de código, documentação | Todo o repositório | Cada etapa foi decidida pelo autor (escolha de bases, espécies, divisões, modelos, padrões); código validado por testes automáticos, execução na interface e métricas em conjuntos de teste separados |
| PaliGemma 2 (Google) | Componente opcional da aplicação: descrição e perguntas sobre a imagem | Painel do VLM | Protocolo de 20 casos em `docs/vlm_protocolo.md` |

## 13. Diário de desenvolvimento

Ver `docs/diario.md` (uma entrada por etapa, com os quatro marcos).

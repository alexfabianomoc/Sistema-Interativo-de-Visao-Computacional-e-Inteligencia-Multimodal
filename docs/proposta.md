# Proposta — Marco 1

**Projeto:** Sistema Interativo de Visão Computacional e Inteligência Multimodal
**Autor:** Alex Fabiano Silva (RA 031600) — trabalho individual
**Data:** 20/11/2026

## Problema

Identificar a espécie de uma árvore a partir de uma foto, de forma rápida e em campo, e mostrar ao usuário qual região da imagem justificou a resposta. A saída é uma sugestão com grau de confiança, nunca uma identificação botânica definitiva: quando a confiança fica abaixo de um limiar, o resultado é "inconclusivo" e a imagem vai para revisão manual por um profissional habilitado.

## Usuário

Estudantes, guardas-parque e equipes de arborização urbana que precisam de uma triagem rápida da espécie, e profissionais (biólogos com CRBio, engenheiros florestais ou agrônomos com CREA) que revisam os casos duvidosos.

## Entrada

- Upload de foto (JPG ou PNG).
- Fotos de demonstração em `data_samples/`.
- Câmera pelo navegador.
- Foto de drone: o usuário recorta a árvore; GPS e altitude são lidos do EXIF. Sem GPS, a aplicação pergunta se segue sem coordenadas ou abre campos de latitude e longitude com validação.

## Dois modos de classificação

| Modo | Base | Espécies | Imagens |
|------|------|----------|---------|
| Casca | TRUNK12 | 12 árvores europeias (amieiro, faia, bétula, castanheiro, ginkgo, carpino, castanheiro-da-índia, tília, carvalho, plátano-oriental, pinheiro, abeto) | 393 fotos padronizadas do tronco (3000×4000, mesma câmera, a 20 cm) |
| Folha | GBIF, conjunto "Pl@ntNet observations" | as mesmas 12 espécies do TRUNK12 | até 500 fotos de folha por espécie, 600 px |

**Escolha das bases (20/11):** a base inicialmente prevista para o modo folha, a Pl@ntNet-300K, não tem nenhuma espécie em comum com o TRUNK12 (1.081 espécies, quase todas ervas e flores). Também foi avaliada a Forest Species Database macroscópica da UFPR, descartada por mostrar o corte da madeira e não a casca da árvore em pé. A solução foi o conjunto "Pl@ntNet observations" do GBIF, que marca o órgão de cada foto: assim o modo folha usa as mesmas 12 espécies do modo casca, o que permite o **uso combinado casca + folha** (fusão tardia: produto das probabilidades dos dois modelos). Ginkgo e plátano-oriental têm poucas fotos de folha (cerca de 70 e 100), o que é tratado com pesos por classe no treino e F1 macro na avaliação.
## Saída

- Espécie prevista com top-3 e confiança.
- Mapa de calor Grad-CAM sobreposto à foto, com opacidade ajustável.
- Grade de patches colorida pela espécie prevista (casca) ou contorno e máscara da folha (folha), com legenda.
- Vista comparativa: original, após realce e resultado.
- Imagem final exportável em PNG.

## Técnica

1. **Pré-processamento:** validação de formato e tamanho; recorte da região de interesse; rotação para endireitar o tronco; redimensionamento com proporção (letterbox); CLAHE no canal L do espaço LAB; normalização ImageNet. No modo folha, segmentação por cor (HSV) com abertura e fechamento morfológicos.
2. **Modelo principal:** MobileNetV3-Large pré-treinada no ImageNet, com ajuste fino em cada base; ResNet-18 treinada também para comparação. Casca: patches de 224 px com voto médio. Folha: imagem inteira + 5 recortes com média das probabilidades.
3. **Linhas de base clássicas:** LBP + histograma de cor + SVM (casca); momentos de Hu, razão de aspecto, solidez + histograma de cor + SVM (folha).
4. **Interpretação multimodal (opcional da rubrica):** PaliGemma 2 3B mix 224 em 4 bits na GPU, local, para descrever a imagem, verificar se há tronco ou folha e responder perguntas curtas. As respostas são exibidas como interpretação, nunca como certeza.

## Avaliação

Acurácia, acurácia top-3, F1 macro e matriz de confusão nos conjuntos de teste; tempo por imagem (ms) e tempo de treino; MobileNetV3 x ResNet-18 x linha de base clássica; concordância entre modelo e profissional na revisão manual; galeria de acertos e erros; teste com fotos próprias.

## Ambiente

Python 3.11, PyTorch 2.6 + CUDA 12.4 (testado na RTX 4050 Laptop 6 GB em 20/11), OpenCV, scikit-image, scikit-learn, Streamlit. A aplicação cai para CPU quando não há GPU.

## Licenças das bases

- **GBIF / Pl@ntNet observations:** conjunto publicado em CC BY 4.0; cada foto tem licença própria (em geral CC BY-SA) e autor. Autor, licença e link de cada foto usada ficam em `data/folhas/manifesto.csv` e são creditados no README.
- **TRUNK12:** Švab (2014), ViCoS Lab, Universidade de Ljubljana. A página oficial não traz licença explícita, só pede citação: *Computer-vision-based tree trunk recognition*, Matic Švab, trabalho de bacharelado (orientador Matej Kristan), FRI, Univerza v Ljubljani, 2014. Por cautela, as fotos não são redistribuídas. Nenhuma base é versionada no repositório.

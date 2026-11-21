# Sistema Interativo de Visão Computacional e Inteligência Multimodal

Identificação de espécies de árvores pela foto da casca (base TRUNK12) e pela foto da folha ou da planta (amostra da Pl@ntNet-300K), com interface em Streamlit, mapa de calor sobre a imagem e revisão manual por profissional habilitado.

**Autor:** Alex Fabiano Silva — RA 031600
**Disciplina:** Computação Gráfica, Realidade Virtual e Metaverso (2026-2)
**Professor:** Pedro Cândido

> Este README é um esqueleto (Marco 1). Instalação, execução, parâmetros, arquitetura, limitações, fontes, créditos e declaração de uso de IA serão preenchidos até a entrega de 23/11/2026.

## Proposta

Ver [docs/proposta.md](docs/proposta.md).

## Estrutura

```
src/acquisition     entrada: upload, amostras, câmera, drone
src/preprocessing   ROI, redimensionamento com proporção, CLAHE, normalização
src/models          CNNs (MobileNetV3-Large, ResNet-18) e linhas de base clássicas
src/visualization   Grad-CAM, grade de patches, legendas, contornos
src/metrics         acurácia, F1 macro, matriz de confusão, tempo por imagem
scripts/            preparar_dados.py, baixar_pesos.py
tests/              casos de falha e avaliação
docs/               proposta, diário, métricas, prompts do VLM
data_samples/       fotos de demonstração com licença clara
revisoes/           revisões manuais (CSV, fora do Git)
```

Bases de dados e pesos dos modelos não são versionados.

## Ambiente

- Python 3.11
- GPU NVIDIA opcional (testado em RTX 4050 6 GB com CUDA 12.4); sem GPU a aplicação roda em CPU.

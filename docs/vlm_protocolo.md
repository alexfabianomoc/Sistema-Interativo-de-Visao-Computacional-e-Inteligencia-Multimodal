# Protocolo de validação do VLM (seção 9 do enunciado)

## Modelo e execução

| Item | Valor |
|---|---|
| Modelo | `google/paligemma2-3b-mix-224` (PaliGemma 2, Google; Gemma Terms of Use) |
| Execução | Local, na GPU, quantizado em 4 bits (NF4, bitsandbytes); sem internet e sem custo por chamada |
| Requisitos medidos | 2,5 GB de VRAM (RTX 4050 6 GB), carga em ~17 s; sem GPU roda em CPU (32 GB de RAM), bem mais lento |
| Imagem enviada | Só o recorte (ROI) já rotacionado, convertido para RGB; o processador redimensiona para 224×224. Nada de EXIF, GPS ou nome do arquivo |
| Tradução | Opus-MT local: pergunta pt → en antes do modelo, resposta en → pt depois; a interface mostra também o original |
| Latência medida (20 casos) | Mediana 247 ms; máximo 1.142 ms (descrições longas); a primeira chamada inclui aquecimento (~940 ms) |

## Prompts principais

A variante *mix* responde a prefixos de tarefa, não a conversa livre. Cada chamada é uma rodada única imagem + texto.

| Uso na interface | Prompt | Por quê |
|---|---|---|
| Descrição | `caption en` | Legenda curta em inglês; `caption pt` também responde em inglês (testado em 20/11) |
| Verificação da entrada (casca) | `answer en Is there a tree trunk or tree bark in this image?` | Pergunta fechada: a resposta sim/não é comparável e serve de alerta antes de confiar na CNN |
| Verificação da entrada (folha) | `answer en Is there a leaf or a plant in this image?` | Idem |
| Descrição da folha | `answer en Describe the shape, edge and color of the leaf.` | Tentar características botânicas úteis ao profissional |
| Pergunta livre | `answer en <pergunta traduzida>` | Uma pergunta por vez, curta e direta |

## Casos de teste (20) e classificação

Fechadas: classificação automática pela resposta esperada. Abertas: classificadas pelo autor após ver a imagem. Script: `scripts/protocolo_vlm.py`; respostas completas em `docs/vlm_casos.json`.

| Id | Imagem | Pergunta | Esperado | Resposta | Classe |
|---|---|---|---|---|---|
| F01 | casca/oak | Há tronco? | sim | yes | correta |
| F02 | casca/beech | Há tronco? | sim | yes | correta |
| F03 | casca/pine | Há tronco? | sim | yes | correta |
| F04 | casca/birch | Há tronco? | sim | yes | correta |
| F05 | folha/oak | Há tronco? | ~~não~~ sim* | yes | correta* |
| F06 | folha/ginkgo | Há tronco? | ~~não~~ sim* | yes | correta* |
| F07 | folha/linden | Há folha? | sim | yes | correta |
| F08 | folha/hornbeam | Há folha? | sim | yes | correta |
| F09 | folha/spruce | Há folha? | sim | yes | correta |
| F10 | sintética (formas e texto) | Há folha? | não | no | correta |
| F11 | sintética | Há tronco? | não | no | correta |
| F12 | casca/spruce | É uma foto de folha? | não | no | correta |
| A01 | casca/oak | Descrição | — | "the bark of a tree" | correta |
| A02 | folha/chestnut | Descrição | — | "a plant" | incompleta (não cita folhas serrilhadas nem as flores em amento) |
| A03 | folha/horse_chestnut | Forma, borda e cor da folha | — | "green" | incompleta (só a cor) |
| A04 | folha/ginkgo | Forma, borda e cor da folha | — | "green" | incompleta (não cita a forma de leque, a mais típica do ginkgo) |
| A05 | casca/birch | Cor da casca | — | "white" | correta |
| A06 | folha/alder | Quantas folhas? | — | "2" | incorreta (há uma folha em destaque e várias ao fundo) |
| A07 | casca/linden | Que espécie é? | — | "oak" | incorreta (é tília; o VLM afirma uma espécie sem base) |
| A08 | sintética | Descrição | — | "a blue square and a brown circle with the words teste 123" | correta |

\* F05 e F06: a resposta esperada foi definida antes de olhar as fotos, como "não há tronco" por serem fotos de folha. Ao conferir, as duas mostram troncos (o ginkgo cresce colado a um tronco; o carvalho tem troncos ao fundo). O VLM estava certo e o gabarito foi corrigido; o caso fica registrado como lição sobre gabaritos.

**Resumo:** 15 corretas, 3 incompletas, 2 incorretas, 0 alucinadas. Fechadas: 12/12. Abertas: 3 corretas, 3 incompletas, 2 incorretas.

## Alucinações e limitações observadas

- Nenhuma alucinação clara nos 20 casos (a imagem sintética foi descrita fielmente, sem inventar planta).
- O VLM **não identifica espécies de forma confiável** (A07: chamou tília de carvalho). Por isso ele não classifica: a espécie vem da CNN, e o painel avisa que as respostas são interpretações.
- Respostas a perguntas descritivas são curtas e pobres (A03, A04): a variante *mix* foi treinada para legendas e respostas breves.
- Contagem é fraca (A06).
- Só descreve em inglês; a tradução local cobre a interface, mas pode perder nuances.
- Uso adequado no projeto: **verificação da entrada** (perguntas fechadas, 12/12 corretas) e descrição geral.

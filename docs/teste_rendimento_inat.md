# Teste de rendimento: espécies brasileiras no iNaturalist (21/11/2026)

Amostra: até 50 observações por táxon no Brasil (`place_id=6878`), grau de pesquisa, fotos com licença Creative Commons, ordenadas por votos; 1.985 fotos no total. Cada foto foi classificada pelo PaliGemma 2 com a pergunta *"Is this a photo of a tree trunk, a leaf, a flower, a fruit, or a whole tree?"* (tronco = resposta com trunk/bark; folha = leaf; resto = outras). Script: `scripts/teste_rendimento_inat.py`.

**Conferência à mão** (24 fotos sorteadas de cada grupo): das marcadas como tronco, cerca de 14 em 24 (58%) mostram o tronco de forma útil; o resto são galhos finos, árvore inteira ou folhas. Das marcadas como folha, cerca de 16 em 24 (67%) mostram folhagem em destaque. Nas "outras" não apareceu tronco perdido (são flores, frutos e árvores inteiras). As estimativas abaixo já descontam essa precisão.

| Espécie | Táxon | Obs. no Brasil | Fotos na amostra | Tronco | Folha | Outras | Troncos úteis (até 300 obs.) | Troncos úteis (todas as obs.) | Folhas úteis (até 300 obs.) |
|---|---|---|---|---|---|---|---|---|---|
| pau-brasil | *Paubrasilia echinata* | 517 | 135 | 41 | 43 | 51 | 144 | 247 | 172 |
| ipês (gênero) | *Handroanthus* | 932 | 174 | 14 | 48 | 112 | 49 | 152 | 192 |
| chapéu-de-sol | *Terminalia catappa* | 346 | 110 | 11 | 65 | 34 | 38 | 44 | 260 |
| sibipiruna | *Cenostigma pluviosum* | 237 | 114 | 9 | 24 | 81 | 25 | 25 | 76 |
| aroeira-pimenteira | *Schinus terebinthifolia* | 1387 | 147 | 7 | 78 | 62 | 25 | 113 | 312 |
| jambolão | *Syzygium cumini* | 172 | 117 | 12 | 43 | 62 | 24 | 24 | 99 |
| pau-ferro | *Libidibia ferrea* | 45 | 120 | 33 | 27 | 60 | 19 | 19 | 18 |
| mulungu | *Erythrina speciosa* | 636 | 90 | 5 | 12 | 73 | 18 | 37 | 48 |
| quaresmeira | *Pleroma granulosum* | 300 | 97 | 5 | 11 | 81 | 18 | 18 | 44 |
| munguba | *Pachira aquatica* | 281 | 92 | 5 | 32 | 55 | 16 | 16 | 120 |
| canafístula | *Peltophorum dubium* | 161 | 144 | 7 | 47 | 90 | 13 | 13 | 101 |
| oiti | *Moquilea tomentosa* | 91 | 99 | 11 | 39 | 49 | 12 | 12 | 47 |
| mangueira | *Mangifera indica* | 402 | 79 | 3 | 32 | 44 | 10 | 14 | 128 |
| pata-de-vaca | *Bauhinia variegata* | 201 | 88 | 3 | 19 | 66 | 7 | 7 | 51 |
| espatódea | *Spathodea campanulata* | 566 | 80 | 1 | 8 | 71 | 4 | 7 | 32 |
| tipuana | *Tipuana tipu* | 85 | 115 | 2 | 44 | 69 | 2 | 2 | 50 |
| flamboyant | *Delonix regia* | 368 | 88 | 0 | 13 | 75 | 0 | 0 | 52 |
| resedá | *Lagerstroemia indica* | 196 | 96 | 0 | 9 | 87 | 0 | 0 | 24 |

# Histórico de treinamentos

Registro dos resultados e das conclusões de cada treino completo de `gardner_cnn.ipynb`.

A execução de 31/10/2026 não entra na numeração: ela foi feita com os bugs de brilho e de altura/largura, e o notebook não guardou métricas dela.

## Treino 1 (02/11/2026)

**Resultado:** apenas EXP é útil. ICM e TE acertam menos do que prever sempre a classe mais frequente.

### Configuração

| Item | Valor |
|---|---|
| Código | commit `830b873` |
| Entrada | 168×224 (altura×largura), RGB, normalizada para 0–1 |
| Dados | 1.743 imagens de treino e 301 de validação (silver); 300 de teste (gold) |
| Batch size | 32 |
| Otimizador | Adam, `learning_rate=0.001` inicial, com `ReduceLROnPlateau` (fator 0,5, patience 4) |
| Loss | `sparse_categorical_crossentropy` em cada saída, com `loss_weights` 1, 1, 1 |
| Pesos de classe | `balanced`, sem limite, só no treino; o maior é cerca de 36 (ICM=2) |
| Parada | Máximo de 60 epochs; `EarlyStopping` com patience 8 sobre a `val_loss` sem pesos |
| Augmentação | Flip horizontal e vertical, rotação de ±15°, brilho de ±10% |

### Execução

- **Duração:** cerca de 49 minutos em CPU, incluindo as demais células do notebook.
- **Epochs:** parou por early stopping na epoch 33; os pesos restaurados são os da epoch 25.
- **Melhor `val_loss`:** 2,749 (epoch 25).
- **Variação da `val_loss`:** de 2,749 a 12,037 ao longo do treino.
- **Loss de treino:** caiu de 5,324 na epoch 1 para 2,799 na epoch 33.

### Validação (silver, epoch 25)

A linha de base é a accuracy de prever sempre a classe mais frequente.

| Alvo | Accuracy | Linha de base | Loss |
|---|---|---|---|
| EXP | 0,691 | 0,558 | 0,833 |
| ICM | 0,492 | 0,671 | 1,097 |
| TE | 0,638 | 0,555 | 0,822 |

A validação inclui as imagens de classe 3 ("não determinado"), que o gold exclui em ICM e TE. Por isso, esses valores não são diretamente comparáveis aos do teste.

### Teste (gold)

| Alvo | Imagens | Accuracy | Linha de base | Kappa quadrático |
|---|---|---|---|---|
| EXP | 298 | 0,664 | 0,513 | 0,701 |
| ICM | 238 | 0,382 | 0,643 | 0,094 |
| TE | 234 | 0,470 | 0,521 | 0,083 |

#### Por subconjunto de embriões

Embriões inéditos são os que não têm nenhuma imagem no CSV silver.

| Alvo | Com imagens no silver: n | Accuracy | Kappa | Inéditos: n | Accuracy | Kappa |
|---|---|---|---|---|---|---|
| EXP | 240 | 0,667 | 0,680 | 58 | 0,655 | 0,769 |
| ICM | 195 | 0,390 | 0,102 | 43 | 0,349 | 0,037 |
| TE | 193 | 0,482 | 0,110 | 41 | 0,415 | -0,053 |

#### Recall por classe

| Classe | EXP | ICM | TE |
|---|---|---|---|
| 0 | 0,57 (23) | 0,50 (153) | 0,85 (122) |
| 1 | 0,23 (31) | 0,12 (68) | 0,06 (82) |
| 2 | 0,52 (86) | 0,35 (17) | 0,03 (30) |
| 3 | 0,87 (153) | sem imagens | sem imagens |
| 4 | 0,00 (5) | não existe | não existe |

O número entre parênteses é a quantidade de imagens da classe.

### Conclusões

1. **EXP aprendeu um sinal real.** A accuracy supera a linha de base, e o Kappa de 0,701 indica boa concordância com o consenso. Mesmo assim, a classe 4 não foi reconhecida (0 de 5 imagens).
2. **ICM e TE não são úteis neste modelo.** A accuracy fica abaixo da linha de base, e o Kappa perto de 0 indica concordância pouco acima do acaso.
3. **ICM prevê demais a classe rara.** Pela precisão e pelo recall arredondados do relatório, a classe 2 é prevista em cerca de 85 das 238 imagens, embora só 17 pertençam a ela. Isso é compatível com o peso de classe de cerca de 36, mas não prova a causa.
4. **O treino é instável.** A `val_loss` varia muito entre epochs: de 2,749 na epoch 25 foi para 3,655 na epoch 26. Por isso, a escolha da melhor epoch pode ter um componente de acaso.
5. **A loss total prejudicou a escolha para ICM.** Na epoch 25, a accuracy de validação de ICM foi 0,492, mas chegou a 0,711 na epoch 27.
6. **Não é overfitting clássico.** A accuracy de treino ficou em torno de 0,5 a 0,57, e a loss de treino continuou caindo devagar. O modelo tem dificuldade de aprender, em vez de memorizar.
7. **Embriões compartilhados não parecem inflar os resultados.** A accuracy é parecida entre os dois subconjuntos, por exemplo 0,667 e 0,655 em EXP. O subconjunto inédito é pequeno, então isso não é conclusivo.

### Hipóteses para o próximo treino

Ainda não testadas:

- Limitar os pesos de classe a 10, para reduzir a previsão excessiva de classes raras.
- Reduzir a taxa de aprendizado para 0,0003, para diminuir a instabilidade.
- Adicionar `CSVLogger`, Kappa de validação por alvo e accuracy de validação só nas imagens avaliáveis, para comparar melhor com o gold.

A escolha de configurações deve usar apenas a validação, para que o gold continue sendo um teste independente.

## Treino 2 (02/11/2026)

**Resultado:** treino sem utilidade. O early stopping restaurou a epoch 1, e o modelo prevê uma única classe em cada alvo. Este treino não mostra se as mudanças ajudam, porque parou cedo demais.

### Mudanças em relação ao Treino 1

| Item | Treino 1 | Treino 2 |
|---|---|---|
| Pesos de classe | Sem limite (máximo de cerca de 36) | Limitados a 10 |
| `learning_rate` inicial | 0,001 | 0,0003 |
| Registro por epoch | Nenhum | `CSVLogger` em `models/training_log.csv` |
| Métricas de validação | Accuracy por alvo | Também accuracy, linha de base e Kappa só nas imagens avaliáveis |

Código: commit `7f7e378`. O restante da configuração é igual ao do Treino 1. O registro por epoch está em `models/runs/treino2/training_log.csv`.

### Execução

- **Duração:** cerca de 13 minutos em CPU, incluindo as demais células do notebook.
- **Epochs:** parou por early stopping na epoch 9; os pesos restaurados são os da epoch 1.
- **Melhor `val_loss`:** 4,451 (epoch 1).
- **Variação da `val_loss`:** de 4,451 a 7,548 ao longo do treino.
- **Loss de treino:** caiu de 4,581 na epoch 1 para 2,887 na epoch 9.

### Validação (silver, epoch 1, imagens avaliáveis)

| Alvo | Imagens | Accuracy | Linha de base | Kappa quadrático |
|---|---|---|---|---|
| EXP | 301 | 0,558 | 0,558 | 0,000 |
| ICM | 251 | 0,016 | 0,805 | 0,000 |
| TE | 251 | 0,307 | 0,665 | 0,000 |

### Teste (gold)

| Alvo | Imagens | Accuracy | Kappa quadrático | Classe prevista em todas as imagens |
|---|---|---|---|---|
| EXP | 298 | 0,513 | 0,000 | 3 |
| ICM | 238 | 0,071 | 0,000 | 2 |
| TE | 234 | 0,350 | 0,000 | 1 |

A classe prevista foi deduzida do relatório por classe, com recall de 1,00 nessa classe e 0,00 nas demais.

### Conclusões

1. **O modelo restaurado não aprendeu.** Ele vem da epoch 1 e prevê sempre a mesma classe em cada alvo, por isso o Kappa é 0.
2. **As mudanças não foram avaliadas de fato.** O treino parou 8 epochs depois da primeira, enquanto o modelo ainda aprendia. O Treino 1 só atingiu sua melhor `val_loss` na epoch 25.
3. **O treino avançou mais rápido que no Treino 1.** Na epoch 9, a accuracy de treino foi de 0,497, 0,590 e 0,502 (EXP, ICM, TE), contra 0,422, 0,425 e 0,450 no Treino 1. A medida usa pesos e augmentação, então serve apenas como indicação.
4. **Treino e validação se comportam de formas muito diferentes.** Até a epoch 5, a accuracy de treino subiu para cerca de 0,45 a 0,53, enquanto a de validação ficou fixa em poucos valores, como 0,090 em EXP e 0,023 em TE.

### Hipóteses para o próximo treino

Ainda não testadas:

- **BatchNorm:** o momentum padrão de 0,99 atualiza devagar as estatísticas usadas na validação, com cerca de 55 lotes por epoch. Isso pode explicar a diferença entre treino e validação. Testar `momentum=0.9`.
- **Early stopping:** aumentar a patience ou impedir a parada nas primeiras epochs, com `start_from_epoch`.

## Treino 3 (02/11/2026)

**Resultado:** o melhor Kappa nos três alvos até agora, mas ICM e TE continuam fracos. EXP perdeu accuracy porque passou a prever demais a classe 4.

### Mudanças em relação ao Treino 2

| Item | Treino 2 | Treino 3 |
|---|---|---|
| `momentum` do BatchNorm | 0,99 (padrão) | 0,9 |
| Early stopping | Considera todas as epochs | `start_from_epoch=15` |
| Modelo salvo ao final | Melhor checkpoint | Mesmo modelo avaliado no notebook |

Os pesos de classe limitados a 10 e o `learning_rate` inicial de 0,0003 continuam iguais.

Código: commit `3219941`. O registro por epoch está em `models/runs/treino3/training_log.csv`.

### Execução

- **Duração:** cerca de 45 minutos em CPU, incluindo as demais células do notebook.
- **Epochs:** parou por early stopping na epoch 31; os pesos restaurados são os da epoch 23.
- **Melhor `val_loss`:** 2,584 (epoch 23), a menor entre os três treinos.
- **Variação da `val_loss`:** de 2,584 a 6,379. Da epoch 20 em diante, com a taxa de aprendizado já reduzida, ficou entre 2,584 e 2,916.
- **Loss de treino:** caiu de 4,594 na epoch 1 para 2,287 na epoch 31.

### Validação (silver, epoch 23, imagens avaliáveis)

| Alvo | Imagens | Accuracy | Linha de base | Kappa quadrático |
|---|---|---|---|---|
| EXP | 301 | 0,442 | 0,558 | 0,778 |
| ICM | 251 | 0,649 | 0,805 | 0,074 |
| TE | 251 | 0,586 | 0,665 | 0,175 |

### Teste (gold)

| Alvo | Imagens | Accuracy | Linha de base | Kappa quadrático |
|---|---|---|---|---|
| EXP | 298 | 0,436 | 0,513 | 0,756 |
| ICM | 238 | 0,550 | 0,643 | 0,143 |
| TE | 234 | 0,440 | 0,521 | 0,146 |

#### Por subconjunto de embriões

| Alvo | Com imagens no silver: n | Accuracy | Kappa | Inéditos: n | Accuracy | Kappa |
|---|---|---|---|---|---|---|
| EXP | 240 | 0,438 | 0,752 | 58 | 0,431 | 0,768 |
| ICM | 195 | 0,569 | 0,152 | 43 | 0,465 | 0,094 |
| TE | 193 | 0,456 | 0,183 | 41 | 0,366 | -0,002 |

#### Recall por classe

| Classe | EXP | ICM | TE |
|---|---|---|---|
| 0 | 0,83 (23) | 0,70 (153) | 0,65 (122) |
| 1 | 0,48 (31) | 0,35 (68) | 0,18 (82) |
| 2 | 0,58 (86) | 0,00 (17) | 0,30 (30) |
| 3 | 0,27 (153) | sem imagens | sem imagens |
| 4 | 0,80 (5) | não existe | não existe |

O número entre parênteses é a quantidade de imagens da classe.

### Comparação entre os treinos (gold)

| Treino | EXP: accuracy | EXP: Kappa | ICM: accuracy | ICM: Kappa | TE: accuracy | TE: Kappa |
|---|---|---|---|---|---|---|
| Linha de base | 0,513 | 0 | 0,643 | 0 | 0,521 | 0 |
| Treino 1 | 0,664 | 0,701 | 0,382 | 0,094 | 0,470 | 0,083 |
| Treino 2 | 0,513 | 0,000 | 0,071 | 0,000 | 0,350 | 0,000 |
| Treino 3 | 0,436 | 0,756 | 0,550 | 0,143 | 0,440 | 0,146 |

### Conclusões

1. **O Kappa melhorou nos três alvos em relação ao Treino 1.** A melhora em ICM e TE é pequena, e os dois seguem com concordância fraca.
2. **As mudanças resolveram o colapso do Treino 2.** O modelo deixou de prever uma única classe. Como o `momentum` do BatchNorm e o `start_from_epoch` mudaram juntos, não dá para separar o efeito de cada um.
3. **A hipótese do BatchNorm ganhou força, mas não foi provada.** A accuracy de validação parou de ficar travada nos mesmos valores já nas primeiras epochs. Ela ainda oscilou bastante até a taxa de aprendizado cair.
4. **EXP prevê demais a classe 4.** Pela precisão e pelo recall arredondados, a classe 4 é prevista em mais de 100 das 298 imagens, embora só 5 pertençam a ela. Por isso, a accuracy caiu para 0,436 e a recall da classe 3 ficou em 0,27. O Kappa alto mostra que muitos desses erros ficam perto da classe correta.
5. **Nenhum alvo supera a linha de base em accuracy.** A escolha entre os modelos depende da métrica: o Treino 1 é melhor em accuracy de EXP, e o Treino 3 é melhor em todos os Kappas e na accuracy de ICM.
6. **ICM=2 continua sem acerto.** A recall foi 0,00 nas 17 imagens, mesmo com o peso de classe de 10.
7. **Embriões inéditos ficam um pouco abaixo em ICM e TE.** O subconjunto é pequeno, então isso não é conclusivo.

## Treino 4 (02/11/2026)

**Resultado:** o melhor modelo até agora, em Kappa nos três alvos e em accuracy. EXP tem boa concordância com o gold, e TE melhorou bastante. ICM segue abaixo da linha de base em accuracy.

### Mudanças em relação ao Treino 3

| Item | Treino 3 | Treino 4 |
|---|---|---|
| Backbone | CNN do zero, com 7 convoluções | EfficientNetB0 pré-treinada no ImageNet e congelada |
| Pesos treináveis | Todos | 331.277 de cerca de 4,4 milhões |
| Entrada do backbone | Pixels de 0 a 1 | Reescalada para 0 a 255 |

Pesos de classe limitados a 10, `learning_rate` inicial de 0,0003, `start_from_epoch=15`, separação por embrião e augmentação continuam iguais. O registro por epoch está em `models/runs/treino4/training_log.csv`.

### Execução

- **Duração:** cerca de 17 minutos em CPU, incluindo as demais células do notebook.
- **Epochs:** parou por early stopping na epoch 46; os pesos restaurados são os da epoch 38.
- **Melhor `val_loss`:** 2,285 (epoch 38), a menor entre os quatro treinos.
- **Variação da `val_loss`:** de 2,285 a 3,192, bem mais estável que nos treinos anteriores.
- **Loss de treino:** caiu de 3,592 na epoch 1 para 1,457 na epoch 46.

### Validação (silver, epoch 38, imagens avaliáveis)

| Alvo | Imagens | Accuracy | Linha de base | Kappa quadrático |
|---|---|---|---|---|
| EXP | 301 | 0,661 | 0,558 | 0,812 |
| ICM | 251 | 0,665 | 0,805 | 0,152 |
| TE | 251 | 0,633 | 0,665 | 0,250 |

### Teste (gold)

| Alvo | Imagens | Accuracy | Linha de base | Kappa quadrático |
|---|---|---|---|---|
| EXP | 298 | 0,765 | 0,513 | 0,864 |
| ICM | 238 | 0,609 | 0,643 | 0,251 |
| TE | 234 | 0,585 | 0,521 | 0,429 |

#### Por subconjunto de embriões

| Alvo | Com imagens no silver: n | Accuracy | Kappa | Inéditos: n | Accuracy | Kappa |
|---|---|---|---|---|---|---|
| EXP | 240 | 0,796 | 0,888 | 58 | 0,638 | 0,790 |
| ICM | 195 | 0,621 | 0,284 | 43 | 0,558 | 0,114 |
| TE | 193 | 0,591 | 0,465 | 41 | 0,561 | 0,290 |

#### Recall por classe

| Classe | EXP | ICM | TE |
|---|---|---|---|
| 0 | 0,78 (23) | 0,69 (153) | 0,73 (122) |
| 1 | 0,55 (31) | 0,59 (68) | 0,46 (82) |
| 2 | 0,76 (86) | 0,00 (17) | 0,33 (30) |
| 3 | 0,81 (153) | sem imagens | sem imagens |
| 4 | 0,80 (5) | não existe | não existe |

O número entre parênteses é a quantidade de imagens da classe.

### Comparação entre os treinos (gold)

| Treino | Backbone | EXP: accuracy | EXP: Kappa | ICM: accuracy | ICM: Kappa | TE: accuracy | TE: Kappa |
|---|---|---|---|---|---|---|---|
| Linha de base | Nenhum | 0,513 | 0 | 0,643 | 0 | 0,521 | 0 |
| Treino 1 | Do zero | 0,664 | 0,701 | 0,382 | 0,094 | 0,470 | 0,083 |
| Treino 2 | Do zero | 0,513 | 0,000 | 0,071 | 0,000 | 0,350 | 0,000 |
| Treino 3 | Do zero | 0,436 | 0,756 | 0,550 | 0,143 | 0,440 | 0,146 |
| Treino 4 | EfficientNetB0 | 0,765 | 0,864 | 0,609 | 0,251 | 0,585 | 0,429 |

### Conclusões

1. **O pré-treino no ImageNet ajudou neste problema.** Só o backbone mudou em relação ao Treino 3, e o Kappa subiu em todos os alvos: de 0,756 para 0,864 em EXP, de 0,143 para 0,251 em ICM e de 0,146 para 0,429 em TE. Uma única execução não mede a variação entre treinos, então os valores exatos têm incerteza.
2. **O treino ficou mais estável e mais rápido.** A `val_loss` variou muito menos, e a execução levou cerca de 17 minutos, contra 45 do Treino 3.
3. **EXP corrigiu boa parte da previsão excessiva da classe 4.** Pela precisão e pelo recall arredondados, a classe 4 é prevista em cerca de 22 imagens, contra mais de 100 no Treino 3. Só 5 pertencem a ela, então o excesso ainda existe.
4. **ICM é o ponto mais fraco.** A accuracy fica abaixo da linha de base, o Kappa é baixo e ICM=2 segue sem acerto nas 17 imagens.
5. **Os embriões compartilhados podem estar inflando os resultados.** Neste treino, os inéditos ficam abaixo em todos os alvos, por exemplo com accuracy de 0,638 contra 0,796 em EXP. Isso revê a leitura do Treino 1. O subconjunto é pequeno, mas os números dos inéditos são a estimativa mais conservadora.
6. **O gold já foi consultado em quatro treinos.** Ele influenciou a escolha do próximo passo, então deixou de ser totalmente independente. Os próximos ajustes devem ser escolhidos pela validação.

### Hipóteses para o próximo treino

Ainda não testadas:

- Treinar ICM e TE só com imagens avaliáveis, zerando o peso da classe 3 nessas saídas.
- Fazer ajuste fino das últimas camadas da EfficientNetB0, com taxa de aprendizado baixa.
- Aumentar a resolução da entrada, para preservar detalhes de ICM e TE.

## Treino 5 (02/11/2026)

**Resultado:** o melhor modelo até agora. ICM melhorou e passou da linha de base em accuracy. EXP e TE ficaram praticamente iguais ao Treino 4. O treino parou cedo, e ICM=2 segue sem acerto.

### Mudanças em relação ao Treino 4

| Item | Treino 4 | Treino 5 |
|---|---|---|
| Classe 3 de ICM/TE no treino | Treinada como uma classe | Peso 0 |
| Pesos de classe de ICM/TE | Calculados com a classe 3 | Calculados só com imagens avaliáveis |
| `val_loss` | Inclui a classe 3 de ICM/TE | Ignora a classe 3 de ICM/TE |

O restante da configuração é igual. O registro por epoch está em `models/runs/treino5/training_log.csv`.

A `val_loss` e a accuracy de validação exibidas durante o treino não são comparáveis às dos treinos anteriores, porque a classe 3 mudou de papel. A comparação abaixo usa as métricas de imagens avaliáveis e o gold.

### Execução

- **Epochs:** parou por early stopping na epoch 24; os pesos restaurados são os da epoch 16.
- **Melhor `val_loss`:** 1,996 (epoch 16).
- **Loss de treino:** caiu de 2,938 na epoch 1 para 1,460 na epoch 24.

### Validação (silver, epoch 16, imagens avaliáveis)

| Alvo | Imagens | Accuracy | Linha de base | Kappa quadrático | Kappa no Treino 4 |
|---|---|---|---|---|---|
| EXP | 301 | 0,618 | 0,558 | 0,799 | 0,812 |
| ICM | 251 | 0,705 | 0,805 | 0,321 | 0,152 |
| TE | 251 | 0,625 | 0,665 | 0,285 | 0,250 |

### Teste (gold)

| Alvo | Imagens | Accuracy | Linha de base | Kappa quadrático |
|---|---|---|---|---|
| EXP | 298 | 0,755 | 0,513 | 0,857 |
| ICM | 238 | 0,660 | 0,643 | 0,326 |
| TE | 234 | 0,590 | 0,521 | 0,427 |

#### Por subconjunto de embriões

| Alvo | Com imagens no silver: n | Accuracy | Kappa | Inéditos: n | Accuracy | Kappa |
|---|---|---|---|---|---|---|
| EXP | 240 | 0,771 | 0,879 | 58 | 0,690 | 0,787 |
| ICM | 195 | 0,677 | 0,321 | 43 | 0,581 | 0,347 |
| TE | 193 | 0,580 | 0,447 | 41 | 0,634 | 0,317 |

#### Recall por classe

| Classe | EXP | ICM | TE |
|---|---|---|---|
| 0 | 0,74 (23) | 0,75 (153) | 0,64 (122) |
| 1 | 0,68 (31) | 0,63 (68) | 0,67 (82) |
| 2 | 0,67 (86) | 0,00 (17) | 0,17 (30) |
| 3 | 0,82 (153) | sem imagens | sem imagens |
| 4 | 0,80 (5) | não existe | não existe |

O número entre parênteses é a quantidade de imagens da classe.

### Comparação entre os treinos (gold)

| Treino | Backbone | EXP: accuracy | EXP: Kappa | ICM: accuracy | ICM: Kappa | TE: accuracy | TE: Kappa |
|---|---|---|---|---|---|---|---|
| Linha de base | Nenhum | 0,513 | 0 | 0,643 | 0 | 0,521 | 0 |
| Treino 1 | Do zero | 0,664 | 0,701 | 0,382 | 0,094 | 0,470 | 0,083 |
| Treino 2 | Do zero | 0,513 | 0,000 | 0,071 | 0,000 | 0,350 | 0,000 |
| Treino 3 | Do zero | 0,436 | 0,756 | 0,550 | 0,143 | 0,440 | 0,146 |
| Treino 4 | EfficientNetB0 | 0,765 | 0,864 | 0,609 | 0,251 | 0,585 | 0,429 |
| Treino 5 | EfficientNetB0 | 0,755 | 0,857 | 0,660 | 0,326 | 0,590 | 0,427 |

### Conclusões

1. **Treinar só com imagens avaliáveis ajudou ICM.** O Kappa subiu de 0,251 para 0,326, e a accuracy de 0,609 para 0,660, acima da linha de base pela primeira vez. A validação aponta na mesma direção, de 0,152 para 0,321. Uma única execução não mede a variação entre treinos.
2. **EXP e TE praticamente não mudaram.** As diferenças de EXP (0,864 para 0,857) e TE (0,429 para 0,427) são pequenas demais para indicar uma mudança real.
3. **Os embriões inéditos também melhoraram em ICM e TE.** O Kappa passou de 0,114 para 0,347 em ICM e de 0,290 para 0,317 em TE. O subconjunto é pequeno, mas o resultado reforça a mudança.
4. **As classes raras seguem sem solução.** ICM=2 continua sem acerto, e a recall de TE=2 caiu de 0,33 para 0,17. Com 12 e 40 imagens de treino, poucos exemplos limitam o que os pesos de classe conseguem fazer.
5. **O treino pode ter parado cedo demais.** A melhor epoch foi a 16, a primeira que o early stopping considera, e ele parou na 24 com a loss de treino ainda caindo. Um `patience` maior pode mudar esse resultado.

### Hipóteses para o próximo treino

Ainda não testadas:

- Aumentar o `patience` do early stopping.
- Fazer ajuste fino das últimas camadas da EfficientNetB0, com taxa de aprendizado baixa.
- Aumentar a resolução da entrada, para preservar detalhes de ICM e TE.
- Remover ou corrigir a imagem `838_02.png`, que tem dois rótulos de EXP. O efeito de uma única imagem deve ser pequeno.

## Treino 6 (02/11/2026)

**Resultado:** o melhor modelo até agora pela validação, por pouco. O ganho não veio do `patience`: com o valor antigo, o treino chegaria aos mesmos pesos. A diferença para o Treino 5 mostra principalmente a variação entre execuções.

### Mudanças em relação ao Treino 5

| Item | Treino 5 | Treino 6 |
|---|---|---|
| `patience` do `EarlyStopping` | 8 | 15 |

O restante da configuração é igual. O registro por epoch está em `models/runs/treino6/training_log.csv`.

A partir deste treino, a comparação e a escolha do modelo usam a validação. O gold continua registrado, mas foi consultado em todos os treinos, e por isso seus resultados podem estar levemente otimistas.

### Execução

- **Duração:** cerca de 15 minutos em CPU, incluindo as demais células do notebook.
- **Epochs:** parou por early stopping na epoch 45; os pesos restaurados são os da epoch 30.
- **Melhor `val_loss`:** 1,960 (epoch 30), contra 1,996 no Treino 5.
- **Loss de treino:** caiu de 2,978 na epoch 1 para 1,352 na epoch 45.
- **Taxa de aprendizado:** caiu de 0,0003 para 0,000038 até a epoch 40.
- **Efeito do `patience`:** a partir da epoch 16, o maior intervalo entre melhoras da `val_loss` foi de 4 epochs, e nenhuma melhora veio depois da epoch 30. Com `patience=8`, esta execução pararia na epoch 38, com os mesmos pesos da epoch 30.

### Validação (silver, epoch 30, imagens avaliáveis)

| Alvo | Imagens | Accuracy | Linha de base | Kappa quadrático | Kappa no Treino 5 |
|---|---|---|---|---|---|
| EXP | 301 | 0,664 | 0,558 | 0,818 | 0,799 |
| ICM | 251 | 0,701 | 0,805 | 0,350 | 0,321 |
| TE | 251 | 0,633 | 0,665 | 0,295 | 0,285 |

### Teste (gold)

| Alvo | Imagens | Accuracy | Linha de base | Kappa quadrático |
|---|---|---|---|---|
| EXP | 298 | 0,765 | 0,513 | 0,864 |
| ICM | 238 | 0,618 | 0,643 | 0,329 |
| TE | 234 | 0,624 | 0,521 | 0,531 |

#### Por subconjunto de embriões

| Alvo | Com imagens no silver: n | Accuracy | Kappa | Inéditos: n | Accuracy | Kappa |
|---|---|---|---|---|---|---|
| EXP | 240 | 0,783 | 0,884 | 58 | 0,690 | 0,798 |
| ICM | 195 | 0,621 | 0,318 | 43 | 0,605 | 0,383 |
| TE | 193 | 0,622 | 0,540 | 41 | 0,634 | 0,480 |

#### Recall por classe

| Classe | EXP | ICM | TE |
|---|---|---|---|
| 0 | 0,74 (23) | 0,67 (153) | 0,70 (122) |
| 1 | 0,61 (31) | 0,65 (68) | 0,61 (82) |
| 2 | 0,73 (86) | 0,00 (17) | 0,37 (30) |
| 3 | 0,82 (153) | sem imagens | sem imagens |
| 4 | 0,80 (5) | não existe | não existe |

O número entre parênteses é a quantidade de imagens da classe.

### Comparação entre os treinos (gold)

| Treino | Backbone | EXP: accuracy | EXP: Kappa | ICM: accuracy | ICM: Kappa | TE: accuracy | TE: Kappa |
|---|---|---|---|---|---|---|---|
| Linha de base | Nenhum | 0,513 | 0 | 0,643 | 0 | 0,521 | 0 |
| Treino 1 | Do zero | 0,664 | 0,701 | 0,382 | 0,094 | 0,470 | 0,083 |
| Treino 2 | Do zero | 0,513 | 0,000 | 0,071 | 0,000 | 0,350 | 0,000 |
| Treino 3 | Do zero | 0,436 | 0,756 | 0,550 | 0,143 | 0,440 | 0,146 |
| Treino 4 | EfficientNetB0 | 0,765 | 0,864 | 0,609 | 0,251 | 0,585 | 0,429 |
| Treino 5 | EfficientNetB0 | 0,755 | 0,857 | 0,660 | 0,326 | 0,590 | 0,427 |
| Treino 6 | EfficientNetB0 | 0,765 | 0,864 | 0,618 | 0,329 | 0,624 | 0,531 |

### Conclusões

1. **O `patience` não explica a melhora.** Esta execução nunca precisou de mais de 4 epochs para melhorar e terminaria com os mesmos pesos usando `patience=8`. O Treino 5 parou na epoch 24 porque sua execução ficou 8 epochs sem melhorar, o que aqui não aconteceu.
2. **A variação entre execuções é grande.** A semente é a mesma, mas as curvas de `val_loss` já diferem desde a epoch 1, até 0,195 nas primeiras 24 epochs. Com configuração praticamente igual, o Kappa de TE no gold foi de 0,427 para 0,531. Por isso, diferenças desse tamanho entre treinos, inclusive nos anteriores, não bastam para concluir que uma mudança funcionou.
3. **A validação indica melhora pequena.** Os três Kappas subiram, de 0,010 a 0,029. A melhora maior no gold, em TE, não apareceu na validação na mesma proporção (0,285 para 0,295).
4. **ICM e TE seguem fracos.** Na validação, a accuracy de ICM (0,701) e de TE (0,633) continua abaixo da linha de base. No gold, a accuracy de ICM (0,618) também ficou abaixo. Só o Kappa mostra que há alguma ordem nas predições.
5. **As classes raras seguem difíceis.** ICM=2 continua sem acerto. TE=2 subiu para 0,37 de recall, mas com apenas 30 imagens e sem saber quanto disso se repetiria em outra execução.
6. **O treino agora chegou ao limite desta configuração.** A `val_loss` parou de melhorar por 15 epochs, mesmo com a taxa de aprendizado reduzida, enquanto a loss de treino seguia caindo. Com a EfficientNetB0 congelada, é pouco provável que mais epochs ajudem.

### Hipóteses para o próximo treino

Ainda não testadas:

- Repetir a configuração com outras sementes, para medir a variação e comparar as mudanças com ela.
- Fazer ajuste fino das últimas camadas da EfficientNetB0, com taxa de aprendizado baixa.
- Aumentar a resolução da entrada, para preservar detalhes de ICM e TE.

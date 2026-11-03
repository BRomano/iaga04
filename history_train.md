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

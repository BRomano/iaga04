# Especificação: CNN para classificação dos escores de Gardner (blastocistos)

## Contexto

Treinar uma CNN que, a partir de imagens estáticas de blastocistos humanos (dia 5 pós-inseminação), preveja os três critérios do sistema de Gardner usados para selecionar embriões para transferência em FIV:

- **EXP** (grau de expansão/eclosão) — ordinal, 5 classes (0–4)
- **ICM** (qualidade da massa celular interna) — ordinal, graus 0–2, mais a classe 3 (não determinado)
- **TE** (qualidade do trofectoderma) — ordinal, graus 0–2, mais a classe 3 (não determinado)

Dados publicados junto ao artigo de Kromp et al. (2023, *Sci Data*), já presentes no repositório em `data/images/`, `data/Gardner_train_silver.csv` e `data/Gardner_test_gold_onlyGardnerScores.csv`.

## Dados

- **2344 imagens** PNG, todas RGB, mesmo tamanho **512×384**.
- **Treino (silver):** 2044 linhas, `Image;EXP_silver;ICM_silver;TE_silver`, sem valores ausentes. Cada linha é uma imagem. Rótulos anotados por **um único embriologista experiente**.
- **Teste (gold):** 300 linhas, `Image;EXP_gold;ICM_gold;TE_gold;` (coluna extra `Unnamed: 4` vazia, descartada). Rótulos por **consenso de um consórcio internacional de embriologistas clínicos**. Contém valores ausentes e `'ND'` (not determined):
  - EXP_gold: 2 NaN de 300
  - ICM_gold: 54 `'ND'` + 8 NaN de 300
  - TE_gold: 54 `'ND'` + 12 NaN de 300
- **Nomenclatura:** `<id_embriao>_<NN>.png` — o mesmo embrião tem várias imagens (ex.: `0005_01.png` … `0005_08.png`). Split treino/validação deve ser feito por **ID de embrião** (group split), nunca por imagem, para não vazar informação do mesmo embrião entre conjuntos.
- **Desbalanceamento de classes** (treino silver): EXP classe 4 tem só 143 amostras vs. 1081 da classe 3; ICM classe 2 tem só 16 amostras; TE classe 2 tem só 47.
- **Classe 3 de ICM/TE:** no silver, marca exatamente as 391 imagens com EXP < 2; no gold, essas imagens aparecem como `'ND'`. Essa equivalência foi inferida dos dados (a correspondência é exata nos dois CSVs) e é verificada no notebook.
- Nenhuma imagem referenciada nos CSVs está faltando no diretório `data/images/`; não há sobreposição de arquivos entre treino e teste.
- **Sobreposição de embriões:** 200 dos 249 embriões do gold também têm outras imagens no silver. Apenas 49 embriões (58 imagens) são inéditos.
- Ambiente: **sem GPU**, treinamento roda inteiramente em CPU.

## Decisões de design

1. **CNN construída do zero** (não transfer learning), alinhada ao conteúdo do curso: conv → ReLU → pooling → GlobalAveragePooling2D (em vez de Flatten, para reduzir parâmetros/overfitting) → denso → softmax.
2. **Modelo único multi-saída** (*multi-task*): um backbone convolucional compartilhado + 3 cabeças de classificação (softmax) — EXP (5 classes), ICM (4 classes), TE (4 classes). Mais eficiente que treinar 3 CNNs separadas e permite compartilhamento de features.
3. **Entrega em notebook Jupyter** (`gardner_cnn.ipynb`, raiz do projeto), com células organizadas e gráficos seguindo paleta/estilo consistente.
4. **Framework:** TensorFlow/Keras (`tf.keras.*`), instalado via pip no `.venv` local. Também `scikit-learn` (split por grupo, matriz de confusão, kappa).

## Pipeline de dados

- Carregar os dois CSVs com `pandas` (`sep=';'`), descartar a coluna `Unnamed: 4` do teste.
- Extrair `embryo_id` do nome do arquivo (prefixo antes do `_`) para possibilitar split por grupo.
- **Split treino/validação:** `GroupShuffleSplit` (sklearn) sobre `Gardner_train_silver.csv`, agrupando por `embryo_id`, ~85/15, seed fixa (42) para reprodutibilidade.
- **Teste:** `Gardner_test_gold_onlyGardnerScores.csv` fica totalmente de fora do treino/validação — benchmark final.
- Para cada alvo (EXP/ICM/TE) no teste, linhas com `'ND'` ou `NaN` naquele alvo são excluídas **apenas da avaliação daquele alvo específico** (a imagem permanece válida para os outros dois alvos).
- `tf.data.Dataset` para carregar/decodificar PNG, redimensionar para **224×168** (largura×altura, mantendo a proporção 4:3 original) e normalizar para [0,1].
- **Augmentação (somente treino):** flip horizontal e vertical, rotação aleatória (±15°), variação de brilho — via `tf.keras.layers.RandomFlip/RandomRotation/RandomBrightness`, ativas só em treino. `RandomBrightness` precisa de `value_range=(0, 1)`, pois as imagens já estão normalizadas.

## Arquitetura do modelo

```
Input (168,224,3)  # altura, largura, canais
 → [Conv2D(32,3x3,relu) ×2 → BatchNorm → MaxPool2D]  # BatchNorm com momentum=0.9 em todos os blocos
 → [Conv2D(64,3x3,relu) ×2 → BatchNorm → MaxPool2D]
 → [Conv2D(128,3x3,relu) ×2 → BatchNorm → MaxPool2D]
 → [Conv2D(256,3x3,relu) → BatchNorm → MaxPool2D]
 → GlobalAveragePooling2D
 → Dense(256, relu) → Dropout(0.5)
 → 3 cabeças:
     - Dense(5, softmax, name="exp_output")
     - Dense(4, softmax, name="icm_output")
     - Dense(4, softmax, name="te_output")
```

- **Loss:** `sparse_categorical_crossentropy` em cada cabeça, `loss_weights` iguais (1.0 cada).
- **Class weights:** calculados por alvo (`compute_class_weight`) e convertidos em `sample_weight` por amostra em cada saída (Keras multi-output não aceita `class_weight` dict diretamente). Limitados a no máximo 10, para que uma única imagem rara não domine o lote. Usados só no treino; a validação não tem pesos, para que a `val_loss` não fique dominada por poucas imagens raras (ICM=2 tem só 4 imagens na validação).
- **Otimizador:** Adam (lr=3e-4).
- **Callbacks:** `EarlyStopping` (patience=8, restore_best_weights, start_from_epoch=15), `ReduceLROnPlateau`, `ModelCheckpoint` salvando o melhor modelo em `models/gardner_cnn.keras`, `CSVLogger` salvando as métricas de cada epoch em `models/training_log.csv`.

### Nota de implementação (Keras 3)

Passar `y` e `sample_weight` como **dicts** nomeados por saída, combinados num `tf.data.Dataset` de 3 elementos `(x, y_dict, weight_dict)`, causa um `KeyError` interno no Keras 3.15 (`CompileLoss.resolve_path`). A solução é usar **tuplas** na mesma ordem das saídas do modelo (EXP, ICM, TE) tanto para `y` quanto para `sample_weight`, e compilar o modelo com `loss`/`loss_weights`/`metrics` como **listas** (não dicts) na mesma ordem. As chaves de métricas no histórico continuam nomeadas pelos nomes das camadas de saída (`exp_output_loss`, etc.), então não há perda de legibilidade.

## Avaliação

- Curvas de treino/validação (loss total e por cabeça, accuracy por cabeça) ao longo das epochs.
- No conjunto de validação (silver, mesma distribuição do treino): accuracy por alvo e, nas imagens avaliáveis (sem a classe 3 em ICM/TE), accuracy, linha de base e Kappa quadrático.
- No conjunto de teste **gold** (benchmark real, com filtragem de `ND`/NaN por coluna):
  - Accuracy por alvo
  - Matriz de confusão (matplotlib)
  - **Cohen's Kappa quadrático** (apropriado para escalas ordinais — penaliza mais erros "distantes" que erros "vizinhos")
  - Relatório de classificação (precisão/recall/F1 por classe)
  - Accuracy e Kappa separados entre embriões presentes no silver e embriões inéditos
- Discussão final: diferença entre rótulos *silver* (1 especialista) e *gold* (consenso) como possível fonte de domain shift; limitações do desbalanceamento de classes; sugestões de melhoria (transfer learning, ordinal regression, test-time augmentation).

## Estrutura do notebook (`gardner_cnn.ipynb`)

1. Setup e imports
2. Carga e exploração dos CSVs (distribuição de classes, exemplos de imagens)
3. Split treino/validação por embrião
4. Pipelines `tf.data` (treino com augmentação, validação/teste sem)
5. Definição do modelo (backbone + 3 cabeças), `model.summary()`
6. Compilação (losses, loss_weights, métricas, otimizador)
7. Callbacks
8. Treinamento (`model.fit`, com pesos de amostra por classe)
9. Curvas de treino (gráficos)
10. Avaliação na validação
11. Avaliação no teste gold (filtragem ND/NaN, matrizes de confusão, kappa)
12. Conclusões

## Arquivos do projeto

- `gardner_cnn.ipynb` — notebook principal com todo o pipeline
- `requirements.txt` — dependências (`tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pillow`, `jupyterlab`, `ipykernel`, `nbconvert`)
- `models/gardner_cnn.keras` — melhor modelo salvo, gerado via `ModelCheckpoint`
- `spec.md` — este documento
- `plan.md` — planejamento/acompanhamento da execução

## Critérios de verificação

- O pipeline de dados carrega lotes sem erro e os shapes batem com o esperado.
- `model.summary()` mostra arquitetura coerente com o desenho acima.
- O treinamento roda sem erro de shape/loss.
- As métricas finais no conjunto gold são impressas e as matrizes de confusão são geradas sem erro.

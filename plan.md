# Plano de execução: CNN Gardner

Acompanhamento da implementação descrita em `spec.md`.

## Etapas

1. **Ambiente** — instalar `tensorflow` e `scikit-learn` no `.venv` existente (Python 3.13; `pandas`/`numpy`/`matplotlib`/`pillow` já presentes). ✅ Concluído (`tensorflow==2.21.0`, `scikit-learn==1.9.1`).
2. **Exploração dos dados** — confirmar schema dos CSVs, contagem de imagens, valores ausentes/`ND` no gold, ausência de sobreposição treino/teste. ✅ Concluído durante o levantamento inicial.
3. **`requirements.txt`** — registrar versões instaladas. ✅ Concluído.
4. **Notebook `gardner_cnn.ipynb`** — criado com as 12 seções descritas em `spec.md` (setup, exploração, split por embrião, pipelines `tf.data`, modelo multi-saída, compilação, callbacks, treino, curvas, avaliação validação, avaliação gold, conclusões). ✅ Concluído.
5. **Smoke tests isolados (fora do notebook)** — antes de rodar o notebook completo, validados em script Python avulso:
   - Pipeline `tf.data` carrega lotes com shapes corretos. ✅
   - `model.fit` com 1 epoch roda sem erro em lote pequeno. ✅
   - Descoberto e corrigido bug do Keras 3: `y`/`sample_weight` como **dicts** nomeados quebra `model.fit` com `KeyError` interno; solução foi usar **tuplas** na ordem das saídas (EXP, ICM, TE) e compilar com `loss`/`metrics` como listas. ✅ Corrigido no notebook.
   - 1 epoch no split completo real (sem subamostragem) roda sem erro, ~88s/epoch em CPU. ✅
6. **Execução completa do notebook** (`jupyter nbconvert --to notebook --execute --inplace`) — 🔄 **refeita após as correções de 02/11/2026** (ver abaixo), rodando até 60 epochs com `EarlyStopping` (patience=8). Estimativa: até ~90 min se rodar todas as 60 epochs; pode terminar antes por early stopping. A execução anterior não deixou métricas no notebook, e o modelo dela foi treinado com os bugs corrigidos abaixo.
7. **Verificação final** (pendente, após a etapa 6 terminar):
   - Conferir que todas as células do notebook executaram sem erro.
   - Conferir `model.summary()` e curvas de treino geradas.
   - Conferir métricas de validação e do conjunto gold (accuracy, Cohen's Kappa quadrático, matrizes de confusão) impressas/plotadas corretamente.
   - Confirmar que `models/gardner_cnn.keras` foi salvo pelo `ModelCheckpoint`.
8. **Entrega** — notebook executado com outputs salvos, `models/gardner_cnn.keras`, `requirements.txt`, `spec.md`, `plan.md`.

## Decisões tomadas durante a execução (não previstas no plano original)

- TensorFlow 2.21.0 suporta Python 3.13 nativamente — não foi necessário downgrade do interpretador do `.venv`.
- Execução do notebook via `jupyter nbconvert --execute --inplace` em processo destacado (`nohup ... &`, sem herdar stdin), para rodar em background sem bloquear a sessão, com `--ExecutePreprocessor.timeout=10800` (3h) para cobrir o pior caso de treino completo em CPU.

## Correções de 02/11/2026

- `RandomBrightness` recebeu `value_range=(0, 1)`. Antes, a variação chegava a ±25,5 em imagens de 0–1, e cerca de metade delas ficava preta.
- Altura e largura corrigidas para `168, 224`. Antes, as imagens eram distorcidas para retrato.
- Pesos de classe removidos da validação, para estabilizar a `val_loss`.
- Métricas do gold separadas entre embriões presentes no silver e inéditos.
- Classe 3 de ICM/TE documentada como não determinado e verificada no notebook.
- Conclusões reescritas para não afirmar resultados antes de eles existirem.
- Teste com 1 epoch em cópia do notebook: as 17 células de código executaram sem erro.

## Status atual

Treinos 1 a 5 concluídos, com resultados em `history_train.md`. O melhor até agora é o Treino 5, com EfficientNetB0 pré-treinada no ImageNet e ICM/TE treinados só com imagens avaliáveis. ICM=2 e TE=2 ainda são os pontos fracos. Notebook executado, registro por epoch e resultados de cada treino são versionados, mesmo sem ser o melhor. Os modelos ficam fora do git, pois podem ser reproduzidos pelo código.

## Pendências de entrega

Segundo `docs/DEEP LEARNING - INFORMAÇÕES SOBRE AVALIAÇÕES.pdf`, a entrega exige arquivo fonte (o notebook `gardner_cnn.ipynb` faz esse papel), link do Colab, vídeo de 5 a 7 min e Resumo Expandido de 4 a 5 folhas. O prazo no PDF era 21/10/2026, mas a entrega foi atualizada para 09/11/2026.

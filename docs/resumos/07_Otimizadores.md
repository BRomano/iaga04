# Otimizadores: SGD, Momentum e Adam
*(utfpr_IAGA_Deep_Learning_OTIMIZADORES.pdf)*

## Ideia central
Otimizadores transformam gradientes em atualizações de parâmetros. Fluxo: `ENTRADA (x/mini-batch) → FORWARD (ŷ) → LOSS (L) → BACKPROP (∇L) → OTIMIZADOR (Δθ) → NOVOS θ`. Em RNA, **aprender** significa modificar parâmetros (principalmente pesos e bias) para melhorar o desempenho.

## 1. Batch, SGD e Mini-batch
Antes da regra de atualização, decide-se quantas amostras estimarão o gradiente:
| Modo | Amostras usadas | Características |
|---|---|---|
| **Batch GD** | todas as amostras | gradiente estável, passo mais caro |
| **SGD estrito** | 1 amostra | atualizações baratas, mais ruído |
| **Mini-batch** | pequeno grupo | equilíbrio prático, vetorização eficiente |

Na prática de Deep Learning, "SGD" frequentemente opera com mini-batches (não apenas 1 amostra).

## 2. SGD — atualização direta com o gradiente atual
`w_{t+1} = w_t − η·g_t`

Regra simples, **sem memória** do caminho anterior. Exemplo (η=0,1, g=-10 constante diminuindo): converge progressivamente (w: 0,00 → 1,00 → 1,60 → 1,96...).

### Ruído do SGD
Mini-batches diferentes fornecem estimativas diferentes do gradiente para o mesmo peso (`mini-batch A → ĝ₁`, `mini-batch B → ĝ₂`). Ruído **não é** erro do algoritmo — é consequência de estimar o gradiente com subconjuntos de dados. Vantagem prática: cada atualização custa muito menos que usar todo o conjunto.

## 3. Momentum — acrescentando memória ao movimento
O gradiente atual é combinado com uma "velocidade" acumulada:
```
Etapa 1 (memória):     v_t = β·v_{t−1} + g_t
Etapa 2 (atualização): w_{t+1} = w_t − η·v_t
```
β controla quanto do passado permanece. Analogia: uma bola ganha velocidade em direção persistente e resiste a oscilações laterais.

**Exemplo numérico** (w₀=0, v₀=0, η=0,03, β=0,9):
| t | g_t | v_t | w_{t+1} |
|---|---|---|---|
| 1 | −10,000 | −10,000 | 0,300 |
| 2 | −8,800 | −17,800 | 0,834 |
| 3 | −6,664 | −22,684 | 1,515 |

A velocidade acumula direção: o passo não depende apenas de g_t.

## 4. Adam (Adaptive Moment Estimation) — duas memórias trabalhando juntas
Mantém média móvel da **direção** e da **magnitude quadrática** dos gradientes:
```
1º momento (direção):   m_t = β1·m_{t−1} + (1−β1)·g_t
2º momento (magnitude²): v_t = β2·v_{t−1} + (1−β2)·g_t²
```
**Correção de viés** (necessária pois médias móveis iniciais ficam puxadas para zero):
```
m̂_t = m_t / (1 − β1^t)
v̂_t = v_t / (1 − β2^t)
```
Valores típicos: β1=0,9 (memória da direção), β2=0,999 (memória da magnitude), ε≈10⁻⁸ (estabilidade numérica no denominador).

Adam = memória de direção + escala adaptativa por parâmetro.

## 5. Tabela comparativa

| Aspecto | SGD | Momentum | Adam |
|---|---|---|---|
| Usa gradiente atual | Sim | Sim | Sim |
| Memória direção | Não | Sim | Sim (m_t) |
| Memória magnitude g² | Não | Não | Sim (v_t) |
| Correção de viés inicial | Não | Não | Sim |
| Escala adaptativa por parâmetro | Não (forma básica) | Não (forma básica) | Sim |
| Estados extras por parâmetro | 0 | 1 | 2 |
| Custo de memória | Menor | Intermediário | Maior |
| Pode usar mini-batches | Sim | Sim | Sim |

Não há como dizer que um otimizador é superior a outro — a escolha depende da arquitetura, dos dados, da taxa de aprendizagem, do esquema de decaimento, da regularização e do objetivo experimental.

## 6. Implementação (Keras/TensorFlow)
```python
import tensorflow as tf

model = keras.Sequential()
model.add(keras.layers.Dense(8))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss="mse")

adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
)
```

## Conclusão
Mesmo backpropagation (cálculo dos gradientes), diferentes regras de atualização: `SGD (g_t)`, `Momentum (g_t + memória)`, `Adam (m̂_t / √v̂_t)`.

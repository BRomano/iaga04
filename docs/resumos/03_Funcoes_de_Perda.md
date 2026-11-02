# Funções de Perda (Loss Function)
*(utfpr_IAGA_Deep_Learning_PERDA.pdf)*

## Ideia central
A função de perda transforma a discrepância entre previsão (ŷ) e alvo (y) em uma **penalização numérica**, definindo o objetivo que a rede tentará minimizar durante o treinamento. A loss **não** atualiza os pesos — ela apenas define o objetivo; quem calcula derivadas é o backpropagation, e quem atualiza parâmetros é o otimizador.

## Fluxo do treinamento
`ENTRADA (x) → MODELO/FORWARD → PREVISÃO (ŷ) → LOSS L(y,ŷ) → BACKPROP → GRADIENTE → OTIMIZADOR`

## Conceitos-chave

### Previsão, alvo, erro e loss são quatro coisas diferentes
- **Alvo (y):** resposta desejada
- **Previsão (ŷ):** saída do modelo
- **Erro (e):** `ŷ − y`
- **Loss (L):** penalização escolhida (depende da regra)

Mesmo erro pode gerar losses diferentes (erro absoluto vs. quadrático).

### Loss, Custo e Métrica
- **Loss (L):** por amostra
- **Custo J(θ):** agrega as losses (ex.: média)
- **Métrica:** interpreta desempenho (ex.: acurácia) — não usada para otimização

### Por que o erro simples não basta
Erros opostos podem se cancelar na soma (ex.: `-2 + 2 = 0`), escondendo que o modelo está, na verdade, errando bastante. Solução: transformar o erro (quadrado ou valor absoluto) antes de agregar.

## Funções de perda para regressão

### MSE — Mean Squared Error (Erro Quadrático Médio)
`MSE = (1/N)·Σ(yi − ŷi)²`
Eleva ao quadrado → impede cancelamento e **enfatiza erros grandes** (ex.: erro passa de 2 para 4, e² passa de 4 para 16).

### MAE — Mean Absolute Error (Erro Absoluto Médio)
`MAE = (1/N)·Σ|yi − ŷi|`
Cresce **linearmente** com o erro (menos sensível a outliers que o MSE).

### MSE × MAE
MSE penaliza muito erros grandes (curva quadrática); MAE cresce linearmente (curva em V).

## Funções de perda para classificação

### Binary Cross-Entropy (BCE) — classificação binária
A previsão `p` representa uma probabilidade (`p = P(y=1|x)`).
- Se `y=1`: `L = −log(p)` (queremos p próximo de 1)
- Se `y=0`: `L = −log(1−p)` (queremos p próximo de 0)
- Forma unificada: `L = −[y·log(p) + (1−y)·log(1−p)]`

Quanto mais perto de zero a probabilidade atribuída à classe verdadeira, maior a penalização (log cresce rapidamente perto de 0).

### Cross-Entropy multiclasse
Com alvo em *one-hot encoding* e saída *softmax*: `L = −Σ yk·log(pk)`.
Exemplo: y=[0,1,0] (classe "cão"), p=[0,10; 0,80; 0,10] → L ≈ 0,2231. Se p(cão) caísse para 0,05, a loss subiria para ≈2,9957.

### Softmax ≠ Cross-Entropy
- **Logits:** pontuações brutas não normalizadas da camada final
- **Softmax:** transforma logits em distribuição de probabilidade
- **Cross-Entropy:** avalia a qualidade dessas probabilidades frente ao alvo

Bibliotecas de DL frequentemente combinam softmax + cross-entropy a partir dos logits (estabilidade numérica).

## Tabela-resumo

| Função | Uso típico | Ideia | Comportamento |
|---|---|---|---|
| MSE | Regressão | média do erro² | penaliza muito erros grandes |
| MAE | Regressão | média do \|erro\| | crescimento linear |
| BCE | Classificação binária | −log prob. correta | pune confiança errada |
| Cross-Entropy | Classificação multiclasse | −log p(classe correta) | natural com softmax |

## Implementação (Keras/TensorFlow)
```python
import tensorflow as tf
# Regressão
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
# Classificação binária
bce = tf.keras.losses.BinaryCrossentropy()
# Classificação multiclasse
cce = tf.keras.losses.CategoricalCrossentropy()
```

## Conclusão
`loss ≠ gradiente ≠ backpropagation ≠ otimizador` — a escolha da função de perda influencia diretamente a forma e a magnitude dos gradientes, portanto duas losses diferentes podem produzir comportamentos de treinamento distintos sobre os mesmos dados.

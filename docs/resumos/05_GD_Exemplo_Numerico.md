# Gradiente Descendente — Exemplo Numérico Completo
*(utfpr_IAGA_Deep_Learning_GD_PERDA_EXEMPLO_NUMERICO.pdf)*

## Ideia central
Exemplo numérico passo a passo mostrando como pesos (w) e bias (b) são atualizados em múltiplas iterações de Gradiente Descendente, usando um modelo de regressão linear simples.

Fluxo: `ENTRADA → PREVISÃO → PERDA → GRADIENTE → ATUALIZAÇÃO → NOVA PREVISÃO`

## Configuração do problema
- Taxa de aprendizado: α = 0,1
- Pontos: (1,2), (2,3), (3,5)
- Modelo: `ŷ = wx + b`
- Inicialização: w₀ = 0, b₀ = 0
- Atualização conjunta de w e b em 3 iterações

## Passo 0 — Previsões, erros e perdas iniciais
Com w₀=0, b₀=0, todas as previsões são 0:
| Ponto | ŷ | erro (e=ŷ−y) | L=½e² |
|---|---|---|---|
| (1,2) | 0 | −2 | 2,0 |
| (2,3) | 0 | −3 | 4,5 |
| (3,5) | 0 | −5 | 12,5 |

**Perda total:** L = 19

## Derivadas parciais (regra da cadeia)
```
∂L/∂w = Σ(ei·xi) = (−2·1) + (−3·2) + (−5·3) = −23
∂L/∂b = Σ(ei) = −2 − 3 − 5 = −10
```

## Iteração 1 (atualização de w₀,b₀ → w₁,b₁)
```
w₁ = w₀ − α·∂L/∂w = 0 − 0,1·(−23) = 2,3
b₁ = b₀ − α·∂L/∂b = 0 − 0,1·(−10) = 1,0
```

Com w₁=2,3, b₁=1: previsões (3,3 / 5,6 / 7,9); erros (1,3 / 2,6 / 2,9); **Perda total = 8,43**.

Gradientes: `∂L/∂w = 15,2`; `∂L/∂b = 6,8`.

## Iteração 2 (w₁,b₁ → w₂,b₂)
```
w₂ = 2,3 − 0,1·15,2 = 0,78
b₂ = 1,0 − 0,1·6,8 = 0,32
```

Com w₂=0,78, b₂=0,32: previsões (1,10 / 1,88 / 2,66); erros (−0,90 / −1,12 / −2,34); **Perda total ≈ 3,77** (já reduzida em relação à iteração anterior).

Gradientes: `∂L/∂w = −10,16`; `∂L/∂b = −4,36`.

## Iteração 3 (w₂,b₂ → w₃,b₃)
```
w₃ = 0,78 − 0,1·(−10,16) = 1,796
b₃ = 0,32 − 0,1·(−4,36) = 0,756
```

## Evolução dos pesos
| Iteração | w | b | Perda total |
|---|---|---|---|
| 0 | 0,00 | 0,00 | 19,00 |
| 1 | 2,30 | 1,00 | 8,43 |
| 2 | 0,78 | 0,32 | 3,77 |
| 3 | 1,796 | 0,756 | — |

## Conclusão
O algoritmo segue atualizando peso e bias iterativamente até atingir um critério de parada: **número máximo de iterações (épocas)** ou **erro abaixo de um limiar aceitável**. Nota-se que a perda não decresce de forma estritamente monótona a cada iteração individual (w oscila entre 2,3 e 0,78), mas a tendência geral com α adequado é de convergência.

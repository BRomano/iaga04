# Gradiente Descendente (GD) e Perda (Loss)
*(utfpr_IAGA_Deep_Learning_GD_PERDA.pdf)*

## Ideia central
A rede não conhece o melhor peso de antemão. Ela começa com um valor inicial e usa a **inclinação local** (derivada) da função de perda para decidir como ajustar esse peso — esse processo iterativo é o **Gradiente Descendente**.

Fluxo: `ENTRADA → PREVISÃO → PERDA → GRADIENTE → ATUALIZAÇÃO → NOVA PREVISÃO`

## 1. Por que uma rede precisa "descer"?
Otimizar é ajustar parâmetros para reduzir progressivamente a função de perda.

**Analogia do vale na neblina:** você não enxerga todo o relevo, apenas a inclinação local, e caminha para baixo.

| Analogia | Machine Learning |
|---|---|
| relevo | função de perda |
| posição | valor do peso |
| inclinação | gradiente |
| passo | learning rate |

## 2. Problema didático fixo
Pontos: (1,2), (2,3), (3,5); modelo `ŷ = wx + b`; α = 0,1; peso inicial w₀=0, bias inicial b₀=0.

## 3. Previsão, erro e perda não são a mesma coisa
- Previsão: `ŷ = 1`
- Erro assinado: `ŷ − y = −5`
- Perda quadrática de uma amostra: `L = ½·(ŷ−y)² = 12,5`

## 4. Experimentando pesos manualmente
Tabela de valores de w vs. perda L(w) mostra uma curva em formato de parábola com **mínimo em w = 2,5** (menor perda).

## 5. Derivada: inclinação local da perda
A derivada responde: "se eu alterar um pouco o peso, a perda sobe ou desce?"
- `dL/dw < 0` → aumentar w tende a **diminuir** L (mover para a direita)
- `dL/dw > 0` → aumentar w tende a **aumentar** L
- `dL/dw = 0` → ponto estacionário

## 6. Derivação passo a passo (regra da cadeia)
```
ŷ = w·x + b
L = ½·(ŷ − y)²
dL/dw = dL/dŷ · dŷ/dw = (ŷ − y)·x
```
No ponto inicial (w=0, x=2, b=1, y=6): `ŷ = 1`, `dL/dw = (1−6)·2 = −10`.

## 7. Regra do gradiente descendente
`w_{t+1} = w_t − η·∂L/∂w`

O **sinal negativo** transforma a direção de maior subida em uma direção de descida. Se usássemos "+", estaríamos subindo a perda (*Gradient Ascent*).

## 8. Trajetória do gradiente descendente
A cada passo, o peso se move em direção ao mínimo seguindo a reta tangente local; com η=0,1 a trajetória converge progressivamente para w ≈ 2,5.

## 9. Learning rate: tamanho do passo
O gradiente indica a **direção**; η controla a **distância** percorrida em cada atualização.
- η muito pequeno (0,01): convergência lenta.
- η adequado (0,1): convergência eficiente.
- η muito grande (0,6): oscilação ou divergência.

Não existe um learning rate universalmente "correto" — depende do problema e da dinâmica de treinamento.

## 10. Gradient Descent ≠ Backpropagation
| Etapa | Função |
|---|---|
| Forward | calcula a saída: x → ŷ |
| Loss | mede a discrepância: ŷ−y → L |
| Backprop | calcula derivadas: L → ∇L |
| GD | atualiza parâmetros: ∇L → θ' |

**Erro comum:** dizer que "Gradient Descent calcula o gradiente" — na verdade, backpropagation calcula; o método de otimização (GD) apenas usa esse gradiente.

## 11. Erros conceituais comuns
- "Gradiente é o erro." → Não. A loss quantifica a discrepância; o gradiente mede a variação da loss.
- "Gradiente negativo significa diminuir w." → Não. Com `w ← w − ∂L/∂w`, um gradiente negativo faz w **aumentar**.
- "Learning rate maior é sempre melhor." → Não. Passos grandes podem oscilar ou divergir.
- "Backprop = Gradient Descent." → Não. Backprop calcula gradientes; GD os usa.
- "Gradiente zero = mínimo global." → Não necessariamente: pode ser mínimo local, máximo ou ponto de sela.

## Conclusão
O gradiente descendente é o mecanismo que traduz a inclinação local da função de perda em uma atualização de peso, mas depende criticamente da taxa de aprendizado e não garante, por si só, encontrar o mínimo global.

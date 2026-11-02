# Backpropagation — Treinamento de Redes Neurais
*(utfpr_IAGA_Deep Learning_BACKPROPAGATION.pdf)*

## Ideia central
**Backpropagation é o algoritmo de treinamento que torna possível o uso de Deep Learning.** Ele calcula gradientes propagando informação do custo (erro) da camada de saída de volta até a camada de entrada, aplicando repetidamente a regra da cadeia.

## Estrutura geral do treinamento de uma rede
1. **Definição da arquitetura**
2. **Aprendizado**
3. **Feedforward**
4. **Backpropagation**

## 1. Por que redes de uma camada não bastam
Redes de uma única camada mapeiam entradas diretamente em saídas, sem representação interna — a codificação de entrada precisa ser suficiente para o mapeamento. Isso impede aprender problemas como a função **XOR** (não linearmente separável).

**Minsky e Papert** demonstraram matematicamente essa limitação do Perceptron e, por não acreditarem em um método de treinamento para múltiplas camadas, concluíram (erroneamente) que redes neurais seriam sempre limitadas. O desenvolvimento do **backpropagation** (Rumelhart, Hinton e Williams, 1986) mostrou que é possível treinar eficientemente redes com camadas intermediárias, originando as **MLPs**.

## 2. Redes feedforward profundas
`y = f(x; θ)` — o dado flui da entrada para a saída, sem realimentação do erro durante o forward.
- **x:** vetor de entrada
- **y:** vetor de saída (previsão/classificação)
- **θ:** parâmetros (pesos W e bias b)
- **f():** composição de transformações lineares + funções de ativação não lineares

## 3. Camadas ocultas
O dado não especifica o que cada camada deve calcular:
- **Observado:** entrada x, alvo y, erro na saída
- **Não observado diretamente:** função de ativação, representação interna, papel de cada unidade
- **Aprendido:** parâmetros θ, features úteis, aproximação da função-alvo

## 4. Unidades ocultas e funções de ativação
ReLU `g(z)=max(0,z)` é escolha padrão; sigmoide e tanh também são usadas. A escolha é avaliada empiricamente no conjunto de validação.

## 5. Arquitetura: profundidade, largura e aproximação universal
- Uma camada oculta já pode aproximar ampla classe de funções.
- Redes mais profundas podem usar mais unidades/parâmetros, mas a otimização fica mais difícil.
- Largura = quantidade de neurônios por camada → maior capacidade de representação, porém maior risco de *overfitting* e custo computacional.

## 6. Retropropagação do erro — ideias-chave
- **Forward propagation:** calcula ativações e custo.
- **Backpropagation:** aplica repetidamente a regra da cadeia, propagando o erro da saída até a entrada.

Sequência de 3 passos durante o treinamento:
1. Um padrão é apresentado à entrada; o resultado flui pela rede até a saída.
2. A saída obtida é comparada à saída desejada; calcula-se o erro.
3. O erro é propagado da saída até a entrada, e os pesos das camadas internas são ajustados conforme o erro retropropagado.

## 7. Do discriminador linear à regra delta generalizada
Treinamento do discriminador linear (ativação sgn):
```
Δwi = η·(y_l^d − y_l)·x_{i,l}
w_i^new = w_i^old + Δwi
```
**Regra delta padrão** implementa gradiente descendente no erro quadrático para ativações lineares:
`E(w*) = Σ(y(w*;x_l^d) − y_l^d)²`

Redes sem camadas intermediárias só resolvem superfícies de erro em formato de **parabolóide** (um único mínimo). Superfícies mais complexas exigem camadas intermediárias — mas ficam sujeitas a **mínimos locais**.

**Regra delta generalizada:** funciona com funções de ativação semilineares e diferenciáveis (ex.: sigmoide).

## 8. Taxa de aprendizado (η) e momentum
- η maior → mudanças maiores nos pesos → aprendizado mais rápido, mas risco de **oscilação**.
- Ideal: maior η possível sem causar oscilação.
- **Momentum:** termo que incorpora o efeito de mudanças passadas de pesos na direção atual do movimento no espaço de pesos — útil em superfícies de erro com gargantas, curvas acentuadas, vales suaves ou platôs.

## 9. Algoritmo geral do backpropagation
1. Escolher aleatoriamente parâmetros iniciais θ (pesos W e taxa η)
2. Estabelecer critério de parada (épocas máximas Kmax, ou erro E(W) ≤ λ)
3. Calcular os erros δ de saída e das camadas ocultas
4. Atualizar os parâmetros Δw
5. Recalcular todos os pesos `W(1) = W(0) + ΔW` e retornar ao passo 3 até o critério de parada

## 10. Por que não basta escolher parâmetros manualmente
| θ muito genérico | θ manual | θ aprendido |
|---|---|---|
| alta capacidade, risco de generalização fraca | depende de conhecimento humano, pode perder fatores relevantes | adaptada aos dados, serve à tarefa |

## 11. Resultado do treinamento: representação aprendida
```
modelo linear:          y = wᵀx + b
features fixas:          y = wᵀφ(x) + b
representações aprendidas: h = g(Wᵀx + c)
saída:                   ŷ = wᵀh + b
```
O aprendizado escolhe a representação que torna a tarefa resolvível.

## Conclusão
O backpropagation é o que permite treinar redes com camadas ocultas, superando a limitação original do Perceptron de camada única e viabilizando o Deep Learning como o conhecemos.

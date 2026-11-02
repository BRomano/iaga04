# Redes Neurais Recorrentes (RNN)
*(utfpr_IAGA_Deep_Learning_RNN.pdf)*

## Ideia central
RNN é uma rede **com memória**, projetada para processar dados sequenciais (séries temporais): lê a sequência passo a passo e carrega um resumo do passado em um **estado oculto**.

## 1. O que é uma série temporal
Em uma série temporal, **a posição no tempo faz parte da informação** — embaralhar a ordem destrói a informação temporal, pois o modelo perderia a relação "antes → depois" que se deseja aprender.

Conceitos-chave:
- **Observação (xₜ):** valor medido em um instante
- **Timestep (t):** posição temporal
- **Janela:** bloco usado como entrada `[x_{t-2}, x_{t-1}, x_t]`
- **Horizonte:** quantos passos prever (t+1)
- **Feature:** atributo em cada instante (ex.: temperatura, venda)

### Série univariada × multivariada
| Aspecto | Univariada | Multivariada |
|---|---|---|
| Nº de features | 1 | 2 ou mais |
| Exemplo | apenas demanda | demanda + temperatura + umidade |
| Formato em RNN | (batch, timesteps, 1) | (batch, timesteps, n) |

## 2. Janelas temporais (sliding window)
Transforma passado em entrada e futuro em alvo: pega-se N valores consecutivos como entrada e o próximo valor como alvo. A **janela deslizante** move-se um passo por vez, gerando múltiplos pares entrada-alvo (amostras de treinamento) a partir da mesma série.

Ex.: `[10,12,13] → alvo: 15`; `[12,13,15] → alvo: 16`; `[13,15,16] → alvo: 18`.

- **L:** comprimento da janela de entrada
- **H:** horizonte de previsão (H=1: one-step; H>1: multi-step)
- **X:** conjunto das janelas de entrada; **y:** conjunto dos alvos

**Erro conceitual comum:** "cada janela é uma série independente" — **Não.** As janelas são exemplos criados a partir da mesma série e mantêm relação temporal, o que influencia a divisão treino/validação/teste.

**Normalização:** usada para facilitar contas e estabilizar o treinamento; parâmetros da transformação devem ser calculados **somente com o conjunto de treino**, para evitar vazamento de informação do futuro.

## 3. MLP × RNN
| Critério | MLP | RNN |
|---|---|---|
| Tipo de entrada | vetor fixo | sequência temporal |
| Memória temporal | não explícita | estado oculto h |
| Ordem dos dados | não é o foco central | fundamental |
| Representação da janela | [x1,x2,x3] | x1 → x2 → x3 |
| Vantagens | simples e eficiente | capta contexto temporal |
| Limitações | não modela dependência temporal | treinamento mais complexo |

A janela deslizante gera os exemplos supervisionados; a diferença está em como cada arquitetura representa e processa esses exemplos.

## 4. A célula recorrente
```
h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)
```
- **x_t:** observação atual; **h_{t-1}:** estado anterior; **h_t:** novo estado oculto
- **W_xh, W_hh, b_h:** parâmetros aprendidos (**compartilhados** em todos os instantes da sequência)
- O estado oculto é uma memória compacta, atualizada a cada passo com os mesmos pesos reutilizados no tempo.

### Unrolling no tempo
"Desenrolar" a RNN = visualizar a mesma célula recorrente em passos temporais sucessivos: `x1→h1→ŷ1, x2→h2→ŷ2, ...`, com pesos `Wxh, Whh, Why` e bias compartilhados.

**Erro conceitual comum:** "o estado oculto é uma cópia dos valores passados" — Não, é uma representação **aprendida e comprimida** do contexto.

## 5. Exemplo numérico de uma RNN
Janela normalizada (L=3): x1=0,0; x2=0,2; x3=0,3. Parâmetros: Wxh=0,7; Whh=0,5; bh=0,1; h0=0,0.

| t | xt | h_{t-1} | a_t (antes tanh) | h_t |
|---|---|---|---|---|
| 1 | 0,0 | 0,0000 | 0,1000 | 0,0997 |
| 2 | 0,2 | 0,0997 | 0,28985 | 0,2820 |
| 3 | 0,3 | 0,2820 | 0,4510 | 0,4227 |

Cada estado depende dos anteriores (h3 depende de x3 e de h2, que já carregava informação de x1 e x2) — essa dependência encadeada é a **recorrência**.

### Da memória interna à previsão
`ŷ = W_hy·h3 + b_y`. Exemplo: `ŷ = 1,1·0,4227 + 0,05 ≈ 0,5150`, muito próximo do alvo normalizado y=0,5.

## 6. Formas de usar uma RNN em séries temporais
- **Many-to-one:** vários timesteps entram, uma previsão sai.
- **Many-to-many:** sequência de entrada gera sequência de saídas.
- **One-step × Multi-step:** previsão de um único passo futuro vs. vários passos.
  - *single-shot/direta:* modelo produz todos os passos futuros de uma vez.
  - *autoregressiva:* modelo prevê um passo, usa a previsão como entrada seguinte (risco de **acúmulo de erro**).

## 7. Long Short-Term Memory (LSTM)
LSTM continua sendo uma rede recorrente, mas introduz **memória explícita (c_t)** e mecanismos chamados **portas** (*gates*).

Analogia: uma agenda com três decisões a cada instante — (1) o que esquecer do que estava guardado? (2) o que escrever de novo? (3) que parte da memória mostrar como saída?

| Porta | Função | Fórmula |
|---|---|---|
| **Esquecimento (f_t)** | decide quanto da memória anterior preservar | `f_t = σ(W_f·[h_{t-1},x_t] + b_f)` |
| **Entrada (i_t)** | controla quanta nova informação incorporar | `i_t = σ(...)`; `c̃_t = tanh(...)` |
| **Saída (o_t)** | controla quanto da memória atual expor como h_t | `o_t = σ(...)`; `h_t = o_t ⊙ tanh(c_t)` |

Próximo de 1 → preserva/escreve/mostra bastante; próximo de 0 → apaga/escreve/mostra pouco. As portas tornam o fluxo de memória controlável, facilitando dependências temporais mais longas.

## 8. Aplicações e limitações

**Aplicações típicas:** previsão de demanda/vendas, monitoramento de sensores industriais, detecção de anomalias, saúde/séries biomédicas, energia/clima/hidrologia, linguagem e áudio.

**Limitações e cuidados:** exigem boa preparação dos dados; podem sofrer com sequências muito longas; são sensíveis à escala dos dados; devem ser comparadas com baselines simples; nem sempre superam arquiteturas mais recentes.

## Conclusão
RNNs processam sequências mantendo um estado oculto que acumula contexto temporal com pesos compartilhados; LSTMs estendem essa ideia com memória explícita e portas para lidar melhor com dependências de longo prazo.

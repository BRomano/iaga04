# Multi-Layer Perceptron (MLP)
*(utfpr_IAGA_Deep Learning_MLP.pdf)*

## Ideia central
MLPs (também chamadas *deep feedforward networks* ou *feedforward neural networks*) são, por excelência, os modelos fundamentais de Deep Learning. Para entendê-las, parte-se do neurônio biológico, passa-se pelo neurônio artificial (Perceptron) e chega-se à organização de neurônios em redes multicamadas.

## Tópicos principais

### 1. Neurônio biológico
Formado por **dendritos** (entradas), **corpo celular** e **axônios** (saídas); comunicação via **sinapses** (químicas ou elétricas). Quando o impulso recebido atinge um limiar, o neurônio dispara um **potencial de ação**.

### 2. Breve histórico
Três publicações fundadoras: **McCulloch e Pitts (1943)** — primeiro modelo de rede neural; **Hebb (1949)** — rede de auto-organização; **Rosenblatt (1958)** — modelo Perceptron (aprendizado supervisionado). Retomada nos anos 1980 com Hopfield (1982) e **Backpropagation** (Rumelhart, Hinton, Williams).

### 3. O neurônio artificial (Perceptron)
Operação proposta por McCulloch e Pitts (1943):
1. Sinais na entrada (**X**)
2. Cada sinal multiplicado por um peso (**W**)
3. Soma ponderada: `v = Σ(Wi·Xi) + b`
4. Soma-se um **bias (b)**, nível basal de ativação
5. Se v exceder um limiar (*threshold*), a unidade produz uma saída (**ŷ**)

**Pesos:** indicam importância de cada entrada (positivo = excitatório, negativo = inibitório, zero = sem influência). **Bias:** desloca a saída independentemente das entradas, facilitando/dificultando a ativação.

Comparação neurônio artificial × biológico: inspiração, não cópia.

### 4. Como o neurônio trabalha — exemplo numérico
Entradas (x1=0,8; x2=0,3; x3=0,6; x4=0,1), pesos (w1=0,7; w2=-0,5; w3=0,4; w4=-0,3), bias=0,2.
`z = Σ(wi·xi) + b = 0,67`. Função de decisão (discriminador binário, limiar 0,5): como z ≥ 0,5, saída = 1 (Classe 1).

Um único neurônio só resolve problemas **linearmente separáveis** (AND, OR); a função **XOR** não é linearmente separável e exige redes de neurônios (múltiplas camadas).

### 5. Neurônios em redes
Redes neurais são organizadas em **camadas**; definidas por topologia, características dos nós e regras de treinamento. Treinar = ajustar pesos e bias com base em exemplos.

**Arquitetura:** camada de entrada (input) → camadas ocultas/intermediárias (extratoras de características) → camada de saída (output).

**Propriedades das RNAs:** Aprendizado, Generalização (robustez a ruído) e Abstração.

**Projeto de uma RNA envolve:** neurônios e função de ativação, topologia, pesos sinápticos (e algoritmo de treinamento), procedimento de validação — não há "fórmula" única.

### 6. Processo de aprendizado
Tipos de aprendizado conforme o *feedback*:
- **Supervisionado:** pares entrada/saída conhecidos.
- **Não supervisionado:** sem resposta desejada (ex.: clustering).
- **Por reforço:** crítico externo aplica recompensa/punição.

### 7. Treinamento supervisionado (Regra Delta)
Dado um conjunto `Ψ = {(x_l^d, y_l^d)}`, o objetivo é encontrar função `h` que aproxime a função verdadeira `f`. Erro = `ŷ − y`; ajustam-se pesos para reduzir esse erro (**Regra Delta**).

**Algoritmo geral:**
1. Iniciar pesos aleatórios (W)
2. Para cada par (X, d): calcular saída O e erro E
3. Se erro > tolerância ε: atualizar `W_novo = W_anterior + η·E·X`
4. Repetir até erro aceitável ou número máximo de epochs

`η` (taxa de aprendizado) controla a velocidade do ajuste.

**Perfil erro quadrático médio:** relacionado a um hiperparabolóide (modelo linear) com mínimo global único, mas com funções de ativação não lineares (ex.: sigmoide) a superfície pode ter **mínimos locais** e **platôs**.

### 8. Funções de ativação
Introduzem não linearidade, essenciais para redes multicamadas:
- **ReLU:** `max(x, 0)` — padrão em DL.
- **Sigmoide:** `1/(1+e^-x)`, saída em (0,1); mais próxima do comportamento biológico.
- **Tanh:** similar à sigmoide, saída em (-1,1).

## Conclusão
O poder das arquiteturas multicamadas está na combinação de neurônios organizados em camadas + funções de ativação não lineares, permitindo resolver problemas não linearmente separáveis — base para o treinamento via backpropagation (ver documento específico).

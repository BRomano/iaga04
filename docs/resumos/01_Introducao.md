# Introdução ao Deep Learning
*(utfpr_IAGA_Deep Learning_Introdução.pdf)*

## Ideia central
Situar o Deep Learning (DL) dentro da Inteligência Artificial (IA) e do Machine Learning (ML), explicando por que a abordagem baseada em regras manuais é insuficiente e por que surge a necessidade de aprendizado por dados e de representações em múltiplos níveis.

## Tópicos principais

### 1. O desafio de descrever o que sabemos
Tarefas triviais para humanos (reconhecer um cachorro, entender fala) são difíceis de formalizar em regras explícitas — características de um cachorro muitas vezes também se aplicam a um gato.

### 2. Inteligência Artificial: não há definição única
Diferentes autores enfatizam aspectos distintos:
- **Winston (1992):** computação que permite perceber, raciocinar e agir.
- **Levine (1987):** fazer o computador pensar inteligentemente.
- **Bellman (1978):** automação de atividades associadas ao pensamento humano.

Elo comum: dotar a máquina de características associadas à inteligência humana. A IA é **interdisciplinar** (linguística, filosofia, matemática, neurofisiologia, genética, engenharias, computação, psicologia, sociologia).

### 3. Paradigmas da IA
| Paradigma | Ideia | Exemplos |
|---|---|---|
| Simbólico | símbolos explícitos, regras e estruturas | sistemas especialistas, árvores de decisão |
| Evolucionista | evolução, seleção e reprodução simuladas | algoritmos genéticos, planejamento, otimização |
| Conexionista | aprendizado profundo e reconhecimento de padrões | redes neurais, visão computacional, voz |

O paradigma **conexionista** é o foco da disciplina.

### 4. Sistemas especialistas (conhecimento codificado manualmente)
Baseados em: pessoas especializadas → engenheiros do conhecimento → representações formais (ex.: Prolog) → base de conhecimento (fatos e regras) → máquina de inferência. Exemplo clássico: regras + fatos → inferência (ex.: "pizza é comida italiana" + "Sam gosta de comida italiana" ⇒ "Sam gosta de pizza").

**Limites:** conteúdo, recursos e prazos impõem escala limitada; com a explosão de dados da internet, a codificação manual se torna inviável.

### 5. Da IA baseada em regras ao aprendizado por dados
Regras rígidas (causa → regra → consequência) dão lugar a **aprender a aprender**, identificando padrões nos dados.

### 6. Machine Learning: indução a partir de dados
Base de treinamento → indução → hipótese/função aprendida. É a partir da interpretação humana que os dados se transformam em conhecimento.

### 7. Relação entre IA, ML e DL
Círculos concêntricos: **IA** (campo amplo) ⊃ **Machine Learning** (algoritmos que melhoram com experiência/dados) ⊃ **Deep Learning** (aprende representações em múltiplos níveis/camadas). IA generativa é uma aplicação moderna de DL (texto, imagem, áudio).

### 8. O que significa "Deep"?
Deep Learning aprende com a experiência e constrói **conceitos complexos a partir de conceitos simples**: conceitos simples → combinações → conceitos complexos → representação profunda.

Exemplo clássico (reconhecimento de imagem): camada de entrada (pixels) → 1ª camada oculta (bordas) → 2ª camada (cantos e contornos) → 3ª camada (partes de objetos) → saída (identidade do objeto). Representações vão de mais simples/específicas a mais abstratas/semanticamente ricas.

"Deep" refere-se à existência de **múltiplos níveis de composição ou camadas** em uma rede neural (profundidade), não apenas ao fato de ser ML.

## Conclusão
`Inteligência Artificial → Machine Learning → Deep Learning`, cada um um subcampo mais especializado do anterior, com o DL resolvendo o problema de representação de conhecimento complexo via composição de representações simples aprendidas automaticamente dos dados.

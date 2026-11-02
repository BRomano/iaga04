# Redes Neurais Convolucionais (CNNs)
*(utfpr_IAGA_Deep_Learning_CNN.pdf)*

## Ideia central
CNNs são redes neurais especializadas no processamento de dados com topologia em **grade** (séries temporais = grade 1D; imagens = grade 2D). A característica definidora de uma CNN é o uso da **convolução** como componente arquitetural, no lugar da multiplicação matricial geral em pelo menos uma camada.

## 1. Input Image — representações de imagem
Uma imagem pode ser representada como:
- **Imagem colorida** → matriz RGB (3 canais empilhados)
- **Imagem em tons de cinza** → matriz 2D (valores 0–255)
- **Imagem preto e branco (BW)** → matriz binária (0 e 1)

## 2. Operação de convolução
Conceito: `y = (f*g)(t) = ∫f(τ)·g(t−τ)dτ`. Em dados discretos, usa-se uma **soma local** em vez da integral.

Na CNN: a convolução aplica um **kernel** (detector de características) localmente sobre a imagem, produzindo um **mapa de características (feature map)**.
- Entrada, kernel e feature map são os três elementos centrais.
- Camadas convolucionais têm número "n" de filtros.
- Filtros (campos receptivos/detectores de características) têm pesos ajustados via **backpropagation**.
- Mapas de características contêm as informações mais importantes da imagem (processo transparente/automático durante o treinamento).

### Exemplos de kernels
| Operação | Efeito |
|---|---|
| Identity | mantém a imagem original |
| Ridge/edge detection | detecta bordas |
| Sharpen | realça detalhes |
| Box blur (normalizado) | suaviza a imagem |

## 3. Zero Padding
Controla a dimensão da saída ajustando o tamanho da imagem (preenchendo bordas com zeros) sem alterar as características, evitando que o feature map encolha excessivamente a cada convolução.

## 4. Função de ativação (ReLU)
Aplicada após a saída da convolução — característica não linear:
- Pixels negativos → transformados em 0.
- Demais valores → mantidos.
- Resultado: conjunto de mapas de características; a rede escolhe quais usar no treinamento.

## 5. Pooling (camada de agrupamento)
Resume respostas locais e introduz invariância aproximada a pequenas transformações (ângulos, fundos, formatos):
- Reduz resolução espacial e custo computacional → menos parâmetros a aprender.
- **Max pooling:** extrai a maior ativação local (mais utilizado).
- **Average pooling:** calcula a média local.

## 6. Flattening (camada de achatamento)
Transforma o *pooled feature map* (matriz) em um vetor, preparando os dados para entrada em uma **RNA densa** (camada totalmente conectada) com saída **softmax**.

## 7. Arquitetura completa da CNN
```
Input → [Convolution+ReLU → Pooling] (repetido N vezes) → Flatten → Fully Connected → Softmax → Output
```
- **Feature Extraction:** camadas convolucionais + pooling.
- **Classification:** camada totalmente conectada.
- **Probabilistic Distribution:** saída softmax com as probabilidades por classe (ex.: Horse 0,2 / Zebra 0,7 / Dog 0,1).

## Conclusão
CNNs combinam convolução (extração automática de características espaciais via kernels aprendidos), ativação não linear (ReLU) e pooling (redução de dimensionalidade/invariância), finalizando com uma rede densa para classificação — arquitetura essencial para dados com estrutura de grade, como imagens.

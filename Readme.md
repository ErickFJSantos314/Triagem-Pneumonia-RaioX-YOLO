# Triagem de Raio X Toráxico com Visão Computacional: Análise Comparativa YOLO (v8m, v9c, v10m)

Este repositório documenta um estudo comparativo aprofundado entre três arquiteturas de Object Detection da família YOLO: **YOLOv8m**, **YOLOv9c** e **YOLOv10m**. O foco é na triagem de achados sugestivos de Pneumonia em radiografias torácicas.

O objetivo principal foi analisar a evolução em termos de **eficiência arquitetônica** e **precisão na detecção**.
<img width="1760" height="432" alt="image" src="https://github.com/user-attachments/assets/e56981ed-c00f-4002-87d3-9ab536b5abbf" />


### Metodologia de Seleção de Modelos (YOLOv8m vs. v9c vs. v10m)

A comparação entre as diferentes gerações do YOLO deve ser baseada na complexidade (número de parâmetros), e não apenas na nomenclatura (m, c, n).

- **YOLOv8m (Medium):** Escolhido como o modelo de linha de base devido ao seu excelente equilíbrio entre precisão e velocidade, sendo o modelo Medium padrão da Ultralytics. Possui $\mathbf{\sim 25.8 \text{M}}$ parâmetros.

- **YOLOv9c (Compact):** Escolhido porque sua arquitetura Compact $\mathbf{\sim 25.8 \text{M}}$ parâmetros possui um número de parâmetros diretamente comparável ao $\text{YOLOv8m}$. Isso nos permite testar as inovações arquitetônicas do $\text{YOLOv9}$ (GELAN e PGI) em condições de complexidade de modelo quase idênticas.

- **YOLOv10m (Medium):** Escolhido para testar a eficiência de nova geração. O $\text{YOLOv10m}$ possui apenas $\mathbf{\sim 15.3 \text{M}}$ parâmetros. A comparação visa verificar se ele consegue superar a precisão dos modelos maiores ($\text{v8m}/\text{v9c}$) com uma arquitetura substancialmente mais leve.

## Foco Principal: Benchmarking e Eficiência

O projeto utiliza um dataset customizado de **~600 imagens** de Raio X Torácico (320x320px). A aplicação **Streamlit** foi desenvolvida como uma ferramenta para visualizar as diferenças de confiança de cada modelo em tempo real.

## Resultados Chave da Análise Comparativa

O desempenho foi medido pela métrica rigorosa **mAP@.5:.95** (Precisão Média em limiares de IoU de 50% a 95%).

| Modelo    | Parâmetros (M) | GFLOPs | mAP@.5:.95 | Inferência (ms) | Conclusão Principal                                                                 |
|-----------|----------------|--------|------------|-----------------|-------------------------------------------------------------------------------------|
| YOLOv8m   | 25.8           | 78.7   | 0.782      | 4.6             | **Vencedor da Velocidade**: Mais rápido e ideal para triagem em tempo real.         |
| YOLOv9c   | 25.3           | 102.3  | 0.787      | 8.0             | **Maior Latência**: Mais lento, mas com leve ganho de precisão.                     |
| YOLOv10m  | 15.3           | 58.9   | 0.794      | 5.9             | **Vencedor da Precisão/Eficiência**: Maior acurácia com 40% menos parâmetros.       |

<img width="1772" height="754" alt="image" src="https://github.com/user-attachments/assets/c75f6e3a-7f6c-4cf8-80ff-fb5a4f07af79" />

## Como Rodar a Aplicação Localmente

A aplicação interativa **Streamlit** (`index.py`) permite testar os modelos com suas próprias imagens e ver a comparação lado a lado.
<img width="1502" height="356" alt="image" src="https://github.com/user-attachments/assets/849580d4-4323-480a-99d6-5c956ed51032" />

### 1. Pré-requisitos

- **Python 3.10** (Recomendado)
- **Git LFS** (Necessário para baixar os modelos `.pt`)

### 2. Configuração do Ambiente

```bash
# Clone o repositório e baixe os modelos .pt (via Git LFS)
git clone https://github.com/ErickFJSantos314/YOLO-Pneumonia-XRay-Comparison.git
cd YOLO-Pneumonia-XRay-Comparison

# Instale as dependências (requirements.txt já está na raiz)
pip install -r requirements.txt]
```
### 3. Execução da Aplicação de Comparação
Execute o script Streamlit:
```bash
streamlit run index.py
```
A aplicação será aberta em seu navegador, pronta para receber o upload das imagens e demonstrar o "Pódio" de eficiência e precisão.
### ⚠️ Nota Importante: Modelos Pré-treinados

Este repositório utiliza o Git LFS (Large File Storage) para os arquivos de peso (.pt). Certifique-se de que o Git LFS está instalado em sua máquina para garantir que os arquivos reais (e não os ponteiros) sejam baixados após o git clone.

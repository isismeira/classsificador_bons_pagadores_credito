# Modelo classificador de bons pagadores de crédito com Naive Bayes

![naive_bayes_img (1).png](data:image/png;base64,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)

Este projeto visa criar um modelo de Machine Learning, utilizando o algoritmo Naive Bayes, para classificar clientes de crédito como bons ou maus pagadores, baseado em um dataset histórico.

## Objetivo

O objetivo principal deste projeto é desenvolver um modelo preditivo capaz de analisar as características de um novo cliente e determinar a probabilidade dele ser um bom pagador de crédito.

## Dataset

O dataset utilizado é composto por dados de diversos clientes de um banco alemão que solicitaram crédito. Para cada cliente, são fornecidas várias características (atributos) e uma classificação final indicando se o cliente foi um bom ou mau pagador.

## Algoritmo Utilizado

O algoritmo escolhido para este classificador é o Naive Bayes. Baseado no Teorema de Bayes, este algoritmo assume que as características de um cliente são independentes entre si, o que simplifica o cálculo das probabilidades e a classificação.

## Estrutura do Código

O código Python, desenvolvido em um notebook Colab, segue os seguintes passos:

1.  **Importação de Bibliotecas:** Importa as bibliotecas necessárias para manipulação de dados (pandas), divisão de datasets (sklearn.model_selection), utilização do algoritmo Naive Bayes (sklearn.naive_bayes), pré-processamento (sklearn.preprocessing), e avaliação do modelo (sklearn.metrics, yellowbrick.classifier).
2.  **Conhecendo o Dataset:** Carrega o dataset 'Credit.csv', exibe suas dimensões e as primeiras linhas para entendimento da estrutura dos dados.
3.  **Transformação de Atributos:** Realiza o pré-processamento dos dados categóricos, transformando-os em representações numéricas utilizando `LabelEncoder`. Isso é crucial, pois a maioria dos algoritmos de Machine Learning trabalha melhor com dados numéricos. Para garantir a consistência, um `LabelEncoder` separado é criado para cada coluna categórica.
4.  **Divisão entre Treino e Teste:** Divide o dataset em conjuntos de treinamento e teste (70% para treinamento, 30% para teste) para avaliar a performance do modelo em dados não vistos durante o treinamento.
5.  **Treinamento e Acurácia do Modelo:**
    *   Cria uma instância do classificador `GaussianNB`.
    *   Treina o modelo utilizando o conjunto de treinamento.
    *   Realiza previsões no conjunto de teste.
    *   Calcula e exibe a matriz de confusão para visualizar os resultados da classificação.
    *   Calcula e exibe a acurácia do modelo, indicando a porcentagem de acertos.
    *   Utiliza a biblioteca `Yellowbrick` para visualizar a matriz de confusão de forma gráfica, facilitando a interpretação.
6.  **Classificação de um Novo Cliente:**
    *   Carrega os dados de um novo cliente a partir do arquivo 'NovoCliente.csv'.
    *   Aplica as mesmas transformações de atributos (usando os `LabelEncoder`s criados anteriormente) nos dados do novo cliente.
    *   Utiliza o modelo treinado para prever a classe (bom ou mau pagador) do novo cliente.

## Como Executar

Para executar este projeto, siga os passos abaixo:

1.  Clone ou faça o download do notebook Colab (`.ipynb`).
2.  Certifique-se de que os arquivos `Credit.csv` e `NovoCliente.csv` estejam disponíveis no ambiente de execução (por exemplo, no Google Drive se estiver usando o Colab).
3.  Abra o notebook no Google Colab.
4.  Execute as células sequencialmente.

## Bibliotecas Necessárias

*   pandas
*   scikit-learn
*   yellowbrick

Estas bibliotecas podem ser instaladas via pip, caso não estejam disponíveis no seu ambiente. No Colab, a maioria delas já vem pré-instalada. Se precisar instalar, pode usar comandos como:
bash pip install pandas scikit-learn yellowbrick

## Resultados

O modelo treinado obteve uma acurácia de aproximadamente 71% no conjunto de teste. A matriz de confusão visualizada com Yellowbrick indica que o modelo é particularmente eficaz em identificar bons pagadores de crédito.

Ao classificar o novo cliente, o modelo previu que ele seria um bom pagador de crédito.

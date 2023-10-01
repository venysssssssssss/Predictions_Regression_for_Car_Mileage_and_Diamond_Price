# Predictions_Regression_for_Car_Mileage_and_Diamond_Price
Claro, aqui está o código com uma documentação detalhada em formato GitHub:

```python
# Importação de Bibliotecas

```python
import pandas as pd  # Importa a biblioteca pandas para manipulação de dados
from sklearn.linear_model import LinearRegression  # Importa o modelo de regressão linear do scikit-learn
import matplotlib.pyplot as plt  # Importa a biblioteca matplotlib para plotagem de gráficos
import numpy as np  # Importa a biblioteca numpy para operações numéricas
```

# Supressão de Avisos

```python
# Suprima os avisos gerados pelo código
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')  # Suprime os avisos para melhor legibilidade do código
```

# Carregamento do Conjunto de Dados MPG

```python
# URL do conjunto de dados MPG
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"

# Carregamento do conjunto de dados MPG em um DataFrame
df = pd.read_csv(URL)
```

# Exploração Inicial do Conjunto de Dados MPG

```python
# Exibe 5 amostras aleatórias do conjunto de dados
print(df.sample(5))

# Calcula a forma do DataFrame (número de linhas e colunas)
print(df.shape)
```

# Gráfico de Dispersão: Horsepower vs. MPG

```python
# Plota um gráfico de dispersão de Horsepower vs. MPG
plt.scatter(df["Horsepower"], df["MPG"])
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Gráfico de Dispersão: Horsepower vs. MPG")
plt.show()
```

# Treinamento de um Modelo de Regressão Linear

```python
# Cria um conjunto de dados de exemplo para a regressão
x = df["Horsepower"]  # Variável independente (Potência)
y = df["MPG"]         # Variável dependente (Consumo de Combustível)

# Inicializa e treina um modelo de regressão linear
lr = LinearRegression()
lr.fit(x.values.reshape(-1, 1), y)
```

# Previsões com o Modelo de Regressão Linear

```python
# Gera valores de Horsepower para previsão
x_pred = np.linspace(x.min(), x.max(), 100)

# Realiza previsões com o modelo treinado
y_pred = lr.predict(x_pred.reshape(-1, 1))
```

# Visualização dos Resultados da Regressão Linear

```python
# Plota os dados reais e a linha de regressão
plt.scatter(x, y, label="Dados Reais", alpha=0.7)
plt.plot(x_pred, y_pred, color='red', label="Linha de Regressão")
plt.xlabel("Potência (Horsepower)")
plt.ylabel("Consumo de Combustível (MPG)")
plt.title("Regressão Linear: Relação entre Potência e Consumo de Combustível")
plt.legend()

# Adiciona informações adicionais ao gráfico
plt.text(100, 10, f"Score do Modelo: {lr.score(x.values.reshape(-1, 1), y):.2f}", fontsize=12, color='green')
plt.text(100, 35, f"Previsão para Horsepower=100: {lr.predict([[100]])[0]:.2f} MPG", fontsize=12, color='blue')

plt.grid(True)  # Adiciona uma grade para melhor legibilidade
plt.show()
```

# Carregamento do Conjunto de Dados Diamonds

```python
# URL do conjunto de dados Diamonds
URL2 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/diamonds.csv"

# Carregamento do conjunto de dados Diamonds em um DataFrame
df2 = pd.read_csv(URL2)
```

# Treinamento de um Segundo Modelo de Regressão Linear

```python
# Cria variáveis de destino e recursos para o segundo conjunto de dados
target2 = df2["price"]
features2 = df2[["carat", "depth"]]

# Inicializa e treina um segundo modelo de regressão linear
lr2 = LinearRegression()
lr2.fit(features2, target2)
```

# Avaliação do Segundo Modelo de Regressão Linear

```python
# Calcula o score (R²) do segundo modelo
score2 = lr2.score(features2, target2)
print(f"Score do Modelo 2: {score2}")
```

# Previsão com o Segundo Modelo de Regressão Linear

```python
# Realiza uma previsão utilizando o segundo modelo
prediction2 = lr2.predict([[0.3, 60]])
print(f"Previsão para carat=0.3 e depth=60: {prediction2}")
```

# Análise Visual da Relação entre Características dos Diamantes e Preço

```python
# Plota gráficos de dispersão para analisar a relação entre características e preços dos diamantes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(df2["carat"], df2["price"], alpha=0.5)
plt.xlabel("Peso em Quilates (Carat)")
plt.ylabel("Preço (em Dólares)")
plt.title("Relação entre Peso em Quilates e Preço dos Diamantes")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(df2["depth"], df2["price"], alpha=0.5)
plt.xlabel("Profundidade (Depth)")
plt.ylabel("Preço (em Dólares)")
plt.title("Relação entre Profundidade e Preço dos Diamantes")
plt.grid(True)

plt.tight_layout()
plt.suptitle("Análise da Relação entre Características dos Diamantes e Preço", fontsize=16)
plt.subplots_adjust(top=0.85)  # Ajusta a posição do título principal
plt.show()
```

# Histograma da Distribuição de Preços dos Diamantes

```python
# Plota um histograma da distribuição de preços dos diamantes
plt.figure(figsize=(10, 6))
plt.hist(df2["price"], bins=30, edgecolor='k', color='skyblue')
plt.xlabel("Preço (em unidades monetárias)")
plt.ylabel("Contagem de Diamantes")
plt.title("Distribuição de Preços de Diamantes")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

# Adiciona uma linha vertical para mostrar a média de preços
mean_price = df2["price"].mean()
plt.axvline(x=

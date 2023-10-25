import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Carregando a base de dados Iris
data_iris = pd.read_csv("iris.csv")
X_iris = data_iris.iloc[:, 0:4]  # Características
y_iris = data_iris.iloc[:, 4]   # Rótulos

# Carregando a base de dados Car
data_car = pd.read_csv("car.csv")

# Codificando os valores categóricos em números
le = LabelEncoder()
data_car_encoded = data_car.apply(le.fit_transform)

X_car = data_car_encoded.iloc[:, 0:6]  # Características
y_car = data_car_encoded.iloc[:, 6]    # Rótulos

# Dividindo os dados em treinamento e teste para ambas as bases
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
X_train_car, X_test_car, y_train_car, y_test_car = train_test_split(X_car, y_car, test_size=0.2, random_state=42)

# Criando e treinando uma árvore de decisão para a base de dados Iris
clf_iris = DecisionTreeClassifier()
clf_iris.fit(X_train_iris, y_train_iris)

# Criando e treinando uma árvore de decisão para a base de dados Car
clf_car = DecisionTreeClassifier()
clf_car.fit(X_train_car, y_train_car)

# Avaliando o desempenho para a base de dados Iris
y_pred_iris = clf_iris.predict(X_test_iris)
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
print(f"Score para a base de dados Iris: {accuracy_iris:.2f}")

# Avaliando o desempenho para a base de dados Car
y_pred_car = clf_car.predict(X_test_car)
accuracy_car = accuracy_score(y_test_car, y_pred_car)
print(f"Score para a base de dados Car: {accuracy_car:.2f}")

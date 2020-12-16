from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import numpy as np

# Carregar o dataset
data = load_breast_cancer()

# Organizar nossos dados
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

frame = pd.DataFrame(np.c_[data['data'], data['target']],
                  columns= np.append(data['feature_names'], ['target']))

df = frame.melt(['target'], var_name='cols',  value_name='vals')
g = sns.FacetGrid(df, col='cols', hue="target", palette="Set1", col_wrap=4)
g = (g.map(sns.distplot, "vals", hist=True, ))

# Olhando para os nossos dados
print(label_names)
print('Class label = ', labels[0])
print(feature_names)
print(features[0])

# Dividir nossos dados
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)

# Inicializar nosso classificador
gnb = GaussianNB()

# Treinar nosso classificador
model = gnb.fit(train, train_labels)

# Fazer previsões
preds = gnb.predict(test)
print(preds)

# Avaliar a precisão
print(accuracy_score(test_labels, preds))

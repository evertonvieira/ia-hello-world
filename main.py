
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from logitboost import LogitBoost

##carrega o dataset presente na própra biblioteca scikit-learn
data = load_breast_cancer()
X = data.data
y = data.target_names[data.target]
n_classes = data.target.size

# Mistura os dados e divida-os em amostras de treinamento / teste
test_size = 1 / 3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    shuffle=True, stratify=y,
                                                    random_state=0)

# aplica oalgoritmo de aprendizado de máquina para visualização baseado na 
# incorporação de vizinhança e projeta os dados e 2D
tsne = TSNE(n_components=2, random_state=0)
X_train_tsne = tsne.fit_transform(X_train)

##seta os atributos do gráfico
sns.set(style='darkgrid', palette='colorblind', color_codes=True)

#monta o gráfico de visualização do conjunto de treinamento
plt.figure(figsize=(10, 8))
mask_benign = (y_train == 'benign')
mask_malignant = (y_train == 'malignant')

plt.scatter(X_train_tsne[mask_benign, 0], X_train_tsne[mask_benign, 1],
           marker='s', c='g', label='Cancer benigno', edgecolor='k', alpha=0.7)
plt.scatter(X_train_tsne[mask_malignant, 0], X_train_tsne[mask_malignant, 1],
           marker='o', c='r', label='Cancer maligno', edgecolor='k', alpha=0.7)
plt.title('Gráfico t-SNE dos dados de treinamento')
plt.xlabel('1st embedding axis')
plt.ylabel('2nd embedding axis')
plt.legend(loc='best', frameon=True, shadow=True)
plt.tight_layout()
plt.show()

#carrega o nosso algoritmo de classificação baseada e árvore de decisão (árvores de decisão com profundidade 1,
lboost = LogitBoost(n_estimators=200, random_state=0)

# realiza o treinamento dos dados
lboost.fit(X_train, y_train)

# faz a validação dos dados de treino e teste (tenta prevê se o cancer é maligno ou benigno)
y_pred_train = lboost.predict(X_train)
y_pred_test = lboost.predict(X_test)

# calcula a porcentagem de acurácia
accuracy_train = (accuracy_score(y_train, y_pred_train) * 100)
accuracy_test = (accuracy_score(y_test, y_pred_test) * 100)

print('Training accuracy: %.1f' % accuracy_train)
print('Test accuracy:     %.1f' % accuracy_test)

##monta o gráfico comparando a presição dos treinos
iterations = np.arange(1, lboost.n_estimators + 1)
staged_accuracy_train = list(lboost.staged_score(X_train, y_train))
staged_accuracy_test = list(lboost.staged_score(X_test, y_test))

plt.figure(figsize=(10, 8))
plt.plot(iterations, staged_accuracy_train, label='Training', marker='.')
plt.plot(iterations, staged_accuracy_test, label='Test', marker='.')

plt.xlabel('Iteração')
plt.ylabel('Acurácia')
plt.title('Acurácia do conjunto durante cada iteração de treinamento')
plt.legend(loc='best', shadow=True, frameon=True)

plt.tight_layout()
plt.show()
plt.close()
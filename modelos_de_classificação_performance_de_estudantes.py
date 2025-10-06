
"""Modelos de Classificação- Performance de Estudantes


Original file is located at
    https://colab.research.google.com/drive/1iykfJNMOMyPRV-pQCDHidV9nbsl_69JH
"""

# Modelo Machine Learning - Performance dos Estudantes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib 
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn import tree
df = pd.read_csv("student_data_tratados.csv")

# Convert categorical features to numerical using one-hot encoding


df.info()

conditions = [
    (df['media_final'] <= 5),
    (df['media_final'] > 5) & (df['media_final'] <= 8),
    (df['media_final'] > 8) & (df['media_final'] <= 11),
    (df['media_final'] > 11) & (df['media_final'] <= 13),
    (df['media_final'] > 13)
]
values = ['Pessima','Ruim','Médio', 'Bom', 'Excelente']
df['classificacao'] = np.select(conditions, values, default = 'Sem classificação')

x = df.drop(['classificacao','media_final','Nota_2T','Nota_3T_Final'], axis = 1)
y = df['classificacao']

padronizacao = StandardScaler()
preprocessador_padronizacao = ColumnTransformer(verbose_feature_names_out = False, remainder = 'passthrough', transformers = [('Ordinal', OrdinalEncoder(), make_column_selector(dtype_include=['object'])),
                                                                                                                 ('Padroninação', padronizacao, make_column_selector(dtype_include=['int']))])

atrib_pre_Padronizacao = pd.DataFrame(preprocessador_padronizacao.fit_transform(x), columns= preprocessador_padronizacao.get_feature_names_out())
atrib_pre_Padronizacao = atrib_pre_Padronizacao[x.columns.values]

x_train, x_test, y_train, y_test = train_test_split(atrib_pre_Padronizacao, y, test_size = 0.2, random_state = 42)

# Modelo 1
### APLICAR VALIDAÇÃO CRUZADA
from sklearn.ensemble import RandomForestClassifier

classificador_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight= 'balanced')
classificador_rf.fit(x_train, y_train)

y_pred_rf = classificador_rf.predict(x_test)
taxa_de_acerto_rf = accuracy_score(y_test, y_pred_rf)

print(f'Random Forest: {taxa_de_acerto_rf*100:.2f}% accuracy')

#Modelo 1 - Random Florest

confusion_mat = confusion_matrix(y_test, y_pred_rf)

report = classification_report(y_test, y_pred_rf)
print(report)
sns.heatmap(confusion_mat,
            square=True,
            annot=True,
            fmt='d',
            xticklabels=classificador_rf.classes_,
            yticklabels=classificador_rf.classes_
)

#Modelo 2 - KNN
from sklearn.neighbors import KNeighborsClassifier

classificador_knn = KNeighborsClassifier(n_neighbors=5)
classificador_knn.fit(x_train, y_train)

# Determinando os valores de K para análise
parametros = {'n_neighbors': range(1,21)}
knn_grid = KNeighborsClassifier()

# GridSearch com validação cruzada
grid = GridSearchCV(knn_grid, parametros, cv=5)
grid.fit(x_train, y_train)

# Apresentando as informações
print(f"Melhor valor de K: {grid.best_params_}")
print(f"Melhor classificador: {grid.best_estimator_}")
print(f"Melhor score (médio): {grid.best_score_}")

best_k = grid.best_params_['n_neighbors']
# Performance KNN [Padronização]
classificador_knn = KNeighborsClassifier(n_neighbors=best_k)
classificador_knn.fit(x_train, y_train)

previsoes_knn = classificador_knn.predict(x_train)
taxa_de_acerto_KNN = accuracy_score(y_train, previsoes_knn)

print(f'KNN: {taxa_de_acerto_KNN*100:.2f}% accuracy')

#Modelo 3 - Decision Tree 

elfo = tree.DecisionTreeClassifier() 
elfo = elfo.fit(x_train, y_train)
previsoes_elfo = elfo.predict(x_test)
taxa_de_acerto_dt = accuracy_score(y_test, previsoes_elfo)
print(f'decision_tree: {taxa_de_acerto_dt*100:.2f}% accuracy')

joblib.dump(classificador_rf, 'modelo_random_forest.pkl')
joblib.dump(atrib_pre_Padronizacao.columns.tolist(), 'colunas_modelo.pkl')






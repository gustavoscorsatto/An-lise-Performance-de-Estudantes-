# 📚 Previsão e Análise da Performance de Estudantes com Machine Learning

Este projeto utiliza Machine Learning para prever o desempenho final de estudantes e identificar os fatores que mais influenciam no seu desempenho acadêmico.

O objetivo é fornecer insights valiosos para entender quais são os principais fatores e como podemos melhora-los a fim de obter uma perfomance acadêmica.

## 🎯 Objetivos

Classificação: Construir um modelo de Classificação capaz de classificar a nota final (media_final) em: "Péssima, "Ruim", "Média", "Boa" e "Excelente"  

Análise de Fatores: Realizar uma Análise Exploratória de Dados (EDA) para entender as correlações e os principais fatores que influenciam na performance.

## 💾 Fonte de Dados
Dataset: Dados Performance de Estudantes (Tratados)

Descrição: O conjunto de dados contém informações sobre estudantes de ensino médio, abrangendo notas de três períodos (Nota 1º, 2º e 3º trimestre) e variáveis demográficas, socioeconômicas e comportamentais (ex: tempo de estudo, apoio familiar, número de reprovações).

Variável Alvo (Target): media_final (Nota final do terceiro período/semestre).

## 🚀 Aspectos Técnicos 

1. Pré-processamento de Dados:
- Tratamento de missing values e outliers.
- Conversão de variáveis categóricas (One-Hot Encoding).

2. Análise Exploratória (EDA):
- Visualização da distribuição das notas.
- Análise de correlação entre as notas (Nota_1T, Nota_2T, Nota_3T) e outros fatores (ex: tempo_estudo, relacao_familiar, tempo_livre).

3. Modelagem e Treinamento:
- *Modelos testados: Random Forest Classifier, KNN Classifier, Decision Tree Classifier.
- Otimização com Grid Search Cross-Validation.

4. Avaliação:
Seleção do modelo com melhor desempenho nas métricas de classificação.

- Acurácia (Geral) do Modelo Random Forest (Escolhido):	71.43%
- acro Avg	0.72 (Precision) / 0.66 (Recall) / 0.67 (F1-Score)
- Weighted Avg	0.73 (Precision) / 0.71 (Recall) / 0.70 (F 1-Score)

5. Criação de um Dashboard Interativo (arquivo app.py)



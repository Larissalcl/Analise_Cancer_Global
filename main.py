#  Objetivo: Analisar as variáveis globais de pacientes com cancer e entender a correlação.

#Passo 1: Importar as bibliotecas e ajustar exibição
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width', None) #Ajustar a exibição para ocupar toda a largura disponível.
pd.set_option('display.max_columns', None)

#Passo 2: Visualizar os dados e informações
df = pd.read_csv('global_cancer_patients_2015_2024.csv')
print(df.head().to_string)

print('Linhas e colunas totais: ', df.shape)
print(df.info())

#Passo 3: Limpeza dos dados
# Tratar valores nulos
linhas_com_nulos = df[df.isnull().any(axis=1)]
print('Linha com valores nulos: \n ', linhas_com_nulos)
print('Análise de dados nulos:\n', df.isnull().sum().sum())
print('Análise de dados nulos:\n',df.isnull().mean())

#Tratar valores duplicados
df.drop_duplicates() #Remover linhas duplicadas.
print('Análise de dados duplicados: \n', df.duplicated().sum())

#Retirar coluna que possam identificar o paciente
df = df.drop(columns=['Patient_ID'])
print('\n Df ajustado: \n', df.head())

#Passo 4: Tratamento dos dados - Outliers
def calcular_limites_iqr(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1

    limite_baixo = Q1 - 1.5 * IQR
    limite_alto = Q3 + 1.5 * IQR

    print(f'\nLimites IQR para {coluna}:', limite_alto, limite_baixo)
    return limite_baixo, limite_alto

limite_baixo, limite_alto = calcular_limites_iqr(df, 'Genetic_Risk')
limite_baixo, limite_alto = calcular_limites_iqr(df, 'Air_Pollution')
limite_baixo, limite_alto = calcular_limites_iqr(df, 'Alcohol_Use')
limite_baixo, limite_alto = calcular_limites_iqr(df, 'Smoking')
limite_baixo, limite_alto = calcular_limites_iqr(df, 'Obesity_Level')


# Passo 5: Análise Exploratória dos Dados
print('\nEstatística dos campos: \n', df[['Age', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Treatment_Cost_USD', 'Survival_Years']].describe())
print('\n Correlação das variáveis:\n', df[['Age','Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Treatment_Cost_USD', 'Survival_Years']].corr())

# Passo 6: Visualização dos Dados
plt.figure(figsize=(13, 6))
sns.countplot(x='Cancer_Type', hue='Cancer_Type', palette='viridis', data=df, legend=False)
plt.title('Distribuição por Tipo de Câncer', fontsize=15)
plt.xlabel('Tipo de Câncer', fontsize=13)
plt.ylabel('Frequência', fontsize=13)

x = df['Gender'].value_counts().index # gerar o index no eixo x
y = df['Gender'].value_counts().values
plt.figure(figsize=(13, 6))
cores = ['#66B3FF','#FF9999','#99FF99']
plt.pie(y,labels=x, autopct='%.1f%%', startangle=90, colors=cores)
plt.title('Distribuição de Gêneros')

x = df['Cancer_Stage'].value_counts().index # gerar o index no eixo x
y = df['Cancer_Stage'].value_counts().values
plt.figure(figsize=(13, 6))
cores = sns.color_palette("viridis")
plt.pie(y,labels=x, autopct='%.1f%%', startangle=90, colors=cores)
plt.title('Distribuição por Tipos de Câncer')

plt.figure(figsize=(13, 6))
sns.countplot(x='Cancer_Type', hue='Gender', palette='viridis', data=df, legend=True)
plt.title('Tipo de Câncer por Gênero', fontsize=15)
plt.xlabel('Tipo de Câncer', fontsize=13)
plt.ylabel('Frequência', fontsize=13)


plt.figure(figsize=(13, 6))
sns.countplot(x='Country_Region',  hue='Country_Region', palette='viridis', data=df, legend=False)
plt.title('Distribuição por País', fontsize=15)
plt.xlabel('País', fontsize=13)
plt.ylabel('Frequência', fontsize=13)


plt.figure(figsize=(13, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues')
plt.title('Correlation Metrics of Numerical Cols', fontsize=15)
plt.show()

plt.figure(figsize=(10,6))
x = df['Cancer_Type'].value_counts().values # gerar o index no eixo x
y = df['Treatment_Cost_USD'].value_counts().values # gerar os valores no eixo y

# Salvar dataframe
df.to_csv('global_cancer_patients', index=False)
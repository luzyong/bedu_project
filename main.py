import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

dataset = pd.read_csv("afluenciametro.csv")

print(dataset.columns)

dataset= dataset[dataset['linea']=='Línea B']


print("Mes:\n"+str(dataset['mes'].value_counts()))
print("Afluencia promedio: "+str(dataset['afluencia'].mean()))
print("Año promedio: "+str(dataset['anio'].mean()))


plt.figure(figsize=(8, 6))
sns.boxplot(x=dataset['afluencia'])
plt.title('Boxplot de Afluecias')
plt.show()

columna = dataset['afluencia']

Q1 = columna.quantile(0.25)
Q3 = columna.quantile(0.75)


IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

dataset = dataset[(dataset['afluencia'] >= lower_bound) & (dataset['afluencia'] <= upper_bound)]

print("Mes:\n"+str(dataset['mes'].value_counts()))
print("Afluencia promedio: "+str(dataset['afluencia'].mean()))
print("Año promedio: "+str(dataset['anio'].mean()))


plt.figure(figsize=(16, 8))
sns.barplot(data=dataset, x='estacion', y='afluencia', hue='mes', palette='viridis')

plt.title('Afluencia por Estación y Mes')
plt.xlabel('Estación')
plt.ylabel('Afluencia')
plt.xticks(rotation=45, ha='right')  
plt.xticks(fontsize=5)
plt.show()





plt.figure(figsize=(12, 8))
sns.boxplot(x='estacion', y='afluencia', data=dataset, palette='viridis')
plt.title('Boxplots de Afluencia por Estación')
plt.xlabel('Estación')
plt.ylabel('Afluencia')
plt.xticks(rotation=45, ha='right',fontsize=5)  
plt.show()

plt.figure(figsize=(12, 8))
sns.violinplot(x='estacion', y='afluencia', data=dataset, palette='viridis', inner='quartile')
plt.title('Violinplot de Afluencia por Estación')
plt.xlabel('Estación')
plt.ylabel('Afluencia')
plt.xticks(rotation=45, ha='right',fontsize=5)  
plt.show()

plt.figure(figsize=(12, 8))

for mes in dataset['mes'].unique():
    sns.kdeplot(dataset[dataset['mes'] == mes]['afluencia'], label=mes, shade=True)

plt.title('Gráfica de Densidad de Afluencia por Mes')
plt.xlabel('Afluencia')
plt.ylabel('Densidad')
plt.legend(title='Mes', bbox_to_anchor=(1, 1))
plt.show()


correlation_matrix = dataset.groupby(['estacion'])[['anio', 'afluencia']].corr(method='spearman').unstack()


print(correlation_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Correlación'})
plt.title('Mapa de Correlación entre Afluencia, anio y Estacion')
plt.show()




n_bootstrap_samples = 1000

bootstrap_statistics = []

for _ in range(n_bootstrap_samples):

    bootstrap_sample = resample(dataset['afluencia'])
    statistic = np.mean(bootstrap_sample)
    bootstrap_statistics.append(statistic)
ci_lower = np.percentile(bootstrap_statistics, 2.5)
ci_upper = np.percentile(bootstrap_statistics, 97.5)

print(f'Intervalo de confianza del 95%: ({ci_lower}, {ci_upper})')

plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_statistics, bins=30, kde=True)
plt.title('Histograma de Afluencia')
plt.xlabel('Afluencia')
plt.ylabel('Frecuencia')
plt.show()

se_bootstrap = np.std(bootstrap_statistics)
print(f'Error estándar de la afluencia: {se_bootstrap}')
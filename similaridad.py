import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

df = pd.read_csv(os.path.join("ds_Zika.csv"))
df = df.drop('Output', axis=1)
resultados = np.zeros((len(df.columns), len(df.columns)))

for i in range(len(df.columns)):
    for j in range(len(df.columns)):
            resultados[i, j] = np.mean(df.iloc[:, i] == df.iloc[:, j])

print('\nPares de atributos duplicados:')
for i, j in it.combinations(range(len(df.columns)), 2):
    # el redonde al cuarto valor decimal no es casual, la minima diferencia que
    # se puede obtener entre dos atributos es de 1 / 5378 = 0.0002
    if round(resultados[i, j], 4) == 1:
        print(str(i).ljust(3) + ': ' + str(j))

plt.figure()
x = sns.heatmap(resultados)
plt.show()

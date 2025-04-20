import pandas as pd

df = pd.read_csv('titanic.csv')
df_v2 = df[['Pclass', 'Sex', 'Age']]
df_v2.to_csv('titanic.csv', index=False)
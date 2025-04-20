import pandas as pd

df = pd.read_csv('titanic.csv')
df['Age'] = df['Age'].fillna(df['Age'].mean())
df.to_csv('titanic.csv', index=False)
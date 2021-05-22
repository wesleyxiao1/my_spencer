import pandas as pd

df = pd.read_csv('tracked_groups.csv')
print(df.head(10))
df = df.drop('index', 1)
df['frameID'] = df['frameID'].astype(int)
df['groupID'] = df['groupID'].astype(int)

df.to_csv('tracked_groups_mod.csv', index=False)
import pandas as pd

df = pd.read_csv('/Users/alanvidal/Desktop/BootCamp-CD/AP/ventas.csv')
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()

print(f"\n{df.head()}")

df_subset=df[['producto', 'precio']]
print(f"\n{df_subset}")
df_subset2=df_subset[df_subset['precio'] > 50]
print(f"\n{df_subset2}")

df_subset2.to_csv('df_subset2.csv', index=False)

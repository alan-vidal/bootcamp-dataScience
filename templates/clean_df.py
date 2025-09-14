import pandas as pd
#import pyarrow
import matplotlib.pyplot as plt

PATH = "/data.csv"
df = pd.read_csv(PATH) #engine="pyarrow", dtype_backend="pyarrow"

df.columns = df.columns.str.strip()
df.columns = df.columns.str.upper()

#EXPLORATORY DATA ANALYSIS
#Display the first few rows of the dataset
print(df.head())
#Summary statistics
print(df.describe())
#Information about the dataset
print(df.info())

#Identify duplicates
print(df.duplicated().sum())
#Remove duplicates
df = df.drop_duplicates()

#Check for missing values
print(df.isnull().sum())
#Drop rows with missing valiues and place it in a new variable "df_cleaned"
# => df_cleaned = df.dropna()
#Fill missing values with mean for numerical data and place it ina new variable called df_filled
# => df_filled = df.fillna(df.mean())

df['column_name'] = df['item'].fillna(df['column_name'].mean())
#df['item'] = df['item'].fillna(df["item"].mode()[0])

#DATA TYPES & CONVERSION
#Convert 'Column1' to float
df['Column1'] = df['Column1'].astype(float)

# QUANTILES
column_name = 'CANTIDAD'
Q1 = df[column_name].quantile(0.25)
Q3 = df[column_name].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
df_filtered =df[df["column_name"].between(lower_bound, upper_bound)]

plt.boxplot(df_filtered[column_name],  showmeans=True)
plt.title(f'Boxplot of {column_name}')
plt.ylabel(f'Value of {column_name}')
plt.grid(color = 'blue', linestyle = '--', linewidth = 0.5)
plt.show()

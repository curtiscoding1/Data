import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


pd.set_option('display.max_columns', None) # enables the option to show every column or row specified
pd.set_option('display.max_rows', None)



df = pd.read_csv(r"C:\Users\Curt\Downloads\Diabetes.csv")

df = df.rename(columns={'Glucose' : 'Glucose Level', 'Insulin' : 'Insulin Level'})

#An overview of the dataset

print(df.head())
print(f"The columns are {df.columns}")
print(df.describe(include='all'))


corr, pval = stats.pearsonr(df['Age'], df['BMI'])

print(f"Pearson's correlaton coefficient between Age and BMI is {corr}")

#find NaNs (there should be none. just verifying)
print("Here are the null values:")
print(df.isnull().sum())

#the dataset uses 0 rather than NaN

zerovalues = {}

for col in df.columns:
    zerovalues[col] = df[df[col] == 0].size

print(f"Here are the number of zero values in each column {zerovalues}")

#from the list of zeros, we dont want them in any column other than pregnancies, pedigree, age and outcome

dropzeroes = ["Glucose Level", "BloodPressure", "SkinThickness", "Insulin Level", "BMI"]

for col in dropzeroes:
    df = df[df[col] > 0]

def categorise_age(age): # a groupby function
    if age > 65:
        return 'd'
    if age > 44:
        return 'c'
    if age > 25:
        return 'b'
    return 'a'

for group, frame in df.groupby(categorise_age):
    print(f"There are {str(len(frame))} people in age group {group}")

#visualising some of the data
fig, ax = plt.subplots()
df = df[df['BMI']!=0]
plt.scatter(df['BMI'], df['Age'])
plt.show()

# predict outcome using scikitlearn







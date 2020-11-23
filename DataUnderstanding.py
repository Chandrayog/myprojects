###
# The objective of the dataset is to diagnostically predict whether or not a patient has diabetes,
# based on certain diagnostic measurements included in the dataset. Data is from Females older than 21 age.

### Below Steps Performed: ###

# 1. Exploratory data analysis
# 2. Data Understanding


#######

### Importing Python libraries
import pandas as pd
import numpy as np

### Plots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py

#from time import time
#from IPython.display import display # Allows the use of display() for DataFrames
from scipy.stats import skew
from scipy.stats import kurtosis
import pylab as p


#ignore warning messages
import warnings
warnings.filterwarnings('ignore')


### setting output display options
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

### Load the dataset
df = pd.read_csv("/Users/chandrayogyadav/Desktop/Diabetes.csv")

# Summary of the records
print("This dataset has {} samples with {} features each.".format(df.shape[0], df.shape[1]))


# Display the first 10 record
print("\nDisplay the first 10 record")
print(df.head(10))

# Describe the data
print("Statistical description of dataset\n--------------------------------------")
print(df.describe())

#outF = open("/Users/chandrayogyadav/Desktop/myOutFile.txt", "w")
#outF.writelines(df.describe())
#outF.close()


print('Note\n-----')
print('All values are numerical')
print("'Outcome' is the target variable that can have only binary value(0 or 1)\n\n")
print(df.info())


### Overview of the data set
num_records = df.shape[0]
n_with_diabetes = df[df["Outcome"]==1].shape[0]
n_without_diabetes = df[df["Outcome"]==0].shape[0]
percent = (n_with_diabetes*100)/float(num_records)

print("\nTotal number of individuals: {}".format(num_records))
print("Individuals with diabetes: {}".format(n_with_diabetes))
print("Individuals without diabetes: {}".format(n_without_diabetes))
print("Percentage of individuals with diabetes: {:.2f}%\n".format(percent))

# plot the target variable
sns.countplot(df['Outcome'],label="Count")


#### Understand the Data distribution and skewness of  dataset ###

print('Skewness of all columns of dataset is:\n---------------')
print(df.skew(axis = 0, skipna = True))
df.hist(alpha=0.5, figsize=(16, 10))

### Skewness of dataset ###
plt.hist(df, bins=20)


### Missing and  Null values

print("Statistical description of dataset\n--------------------------------------")
print(df.describe())

# Check the values now
df.head()

#Checking Null Values
print(df.isnull())

# Or
df.isna()


# Replace 0 with NaN
df.replace(0, np.NaN, inplace=True)

# sum of null population in all columns
df.isnull().sum()

# Reverting Target variable
df['Outcome'].fillna(0, inplace=True)

### Understand the target variable relationship between variable
#sns.boxplot(data=df)

sns.scatterplot(x="Age", y="Glucose", data=df)


#### Correlation between varibles ####
pearsoncorr = df.corr(method='pearson')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(pearsoncorr, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(pearsoncorr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})



#### With values of  coefficients ###
pearsoncorr = df.corr(method='pearson')
pearsoncorr
sns.heatmap(pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)


### Method 1: Check extreme outliers
max_threshold = df['Glucose'].quantile(0.98)
print(max_threshold)

outlier = df[df['Glucose'] > max_threshold]
print(outlier['Glucose'])

#### Method 2: Finding Outliers using Tukey IQR method

def find_outliers(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    ind = list(x.index[(x < lower) | (x > ceiling)])
    outlier = list(x[ind])
    return ind, outlier


out_inx, out_val = find_outliers(df['Insulin'])
print(np.sort(out_val))


# Outliers between Target Variable and Strong related variable
#### Comparing distributions, the centre, spread and overall range  w.r.t two binary outcome(0/1)

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10,10))

df.boxplot(column='Pregnancies', by='Outcome',ax=axes[0,0])
df.boxplot(column='Glucose', by='Outcome', ax=axes[0,1])
df.boxplot(column='BloodPressure', by='Outcome',ax=axes[1,0])
df.boxplot(column='SkinThickness', by='Outcome', ax=axes[1,1])
df.boxplot(column='Insulin', by='Outcome',ax=axes[2,0])
df.boxplot(column='BMI', by='Outcome', ax=axes[2,1])
df.boxplot(column='DiabetesPedigreeFunction', by='Outcome',ax=axes[3,0])
df.boxplot(column='Age', by='Outcome', ax=axes[3,1])

fig.tight_layout()



### Diabetic patients GLucose and insulin

sns.scatterplot(x="Outcome", y="Glucose", data=df)
sns.scatterplot(x="Outcome", y="Insulin", data=df)

### Realtionship for Diabetic Pateints, Outcome = 1
p = df[df['Outcome']==1].hist(figsize = (20,20))
plt.title('Diabetes Patient')

### Relationship of all variables with Target variable ###

sns.pairplot(df, vars=["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction", "Age"],hue="Outcome")
plt.title("Pairplot of Variables by Outcome")

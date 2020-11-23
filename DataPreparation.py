###
# The objective of the dataset is to diagnostically predict whether or not a patient has diabetes,
# based on certain diagnostic measurements included in the dataset. Data is from Females older than 21 age.

##Current Steps Performed:


### Data Preparation

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score


#ignore warning messages
import warnings
warnings.filterwarnings('ignore')

#### Data Preparation ####


### setting output display options
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

### Load the dataset
df = pd.read_csv("/Users/chandrayogyadav/Desktop/Diabetes.csv")

# Describe the data
print("Statistical description of dataset\n--------------------------------------")
print(df.describe())



### Treatment of Missing, NUll or NaN values. ####


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

# Check the values now
df.head()


# Reverting Target variable
df['Outcome'].fillna(0, inplace=True)

# sum of null population in all columns
df.isnull().sum()

# fill missing values with mean column values
df.fillna(df.median(), inplace=True)

# Check now
df.head()


### Renaming the column ###
df.rename(columns={'DiabetesPedigreeFunction':'DPF'}, inplace=True)
df.info()





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


## Dropping outliers Wil perform later after verification of feautres ###

df.drop(out_inx, inplace=True)
df.head(10)


df.head(10)



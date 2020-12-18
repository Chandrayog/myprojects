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

### Importing scikit-learn libraries
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score,confusion_matrix,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import neighbors, datasets, preprocessing


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

#### Correlation between varibles ###
#### With values of  coefficients ###
pearsoncorr = df.corr(method='pearson')
pearsoncorr
sns.heatmap(pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)



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



### Pairplot Relationship of all variables with Target variable ###

sns.pairplot(df, vars=["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction", "Age"],hue="Outcome")
plt.title("Pairplot of Variables by Outcome")


###-------------******** Data Preparation ********-------------------####


#### Start with replacing 0 values ####

# Replace 0 with NaN
df.replace(0, np.NaN, inplace=True)

# Check the values now
df.head()


# Reverting Target variable
df['Outcome'].fillna(0, inplace=True)

# sum of null population in all columns
df.isnull().sum()

# fill missing values with for the columns in accordance with their distribution
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)

# To replace every othe values
df.fillna(df.median(), inplace=True)

# Check now
df.head()


### Renaming the column ###
df.rename(columns={'DiabetesPedigreeFunction':'DPF'}, inplace=True)
df.info()

####--------******* Feature Selection/Importance ********------------#####

### 1. Method 1- Variance method ###
df.var()


### Method 2. Wrapper Method- Recursive Feature elimination


### Creating input and target variables set ###
target = 'Outcome'
X = df.loc[:, df.columns != target]
Y = df.loc[:, df.columns == target]

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=8)

### use a baseline classifier logistic regression
log_reg = LogisticRegression()
lr_model = log_reg.fit(x_train,y_train)

## Call RFE Wrapper
rfe = RFE(estimator=lr_model, step=1)
### fit the rfe function for ranking
rfe = rfe.fit(x_train, y_train)

print('No of feautres : %d' %rfe.n_features_)
print('Selected feautres : %s' %rfe.support_)
print('Feautres rank   : %s' %rfe.ranking_)

## Print features with ranking
selected_rfe_features =pd.DataFrame({'Feature':list(x_train.columns), 'ranking':rfe.ranking_})
print(selected_rfe_features.sort_values(by='ranking', ascending=True))


### Embedded Method - CART Regression Feature Importance
cart_model = DecisionTreeRegressor()
cart_tree = cart_model.fit(X, Y)
imp_feature = cart_tree.feature_importances_

#check summary
feature_val = pd.Series(imp_feature, index = X.columns)

# draw chart of important feature
feature_val.nlargest(10).plot(kind='bar')
plt.show()


###-----------********* Model Creation *******------###

## 1st iteration using Logisitic Regression model with all features

### Model 1 - Logistic Regression
log_reg = LogisticRegression()
lr_model = log_reg.fit(x_train,y_train)
y_pred = lr_model.predict(x_test)

### scores
ac = accuracy_score(y_test, y_pred)
f_score = f1_score(y_test, y_pred)

print('Baseline model accuracy score with all feature:', ac)
print('Baseline model F1 score with all feature:', f_score)

### Check confusion matrix
plot_confusion_matrix(lr_model, x_test, y_test)
plt.show()


### We will select only top 5 features identified in feautre selection step above, thus removing 3 least important feature below ###

##### Dropping not required columns  #####
df.drop(["Pregnancies", "BloodPressure", "SkinThickness"], 1 , inplace = True)

df.head()

# reassign data to train and test sets
X = df.loc[:, df.columns != target]
Y = df.loc[:, df.columns == target]


x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=8)


### logistic regression with selected feature
log_reg = LogisticRegression()
lr_model = log_reg.fit(x_train,y_train)
y_pred = lr_model.predict(x_test)

### scores
ac = accuracy_score(y_test, y_pred)
f_score = f1_score(y_test, y_pred)

print('Baseline model accuracy score with selected features:', ac)
print('Baseline model F1 score with selected features:', f_score)


### Check confusion matrix
plot_confusion_matrix(lr_model, x_test, y_test)  # doctest: +SKIP
plt.show()



######  Model 3 - Decision Tree- Random Forest Classifier #######

rfc = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)

### Train the classifier
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)

rfc_acc= accuracy_score(y_test, y_pred_rfc)
rfc_f_score= f1_score(y_test, y_pred_rfc)

print('Random Forest Accuracy:', rfc_acc)
print('Random Forest F score:', rfc_f_score)

### graph for feature importance
f_imp= pd.Series(rfc.feature_importances_, index=x_train.columns)
f_imp.nlargest(10).plot(kind='barh')
plt.show()


##### Model 4 - CART -  Decision Tree Model ####

DT = DecisionTreeClassifier()

### fiting the model
DT.fit(x_train, y_train)

####prediction
y_pred = DT.predict(x_test)


### scores
ac = accuracy_score(y_test, y_pred)
f_score = f1_score(y_test, y_pred)

print('CART model accuracy score:', ac)
print('CART model F1 score:', f_score)
print(classification_report(y_test,y_pred))
#print(DT.score(x_train,y_train))

#Plot the confusion matrix
sns.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True, fmt='g')
plt.show()



#### Method 5 Using KNN Classification #####

### call the instance of KNN
knn = KNeighborsClassifier()

### Transform the data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

### Fit the model
knn = neighbors.KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

### Print the results
print('KNN Accuracy with selected features:' , accuracy_score(y_test, y_pred))
print('KNN F1 Score with selected features:' , f1_score(y_test, y_pred))

### Check confusion matrix
plot_confusion_matrix(knn, x_test, y_test)
plt.show()

#####-----------------********* End *************---------------------#######



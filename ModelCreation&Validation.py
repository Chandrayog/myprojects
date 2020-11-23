###
# The objective of the dataset is to diagnostically predict whether or not a patient has diabetes,
# based on certain diagnostic measurements included in the dataset. Pateints data is from Females older than 21 age.

##Current Steps Performed:

## Feature Extraction ##


#######

### Importing Python libraries
import pandas as pd
import numpy as np

### Importing scikit-learn libraries
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score,confusion_matrix,plot_confusion_matrix
from sklearn.model_selection import train_test_split


### Plots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py


#ignore warning messages
import warnings
warnings.filterwarnings('ignore')

### setting output display options
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

### Load the dataset
df = pd.read_csv("/Users/chandrayogyadav/Desktop/Diabetes.csv")

### Creating Baseline model using Logistic Regression model for training ###
target = 'Outcome'

X = df.loc[:, df.columns != target]
Y = df.loc[:, df.columns == target]
X.shape
Y.shape

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=8)

## Method 1 Use the Logit model
logit_model=sm.Logit(Y,X)

## Fitting the model and publishing the results
result = logit_model.fit()
print(result.summary())

### With selected features


## removing logit model uminportant predictor
df.drop(["DiabetesPedigreeFunction", "Age", "Insulin"], 1 , inplace = True)


logit_model=sm.Logit(Y,X)
result = logit_model.fit()
print(result.summary())

### Model 2 - Logistic Regression
log_reg = LogisticRegression()
lr_model = log_reg.fit(x_train,y_train)
y_pred = lr_model.predict(x_test)

### scores
ac = accuracy_score(y_test, y_pred)
f_score = f1_score(y_test, y_pred)

print('Baseline model accuracy score with all feature:', ac)
print('Baseline model F1 score with all feature:', f_score)


### Check confusion matrix
plot_confusion_matrix(lr_model, x_test, y_test)  # doctest: +SKIP
plt.show()



##### Dropping not required columns  #####
df.drop(["Pregnancies", "BloodPressure", "SkinThickness"], 1 , inplace = True)

df.head()

# reassign data to train and test sets
X = df.loc[:, df.columns != target]
Y = df.loc[:, df.columns == target]


x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=8)


### use a baseline classifier , here for example logistic regression
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


#Plot the confusion matrix
sns.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True, fmt='g')
plt.show()

#####----------------- End ---------------------#######

















### Dimensionality reduction ###
array = df.values
X = array[:, 0:5]
Y = array[:, 5]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
print(fit.features)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
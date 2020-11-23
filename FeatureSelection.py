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
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor

#from scipy.stats import chi2_contingency
#from scipy.stats import chi2
from sklearn.feature_selection import SelectKBest


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

#df.head()
### 1. Method 1- Variance method ###
df.var()


### Creating input and target variables set ###
target = 'Outcome'
X = df.loc[:, df.columns != target]
Y = df.loc[:, df.columns == target]

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=8)

### Method 2. Wrapper Method- Recursive Feature elimination
### use a baseline classifier logistic regression
log_reg = LogisticRegression()
lr_model = log_reg.fit(x_train,y_train)

rfe = RFE(estimator=lr_model, step=1)
### fit the rfe function for ranking
rfe = rfe.fit(x_train, y_train)

print('No of feautres : %d' %rfe.n_features_)
print('Selected feautres : %s' %rfe.support_)
print('Feautres rank   : %s' %rfe.ranking_)

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


### Method 3. Embedded Method - LassoCV
### use LassoCV estimator, the features with the highest absolute coef. value are considered the most important.
clf = LassoCV().fit(X, Y)
signf_val = np.abs(clf.coef_)
print(signf_val)

coef = pd.Series(clf.coef_, index = X.columns)
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
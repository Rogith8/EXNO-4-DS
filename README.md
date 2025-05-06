# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
# FEATURE SCALING

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![Screenshot 2025-05-06 185553](https://github.com/user-attachments/assets/a70deb05-5948-4284-ae3f-4e0369ca2290)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![Screenshot 2025-05-06 185604](https://github.com/user-attachments/assets/95203530-0708-49c4-8766-9d74c834762c)
```
df.dropna()
```
![Screenshot 2025-05-06 185617](https://github.com/user-attachments/assets/506f24e2-c33d-4f0e-950d-8e3bf6b16ee6)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
# This is typically used in feature scaling,
#particularly max-abs scaling, which is useful
#when you want to scale data to the range [-1, 1]
#while maintaining sparsity (often used with sparse data).
```
![Screenshot 2025-05-06 185630](https://github.com/user-attachments/assets/d48a74ad-437a-4342-a96a-cd7f59ec9f62)
```
# Standard Scaling

from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
![Screenshot 2025-05-06 185643](https://github.com/user-attachments/assets/7e6edc5f-a732-45ea-81e0-638e0b16cb32)
```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![Screenshot 2025-05-06 185655](https://github.com/user-attachments/assets/354988d5-2ccf-42e7-b9e7-79414efc90a2)
```
#MIN-MAX SCALING:

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2025-05-06 185710](https://github.com/user-attachments/assets/c655e3d4-2ab6-4be6-b28e-6b5ddeb709f0)
```
#MAXIMUM ABSOLUTE SCALING:

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2025-05-06 185722](https://github.com/user-attachments/assets/bdeec927-0bd0-4c39-8b91-f1644dd94915)
```
#ROBUST SCALING

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
![Screenshot 2025-05-06 185736](https://github.com/user-attachments/assets/ad7f80e3-d324-4e74-aede-df3526ef9f2d)
```
#FEATURE SELECTION:

df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![Screenshot 2025-05-06 185751](https://github.com/user-attachments/assets/2dd416cf-7d0b-49c1-a431-7a70429dde8a)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![Screenshot 2025-05-06 185805](https://github.com/user-attachments/assets/604095b6-e376-471c-9b50-4b4b7ce5782f)
```
# Chi_Square

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
 # process non-numeric features. It ensures the model treats these columns as categories,
  # not as continuous numerical values.
df[categorical_columns]
```
![Screenshot 2025-05-06 185822](https://github.com/user-attachments/assets/1fcc4918-3f58-4c1a-bb12-4d6b7a802c86)
```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
```
![Screenshot 2025-05-06 185836](https://github.com/user-attachments/assets/906933bf-d0aa-42d8-bdea-c4ca87bf5e1d)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
#X contains all columns except 'SalStat' — these are the input features used to predict something.
#y contains only the 'SalStat' column — this is the target variable you want to predict.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![Screenshot 2025-05-06 185853](https://github.com/user-attachments/assets/fe2b04cd-ed0f-4b6b-8b08-2516aa0dc0ba)
```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![Screenshot 2025-05-06 185905](https://github.com/user-attachments/assets/ab9ed4d2-1287-40cd-9508-35b1ab54f57a)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![Screenshot 2025-05-06 185920](https://github.com/user-attachments/assets/3a15d8ec-daa5-4ffc-938d-eda30d53c1fe)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![Screenshot 2025-05-06 185933](https://github.com/user-attachments/assets/d0f4e748-9b18-4f38-b426-f3bff28c3e63)
```

X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![Screenshot 2025-05-06 185949](https://github.com/user-attachments/assets/a12d1528-2b63-4cd5-8658-ab394b41eb68)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![Screenshot 2025-05-06 190000](https://github.com/user-attachments/assets/c00392bd-b3e4-4ac8-8a20-6a675c928426)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![Screenshot 2025-05-06 190012](https://github.com/user-attachments/assets/b002b385-1610-43ce-8b2d-7df257cf7bd0)
```
!pip install skfeature-chappers
```
![Screenshot 2025-05-06 190028](https://github.com/user-attachments/assets/9c30ed7b-34ea-43b4-a2ce-31bc66f57f73)
```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![Screenshot 2025-05-06 190044](https://github.com/user-attachments/assets/e4f51c45-05e5-4a18-bfe2-98abcc4b1d44)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![Screenshot 2025-05-06 190108](https://github.com/user-attachments/assets/185ff838-131e-41e8-8f55-2ff89dfc48a2)
```
# Wrapper Method

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![Screenshot 2025-05-06 190127](https://github.com/user-attachments/assets/3dd516ac-2935-46b7-8f8d-4dfc4c8bc554)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```

![Screenshot 2025-05-06 190212](https://github.com/user-attachments/assets/2a90351a-c6b1-4dde-9bba-d453f93cd381)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.

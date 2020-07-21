import pandas as pd
import xgboost as xgb

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score

col = ['name', 'type', 'C1', 'C2', 'C3', 'COR1', 'COR2', 'COR3', 'E1', 'E2', 'E3', 'H1', 'H2', 'H3', 'DWF1', 'DWF2', 'DWF3']
data = pd.read_csv("Copy of results for noise 20%(3525).csv", names = col)

data = data.iloc[1:,1:]
data.C1 = data.C1.apply(lambda x: float(x))
data.COR1 = data.COR1.apply(lambda x: float(x))
data.E1 = data.E1.apply(lambda x: float(x))
data.H1 = data.H1.apply(lambda x: float(x))
data.DWF1 = data.DWF1.apply(lambda x: float(x))

print(f'Before Label Encoding{data.type.unique()}.')
data.type = preprocessing.LabelEncoder().fit_transform(data.type)
print(f'After Label Encoding{data.type.unique()}.')

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:], data.type, test_size = 0.1)

xclas = xgb.sklearn.XGBClassifier()  # and for classifier
xclas.fit(x_train, y_train)
xclas.predict(x_test)

print(cross_val_score(xclas, x_train, y_train)  )
print(xclas.predict(x_test))
print(list(y_test))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsCV


dat = pd.read_csv('~/f1_ml/cleaning/FULL_F1_DF.csv')
dat = dat.drop(['Unnamed: 0', "driver", "qualifying_time"], axis=1)
dat = dat.dropna()
predframe = pd.read_csv('~/Downloads/f1_ml_race_1_data.csv')
predframe = predframe.drop(["driver", "qualifying_time"], axis=1)
#print(dat)
y_train = dat['podium']
X_train = dat.drop(['podium'], axis = 1)

model = LassoLarsCV(cv = 10)
model.fit(X_train, y_train)
model.predict(predframe)
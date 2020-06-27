#importing important libraries
import pandas as pd
import _pickle as cPickle
import gzip
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#reading final data with final selected features
df = pd.read_csv('final_features.csv')

#handeling categorical data
df1 = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

#spliting data
X=df1.drop('total',axis=1)
Y=df1['total']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 40)

#building model
random_regressor = RandomForestRegressor()
random_regressor.fit(X_train,y_train)

#prediction
y_pred_rf = random_regressor.predict(X_test)

#saving model(compressed)
with gzip.open('model_compressed', 'wb') as f:
    cPickle.dump(random_regressor, f)


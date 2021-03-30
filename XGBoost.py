import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as MSE

import numpy as np


class XGBobj:
    def __init__(self, max_depth=6, objective="binary:logistic", gamma = 0, learning_rate = 0.01,
                    n_estimators = 1000, subsample = 0.5, nthread = 2):
        self.param = {'max_depth': max_depth,
                    'objective': objective,
                    'gamma': gamma,
                    'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                    'subsample': subsample,
                    "nthread": nthread}

    def train(self, train_X, train_Y, val_X, val_Y, epochs):
        dtrain = xgb.DMatrix(train_X, label=train_Y)
        dval = xgb.DMatrix(val_X, label=val_Y)

        self.bst = xgb.train(self.param, dtrain, epochs, [(dtrain, 'train'), (dval, 'eval')], early_stopping_rounds = 100, verbose_eval = True)

    def test(self, X, y):
        dtest = xgb.DMatrix(X, label=y)
        pred = self.bst.predict(dtest)

        out = np.round(pred)
        healthyIndex = (y == 0)
        sickIndex = (y == 1)
        
        valHealthyCorrect = np.sum(out[healthyIndex] == 0)
        valHealthyIncorrect = np.sum(healthyIndex) - valHealthyCorrect

        valSickCorrect = np.sum(out[sickIndex] == 1)
        valSickIncorrect = np.sum(sickIndex) - valSickCorrect

        print("Healthy Correct")
        print(valHealthyCorrect)
        print("Healthy Incorrect")
        print(valHealthyIncorrect)
        print("Sick Correct")
        print(valSickCorrect)
        print("Sick Incorrect")
        print(valSickIncorrect)


        print(MSE(y, pred))
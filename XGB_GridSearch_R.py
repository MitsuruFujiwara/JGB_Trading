import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def main():
    # load Data
    df = pd.read_hdf('data.h5', key='train_reg', mode='r')

    # set data
    trX, trY = np.array(df.drop('label', axis=1)), np.array(df['label'])

    # define Xgboost Classifier
    XGB = XGBRegressor()

    prm_learning_rate = [0.01, 0.10, 0.5]
    prm_max_depth = [50, 25, 10, 5]
    prm_n_estimators = [1000, 100, 10]
    prm_min_child_weight = [0.5, 0.75, 1.0]

    param_grid = [{'learning_rate':prm_learning_rate, 'max_depth':prm_max_depth,
                    'n_estimators': prm_n_estimators, 'min_child_weight': prm_min_child_weight}]

    gs = GridSearchCV(estimator=XGB, param_grid=param_grid, scoring='r2', cv=3, n_jobs=-1)

    gs.fit(trX, trY)

    result = pd.DataFrame(gs.cv_results_)

    result.to_csv('result_xgb_r.csv')

    joblib.dump(gs.best_estimator_, 'xgb_r.pkl')

    print(gs.best_score_)
    print(gs.best_params_)
    print(result)

if __name__ == '__main__':
    main()

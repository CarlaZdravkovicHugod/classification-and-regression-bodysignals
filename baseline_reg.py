import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import mean_squared_error


filename = 'sd-data_cleanedv1.csv'
data = pd.read_csv(filename)

cl_data = data.values

# Getting data and labels separated
X = np.delete(cl_data, 21, axis=1)
y = cl_data[:, 21]

K_outer = 10

outer_cv = KFold(n_splits=K_outer, shuffle=True)

final_preformance_for_k_outer = np.empty((K_outer,1))

for i_outer, (train_outer_index, test_outer_index) in enumerate(outer_cv.split(X)):
    # train_outer_index and test_outer_index is a list of indicies as the name suggests.
    # we now assign the data, in allignment with the splits, to varaibles
    X_train_outer, X_test_outer = X[train_outer_index, :], X[test_outer_index, :]
    y_train_outer, y_test_outer = y[train_outer_index], y[test_outer_index]

    guess = np.mean(y_test_outer)

    y_pred = np.empty((len(y_test_outer), 1))
    for i in range(len(y_pred)):
        y_pred[i, 0] = guess

    mse_outer = mean_squared_error(y_test_outer, y_pred)

    final_preformance_for_k_outer[i_outer] = mse_outer

print(final_preformance_for_k_outer)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import KFold

filename = 'sd-data_cleanedv1.csv'
data = pd.read_csv(filename)

cl_data = data.values

# Getting data and labels separated
X = cl_data[:, :-1]
y = cl_data[:, -1]

K_outer = 10

outer_cv = KFold(n_splits=K_outer, shuffle=True)

final_preformance_for_k_outer = np.empty((K_outer,1))

for i_outer, (train_outer_index, test_outer_index) in enumerate(outer_cv.split(X)):
    
        # train_outer_index and test_outer_index is a list of indicies as the name suggests.
        # we now assign the data, in allignment with the splits, to varaibles
        X_train_outer, X_test_outer = X[train_outer_index, :], X[test_outer_index, :]
        y_train_outer, y_test_outer = y[train_outer_index], y[test_outer_index]

        class_1 = np.count_nonzero(y_train_outer == 1.0)
        class_0 = np.count_nonzero(y_train_outer == 0.0)
        
        # classify based on majority and cal test error
        if class_1 >= class_0:
            final_preformance_for_k_outer[i_outer] = (np.count_nonzero(y_test_outer == 0.0))/len(X_test_outer)
        else:
            final_preformance_for_k_outer[i_outer] = (np.count_nonzero(y_test_outer == 1.0))/len(X_test_outer)
        
print(final_preformance_for_k_outer)
            
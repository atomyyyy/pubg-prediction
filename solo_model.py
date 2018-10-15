import pickle
from sklearn.linear_model import *
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

def get_absolute_error(predict, real):
    sum = 0
    for x,y in zip(predict, real):
        sum += abs(x-y)
    return sum

solo_data = pickle.load(open('solo_data.p','rb'))
solo_label = pickle.load(open('solo_label.p','rb'))

# Use ridge regression
kf = KFold(n_splits=5, shuffle=True)
alpha_list = [5**x for x in range(-5, 5)]
score = []
for alpha in alpha_list:
    error = 0
    for train_index, test_index in kf.split(solo_data):
        X_train, X_test = solo_data.iloc[train_index], solo_data.iloc[test_index]
        y_train, y_test = solo_label.iloc[train_index], solo_label.iloc[test_index]
        lm_model = Ridge(alpha=alpha)
        lm_model.fit(X_train, y_train)
        error += get_absolute_error(lm_model.predict(X_test), y_test)
    print('Ridge with regularization term {0} has mean absolute error {1}'.format(alpha, error/len(solo_data)))
    score.append(error/len(solo_data))

plt.plot(np.log10(alpha_list), score)
plt.show()
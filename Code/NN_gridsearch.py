from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load the data
# X = np.load('data/multiple_dipoles_eeg_10000_2.npy')
# y = np.load('data/multiple_dipoles_locations_10000_2.npy')
# y = np.reshape(y, (10000, 6))

X = np.load('data/multiple_dipoles_eeg_10000_1.npy')
y = np.load('data/multiple_dipoles_locations_10000_1.npy')
y = np.reshape(y, (10000, 3))

print(y)
input()

pipe_sgd = Pipeline([('scl', StandardScaler()),
        ('reg', MultiOutputRegressor(SGDRegressor()))])

grid_param_sgd = {
    'reg__estimator__alpha': [0.0001, 0.001],
    'reg__estimator__penalty': ['l2', 'l1'],
    'reg__estimator__fit_intercept': [True, False],
    'reg__estimator__learning_rate': ['constant', 'optimal', 'invscaling'],
    'reg__estimator__power_t: [0.1, 0.25, 0.3]'
    'reg__estimator__eta0': [0.0001, 0.001, 0.1], # learning rate



}

gs_sgd = (GridSearchCV(estimator=pipe_sgd,
                      param_grid=grid_param_sgd,
                      cv=2,
                      scoring = 'neg_mean_squared_error',
                      n_jobs = -1))

gs_sgd = gs_sgd.fit(X,y)

# Print the best hyperparameters and their corresponding score
print(f"Best hyperparameters: {gs_sgd.best_params_}")
print(f"Best score: {gs_sgd.best_score_}")

# Pipeline(steps=[('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
# ('reg', MultiOutputRegressor(estimator=SGDRegressor(loss = squared_error), n_jobs=1))])

# Evaluate the performance of the best model on the test set
best_model = gs_sgd.best_estimator_
y_pred = best_model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Test MSE: {mse}")

# param_grid = {
#     # 'scaler__with_mean': [True, False],
#     # 'scaler__with_std': [True, False],
#     'estimator__alpha': [0.0001, 0.001] #, 0.01, 0.1],
#     # 'regressor__average': [True, False],
#     # 'regressor__early_stopping': [True, False],
#     # 'regressor__epsilon': [0.1, 0.01, 0.001],
#     # 'regressor__eta0': [0.01, 0.1, 1],
#     # 'regressor__fit_intercept': [True, False],
#     # 'regressor__l1_ratio': [0.25, 0.5, 0.75],
#     # 'regressor__learning_rate': ['constant', 'optimal', 'invscaling'],
#     # 'regressor__penalty': ['l2', 'l1', 'elasticnet'],
#     # 'regressor__shuffle': [True, False],
#     # 'regressor__tol': [0.0001, 0.001, 0.01]
# }

# Best hyperparameters: {'reg__estimator__alpha': 0.0001, 'reg__estimator__eta0': 0.0001,
# 'reg__estimator__fit_intercept': False, 'reg__estimator__learning_rate': 'invscaling',
# 'reg__estimator__penalty': 'l2'}
# Best score: -1516.897186385399
# Test MSE: 1507.8309816235776

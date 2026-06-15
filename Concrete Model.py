import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Data
data = pd.read_excel('Concrete_Data.xls')
data.columns = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Super-Plasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Compressive Strength']
pd.set_option('display.max_columns', None)

#Adding A New Column to the Data Containing Water to Cement Ratio
data.insert(8, 'Var. 9', data['Water'] / data['Cement'])

#Checking data for any NaN Values
print(f'Is there any missing data in this dataset: {data.isnull().values.any()}')

#Convert DataFrame to a Numpy Array
data_array = np.array(data)
X = data_array[:, :-1]
y = data_array[:, -1]

#Variable Interactions and EDA
correlation_matrix = data.corr()
plt.figure(figsize = (5,5))
sns.heatmap(correlation_matrix, annot=True,annot_kws={"size": 6}, cmap = 'coolwarm', fmt = '.1f', linewidths = 1)
plt.title('Correlation Heatmap')
plt.xticks(rotation = 35, ha='right')
plt.yticks(rotation = 0, va='center')
plt.tight_layout()

fig, axes = plt.subplots(3, 3, figsize = (15, 15))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.scatter(X[:, i], y)
    ax.set_title(f'Independent Variable {i+1}')
    ax.grid(True)
fig.suptitle('Independent Variable Interactions with Dependent Variable', va='top')
plt.tight_layout()
plt.show()

#Scaling the Data
Scaler = sk.preprocessing.StandardScaler()
scaled_data_array = Scaler.fit_transform(data_array)
X_scaled = Scaler.fit_transform(X)

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.3, shuffle=True, random_state=69)
X_train_scaled = Scaler.fit_transform(X_train)
X_test_scaled = Scaler.transform(X_test)

poly_features = sk.preprocessing.PolynomialFeatures(degree = 2)
X_poly = poly_features.fit_transform(X)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = sk.model_selection.train_test_split(X_poly, y, test_size=0.3, shuffle=True, random_state=69)

#Linear Regression Model Evaluation
linear_model = sk.linear_model.LinearRegression().fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)
mse_lm = sk.metrics.mean_squared_error(y_test, y_pred_linear)
rmse_lm = sk.metrics.root_mean_squared_error(y_test, y_pred_linear)
r_squared_lm = sk.metrics.r2_score(y_test, y_pred_linear)

scores_lm = sk.model_selection.cross_val_score(linear_model, X, y, scoring = 'neg_mean_squared_error', cv=5)
rmse_score_lm = (-scores_lm)**0.5

print(f'\nLinear Model - Mean Squared Error: {mse_lm}')
print(f'Linear Model - Root Mean Squared Error: {rmse_lm}')
print(f'Linear Model - R-squared: {r_squared_lm}')

print(f'\nCV Linear Model - RMSE Score: {rmse_score_lm}')
print(f'CV Linear Model - Mean RMSE: {rmse_score_lm.mean()}')
print(f'CV Linear Model - Standard Deviation of RMSE: {rmse_score_lm.std()}')

#Regulatization with Lasso
lasso_model = sk.linear_model.LassoCV(cv=5, random_state=69).fit(X_train_scaled, y_train)
print(f'\nLasso Model Coefficients: {lasso_model.coef_}')
unimportant_features = np.where(lasso_model.coef_ == 0)
print(f'\nLasso Model Removed the Fine Aggregate Feature From the Model and the Coarse Aggregate Feature Was the Next Smallest')

y_pred_lasso = lasso_model.predict(X_test_scaled)
mse_lasso = sk.metrics.mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = sk.metrics.root_mean_squared_error(y_test, y_pred_lasso)
r_squared_lasso = sk.metrics.r2_score(y_test, y_pred_lasso)

print(f'\nLasso Model - Mean Squared Error: {mse_lasso}')
print(f'Lasso Model - Root Mean Squared Error: {rmse_lasso}')
print(f'Lasso Model - R-squared: {r_squared_lasso}')

#Regularization with Elastic Net
elastic_net_model = sk.linear_model.ElasticNetCV(cv=5, random_state=69).fit(X_train_scaled, y_train)
print(f'\nElastic Net Coefficients: {elastic_net_model.coef_}')
print(f'\nElastic Net Model Shrunk the Fine Aggregate Feature to a Very Small Value and the Coarse Aggregate Feature Was the Next Smallest')

y_pred_elastic = elastic_net_model.predict(X_test_scaled)
mse_elastic = sk.metrics.mean_squared_error(y_test, y_pred_elastic)
rmse_elastic = sk.metrics.root_mean_squared_error(y_test, y_pred_elastic)
r_squared_elastic = sk.metrics.r2_score(y_test, y_pred_elastic)

print(f'\nElastic Net Model - Mean Squared Error: {mse_elastic}')
print(f'Elastic Net Model - Root Mean Squared Error: {rmse_elastic}')
print(f'Elastic Net Model - R-squared: {r_squared_elastic}')

#Polynomial Regression
poly_model = sk.linear_model.LinearRegression().fit(X_train_poly, y_train_poly)
y_pred_poly = poly_model.predict(X_test_poly)

mse_pm = sk.metrics.mean_squared_error(y_test_poly, y_pred_poly)
rmse_pm = sk.metrics.root_mean_squared_error(y_test_poly, y_pred_poly)
r_squared_pm = sk.metrics.r2_score(y_test_poly, y_pred_poly)

scores_pm = sk.model_selection.cross_val_score(poly_model, X_train_poly, y_train_poly, scoring = 'neg_mean_squared_error', cv=5)
rmse_score_pm = (-scores_pm)**0.5

print(f'\nPolynomial Model - Mean Squared Error: {mse_pm}')
print(f'Polynomial Model - Root Mean Squared Error: {rmse_pm}')
print(f'Polynomial Model - R-squared: {r_squared_pm}')

print(f'\nCV Polynomial Model - RMSE Score: {rmse_score_pm}')
print(f'CV Polynomial Model - Mean RMSE: {rmse_score_pm.mean()}')
print(f'CV Polynomial Model - Standard Deviation of RMSE: {rmse_score_pm.std()}')

#Support Vector Regressor Model - Radial Basis Function
svr_model = sk.svm.SVR(kernel = 'rbf', C = 100, gamma ='auto', epsilon = 1).fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)
mse_svr = sk.metrics.mean_squared_error(y_test, y_pred_svr)
rmse_svr = sk.metrics.root_mean_squared_error(y_test, y_pred_svr)
r_squared_svr = sk.metrics.r2_score(y_test, y_pred_svr)

scores_svr = sk.model_selection.cross_val_score(svr_model, X_train_scaled, y_train, scoring = 'neg_mean_squared_error', cv=5)
rmse_score_svr = (-scores_svr)**0.5

print(f'\nSVR Model - Mean Squared Error: {mse_svr}')
print(f'SVR Model - Root Mean Squared Error: {rmse_svr}')
print(f'SVR Model - R-squared: {r_squared_svr}')

print(f'\nCV SVR Model - RMSE Score: {rmse_score_svr}')
print(f'CV SVR Model - Mean RMSE: {rmse_score_svr.mean()}')
print(f'CV SVR Model - Standard Deviation of RMSE: {rmse_score_svr.std()}')

#Random Forest Regressor - Ensemble Methods
rfr_model = sk.ensemble.RandomForestRegressor(n_estimators = 100, random_state=69).fit(X_train_scaled, y_train)
y_pred_rfr = rfr_model.predict(X_test_scaled)
mse_rfr = sk.metrics.mean_squared_error(y_test, y_pred_rfr)
rmse_rfr = sk.metrics.root_mean_squared_error(y_test, y_pred_rfr)
r_squared_rfr = sk.metrics.r2_score(y_test, y_pred_rfr)

scores_rfr = sk.model_selection.cross_val_score(rfr_model, X_train_scaled, y_train, scoring = 'neg_mean_squared_error', cv=5)
rmse_score_rfr = (-scores_rfr)**0.5

print(f'\nRandom Forest Regressor Model - Mean Squared Error: {mse_rfr}')
print(f'Random Forest Regressor Model - Root Mean Squared Error: {rmse_rfr}')
print(f'Random Forest Regressor Model - R-squared: {r_squared_rfr}')

print(f'\nCV Random Forest Regressor Model - RMSE Score: {rmse_score_rfr}')
print(f'CV Random Forest Regressor Model - Mean RMSE: {rmse_score_rfr.mean()}')
print(f'CV Random Forest Regressor Model - Standard Deviation of RMSE: {rmse_score_rfr.std()}')
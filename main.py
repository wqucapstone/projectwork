import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

### Read Macroeconomic, Microeconomic, and Geopolitical Data from downloaded spreadsheets

gdp_data = pd.read_excel(r'US_GDP.xlsx', index_col='Date', parse_dates=True)
spx_data = pd.read_excel(r'SPX_Historical_Levels.xlsx', index_col='Date', parse_dates=True)
ust_yield = pd.read_excel(r'US_10Y_Constant_Maturity_Yield.xlsx', index_col='Date', parse_dates=True)
wti_oil_price = pd.read_excel(r'WTI_Historical_Prices.xlsx', index_col='Date', parse_dates=True)
us_oil_production = pd.read_excel(r'U.S._Field_Production_of_Crude_Oil.xlsx', index_col='Date', parse_dates=True)
us_oil_imports = pd.read_excel(r'U.S._Imports_of_Crude_Oil.xlsx', index_col='Date', parse_dates=True)
geopolitical_risk = pd.read_excel(r'geopolitical_risk_data.xlsx', index_col='Date', parse_dates=True)

### Align Monthly Data with Quarterly GDP Data

aligned_wti = wti_oil_price.reindex(gdp_data.index, method='ffill')
aligned_ust = ust_yield.reindex(gdp_data.index, method='ffill')
aligned_spx = spx_data.reindex(gdp_data.index, method='ffill')
aligned_production = us_oil_production.reindex(gdp_data.index, method='ffill')
aligned_imports = us_oil_imports.reindex(gdp_data.index, method='ffill')
aligned_risk = geopolitical_risk.reindex(gdp_data.index, method='ffill')

### Combine GDP and aligned data
combined_data = pd.concat([gdp_data, aligned_ust, aligned_spx, aligned_wti, aligned_production, aligned_imports, aligned_risk], axis=1)

data_dictionary = {
    'GDP': gdp_data,
    'wti_oil_price': aligned_wti,
    'us_10y_treasury_constant_maturity_yield': aligned_ust,
    'sp_500_index': aligned_spx,
    'us_oil_production': aligned_production,
    'us_oil_imports': aligned_imports,
    'geopolitical_risk': aligned_risk
}

print(data_dictionary)

# Data Cleaning

# Fill missing values using linear interpolation

combined_data.interpolate(method='linear', inplace=True)

# Cleaning Data (Outlier Removal)- removes outliers using z-scores to filter data within a certain range

z_scores = np.abs((combined_data - combined_data.mean()) / combined_data.std())
combined_data = combined_data[(z_scores < 3).all(axis=1)]

# Handling Duplicates- ensures only the first occurrence of duplicate dates is kept.

combined_data = combined_data[~combined_data.index.duplicated(keep='first')]

# Data Visualization - Plotting Time Series of all the Factors being considered

combined_data.plot(figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Data for the West Texas Intermediate Crude Oil Price Forecasting')
plt.legend()
plt.grid(True)
plt.show()

# Data Visualization - US GDP steadily growing from 1990 till 2023

import matplotlib.pyplot as plt
plt.plot(combined_data.GDP, label = 'US GDP in $ Billions')
plt.legend()
plt.show()


# WTI Price increased significantly from 1990 till 2023, however it has been range-bound since peaking in 2008

plt.plot(combined_data.WTI_Price_in_USD, label = 'WTI Crude Oil Price in US$')
plt.legend()
plt.show()

bull_start_date = '2000-01-01'
bull_end_date = '2008-03-01'


### GDP vs US Aggregate Daily Crude Oil Demand

plt.plot(combined_data.GDP, label='US Nominal GDP in US$')
plt.plot(combined_data['U.S. Field Production of Crude Oil Thousand Barrels per Day']+combined_data['U.S. Imports of Crude Oil Thousand Barrels per Day'], label = 'U.S Daily Crude Oil Consumption (tbpd)')
plt.legend(title='US Nominal GDP vs US Daily Production')
plt.show()

plt.plot(combined_data.GDP, label='US Nominal GDP in US$')
plt.plot(combined_data['U.S. Field Production of Crude Oil Thousand Barrels per Day'], label = 'U.S Daily Crude Oil Production (tbpd)')
plt.legend(title='US Nominal GDP vs US Daily Production')
plt.show()

plt.plot(combined_data.GDP, label='US Nominal GDP in US$')
plt.plot(combined_data['U.S. Imports of Crude Oil Thousand Barrels per Day'], label = 'U.S Daily Crude Oil Imports (tbpd)')
plt.legend(title='US Nominal GDP vs US Daily Imports')
plt.show()


# ### Distributional Plots
#      A pair plot creates a grid of scatter plots to compare the distribution of pairs of numeric variables.
#      It also features a histogram for each feature in the diagonal boxes.

# ### Correlation matrix of the Macroeconomic, microeconomic and geopolitical factors

print(combined_data.corr())

sns.set(font_scale=1.15)
plt.figure(figsize=(8,4))
print(sns.heatmap(combined_data.corr(), cmap='RdBu_r', annot=True, vmin=-1, vmax=1))


### Distributional Plots
#      A pair plot creates a grid of scatter plots to compare the distribution of pairs of numeric variables
#      It also features a histogram for each feature in the diagonal boxes

print(sns.pairplot(combined_data, kind='scatter'))
print(sns.pairplot(combined_data, kind='reg'))


# Load crude oil price data (assuming it's already loaded as a DataFrame)

data = pd.read_excel(r'WTI_Historical_Prices.xlsx', index_col='Date', parse_dates=True)

X = combined_data[['US_10Y_CMT_Yield', 'SPX_Close', 'U.S. Field Production of Crude Oil Thousand Barrels per Day',
                   'U.S. Imports of Crude Oil Thousand Barrels per Day', 'GPR']]
y = combined_data['WTI_Price_in_USD']

# split the dataset into training and testing datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 42)

# Fitting the multiple linear regression model using the Training dataset

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# Print the model parameters - Intercept, and the Coefficients associated with each explanatory variable

print("Multilinear Regression Model Intercept : ", mlr_model.intercept_)
print("Multilinear Regression Model Coefficients : ", mlr_model.coef_)
print(list(zip(X, mlr_model.coef_)))

# Prediction using the testing dataset

y_predicted = mlr_model.predict(X_test)

# Predicted WTI Prices vs Actual WTI prices
print(pd.DataFrame({'Actual WTI Price': y_test, 'mlr_model Predicted WTI Price': y_predicted}))

# Evaluating the Multi Linear Regression Model Output

from sklearn import metrics

mean_absolute_error = metrics.mean_absolute_error(y_test, y_predicted)
mean_squared_error = metrics.mean_squared_error(y_test, y_predicted)
root_mean_squared_error = np.sqrt(mean_squared_error)

print('R-Squared Coefficient: {:.3f}'.format(mlr_model.score(X,y)*100))
print('Mean Absolute Error (MAS): ', mean_absolute_error)
print('Mean Squared Error (MSE): ', mean_squared_error)
print('Root Mean Squared Error (RMSE): ', root_mean_squared_error)

# Building a ARIMA Model using the TIME SERIES WTI Price data to forecast the WTI Prices

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict

# Load your data with 'Date' and 'Price' columns
wti_hist_data = 'WTI_Historical_Prices.xlsx'
df = pd.read_excel(wti_hist_data, index_col='Date', parse_dates=True)


# Determine ARIMA parameters (p, d, q)

# Checking the Autocorrelation Function (ACF) and the Partial-Autocorrelation function (PACF) to see data correlation

plt.figure(figsize=(16,8))
plot_acf(df.WTI_Price_in_USD)

fig = plt.figure(figsize=(16,8))
axis1 = fig.add_subplot(121)
axis1.set_title('First order differencing of the WTI Prices')
axis1.plot(df.WTI_Price_in_USD.diff())

axis2 = fig.add_subplot(122)
plot_acf(df.WTI_Price_in_USD.diff().dropna(), ax=axis2)
plt.show()

# Employing the ADF (Augmented-Dickey-Fuller Test) to test the null-hypothesis that the WTI Time series data is Non-Stationary.
# Check the p-value for a significance of 0.05, and see if the data is stationary or not.

from statsmodels.tsa.stattools import adfuller
test_output = adfuller(df.WTI_Price_in_USD.dropna())
print('The p-value is: ', test_output[1])

test_output = adfuller(df.WTI_Price_in_USD.diff().dropna())
print('The p-value is: ', test_output[1])

test_output = adfuller(df.WTI_Price_in_USD.diff().diff().dropna())
print('The p-value is: ', test_output[1])

# From the above as the p-value is below the threshold after the 1st order differencing, and drops to zero
# after the 2nd order differencing, we can choose the order as 1.
# this is also shown by the acf plot above.

# Determine the p parameter -- using the PACF plot

fig = plt.figure(figsize=(16,8))
axis1 = fig.add_subplot(121)
axis1.set_title('First order differencing of WTI prices')
axis1.plot(df.WTI_Price_in_USD.diff())

axis2 = fig.add_subplot(122)
plot_pacf(df.WTI_Price_in_USD.diff().dropna(), ax=axis2)
plt.show()

## From the above we can observe that the First lag is the most significant. Hence we set the parameter p to 1.

## Determine the q parameter -- using the ACF plot we can see that the MA parameter can be set to 1 as well.

p, d, q = 1, 1, 1  # ARIMA(p=1, d=1, q=1)

### Fit our ARIMA model to the 'WTI Price' column

arima = ARIMA(df.WTI_Price_in_USD, order=(1,1,1))
fitted_model = arima.fit()

print(fitted_model.summary())


arima = ARIMA(df.WTI_Price_in_USD, order=(1,2,1))
fitted_model = arima.fit()

print(fitted_model.summary())

fig, ax = plt.subplots(figsize=(14,8))
ax = df.loc['1990-01-01':].plot(ax=ax)
plot_predict(fitted_model, '2022-11-01', '2023-12-29', ax=ax)
plt.show()

plot_predict(fitted_model, '2022-12-29', '2023-12-29')
plt.show()

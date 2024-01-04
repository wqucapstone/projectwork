import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

### Read Macroeconomic, Microeconomic, and Geopolitical Data from downloaded spreadsheets

gdp_data = pd.read_excel(r'D:\Worldquant\9 Risk Management\GWP\GDP.xlsx', index_col='Date', parse_dates=True)
Oil_price= pd.read_excel(r'D:\Worldquant\9 Risk Management\GWP\DCOILWTICO.xlsx', index_col='Date', parse_dates=True)
Production=pd.read_excel(r'D:\Worldquant\9 Risk Management\GWP\U.S._Field_Production_of_Crude_Oil.xlsx', index_col='Date', parse_dates=True)
Import=pd.read_excel(r'D:\Worldquant\9 Risk Management\GWP\U.S._Imports_of_Crude_Oil.xlsx', index_col='Date', parse_dates=True)
GPR=pd.read_excel(r'D:\Worldquant\9 Risk Management\GWP\data_gpr_export.xlsx', index_col='Date', parse_dates=True)

### Align Monthly Data with Quarterly GDP Data

aligned_data1 = Oil_price.reindex(gdp_data.index, method='ffill')
aligned_data2 = Production.reindex(gdp_data.index, method='ffill')
aligned_data3 = Import.reindex(gdp_data.index, method='ffill')
aligned_data4 = GPR.reindex(gdp_data.index, method='ffill')

### Combine GDP and aligned data
combined_data = pd.concat([gdp_data, aligned_data1, aligned_data2, aligned_data3, aligned_data4], axis=1)

data_dictionary = {
    'GDP': gdp_data,
    'Oil_price': aligned_data1,
    'Production': aligned_data2,
    'Import': aligned_data3,
    'GPR': aligned_data4
}
print(data_dictionary)

plt.plot(Oil_price)
plt.ylabel('WTI Crude Price')
plt.title('WTI Crude Price from January 1990 till October 2023')
plt.grid(True)
plt.show()

# Data Cleaning

# Fill missing values using linear interpolation

combined_data.interpolate(method='linear', inplace=True)

# Cleaning Data (Outlier Removal)- removes outliers using z-scores to filter data within a certain range

z_scores = np.abs((combined_data - combined_data.mean()) / combined_data.std())
combined_data = combined_data[(z_scores < 3).all(axis=1)]

# Handling Duplicates- ensures only the first occurrence of duplicate dates is kept.

combined_data = combined_data[~combined_data.index.duplicated(keep='first')]

# US GDP significantly grew from 1990 till 2023

import matplotlib.pyplot as plt
plt.plot(combined_data.GDP, label = 'US GDP in $ Billions')
plt.legend()
plt.show()


# WTI Price increased significantly from 1990 till 2023, however it has been range boundsince peaking in 2008

plt.plot(combined_data.DCOILWTICO, label = 'WTI Crude Oil Price in US$')
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

data = pd.read_excel(r'DCOILWTICO.xlsx', index_col='Date', parse_dates=True)

### Split the data into training, validation, and testing sets
### 'test_size' parameter is to specify the size of the validation and testing sets (10% each)
### The remaining 80% of the data will be used for training

train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

### Now, we have our data split into training, validation, and testing sets
### We can use these sets for machine learning model development and evaluation

print(train_data)
print(temp_data)
print(valid_data)
print(test_data)

### In the code above, we use the train_test_split function from the scikit-learn library to split our data into training, validation, 
### and testing sets. The test_size parameter is set to 0.2 for the initial split (80% training, 20% combined for validation and testing), 
### and then the 20% is further split into 10% each for validation and testing.

### In the code above, we use the train_test_split function from the scikit-learn library to split our data into training, validation, 
### and testing sets. The test_size parameter is set to 0.2 for the initial split (80% training, 20% combined for validation and testing), 
### and then the 20% is further split into 10% each for validation and testing.### In the code above, we use the train_test_split function 
### from the scikit-learn library to split our data into training, validation, and testing sets. The test_size parameter is set to 0.2 for 
### the initial split (80% training, 20% combined for validation and testing), and then the 20% is further split into 10% each for validation 
### and testing.

### With this allocation, we can confidently proceed with building, tuning, and evaluating our crude oil price forecasting model. The validation 
### set will help us fine-tune our model's hyperparameters, while the testing set will provide a final assessment of its performance.

!pip install pgmpy

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load your data with 'Date' and 'Price' columns
excel_file = 'DCOILWTICO.xlsx'
df = pd.read_excel(excel_file)

# Determine ARIMA parameters (p, d, q)
p, d, q = 1, 1, 1  # Adjust based on your data analysis

### Fit an ARIMA model to the 'Price' column

### Forecast future values
forecast_steps = 3  # Forecasting for the next 3 periods

print(df['DCOILWTICO'].shape)

type(results.forecast(steps=forecast_steps))

forecast, stderr, conf_int = results.forecast(steps=forecast_steps)

print(stderr)

### Create an index for the forecasted data

forecast_index = pd.date_range(start=df['Date'].iloc[-1], periods= forecast_steps) 
print(forecast_index)

print(results.forecast(steps=3))

### Plot the observed 'Price' and the forecast

plt.figure(figsize=(12, 6))
plt.plot(df.Date, df['DCOILWTICO'], label='Price', color='black')
plt.plot(forecast_index, results.forecast(steps=3), label='Forecasted Price', color='green', linewidth=10)
plt.title('Oil Market from 1990 till 2023 - Forecasts for the next 3 months in Yellow')
plt.show()

Oil_price= pd.read_excel(r'DCOILWTICO.xlsx', index_col='Date', parse_dates=True)

## Stagnant Oil Prices Periods 

plt.plot(Oil_price[:120])
plt.ylabel('WTI Price')
plt.title('Oil price stagnated between 1990 till 2000 due to conflicting factors - Gulf wars vs Economic Crisis in Asia, Mexico, Russia')
plt.grid(True)
plt.show()

plt.plot(Oil_price[256:293])
plt.ylabel('WTI Price')
plt.title('Oil price stagnated between 2011 till 2014 despite Global Economic Recovery and QE, as US Shale boosted supply significantly')
plt.grid(True)
plt.show()

## Bullish Oil Prices Periods 

plt.plot(Oil_price[144:223])
plt.ylabel('WTI Price')
plt.title('Oil Bull Market from Jan 2002 till July 2008, driven by solid US, China GDP growth driven Consumption')
plt.grid(True)
plt.show()

plt.plot(Oil_price[231:256])
plt.ylabel('WTI Price')
plt.title('Post 2008 Global Financial Crisis, Oil Bull Regime resumed in Apr 2009 till Apr 2011 driven by QE and Global recovery')
plt.grid(True)
plt.show()

## Bearish Oil Prices Periods 

plt.plot(Oil_price[222:231])
plt.ylabel('WTI Price')
plt.title('Oil Bear Regime from July 2008 till Mar 2009 due to the onset of the 2008 Global Financial Crisis')
plt.grid(True)
plt.show()

plt.plot(Oil_price[293:315])
plt.ylabel('WTI Price')
plt.title('Oil Bear Regime from July 2014 till Jan 2016 due to Saudi Led OPEC War on US Shale Industry by increasing supply')
plt.grid(True)
plt.show()

plt.plot(Oil_price[:143], label='Price', color='grey')
plt.plot(Oil_price[144:221], label='Price', color='green')
plt.plot(Oil_price[222:232], label='Price', color='red')
plt.plot(Oil_price[233:256], label='Price', color='green')
plt.plot(Oil_price[257:293], label='Price', color='grey')
plt.plot(Oil_price[294:315], label='Price', color='red')
plt.plot(Oil_price[316:390], label='Price', color='green')
plt.plot(Oil_price[391:], label='Price', color='grey')
plt.ylabel('WTI Price')
plt.title('Oil Market from 1990 till 2023 - Stagnant, Bull, Bear Market Regimes Fitted')
plt.grid(True)
plt.show()


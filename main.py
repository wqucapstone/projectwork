#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### i. Read Macroeconomic, Microeconomic, and Geopolitical Data from downloaded spreadsheets

plt.plot(Oil_price)
plt.ylabel('WTI Crude Price')
plt.title('WTI Crude Price from January 1990 till October 2023')
plt.grid(True)
plt.show()


# ### ii. From the observed macro, geopolitical, and markets data, apply EDA and identify the market regimes 

# ### Bull regimes

plt.plot(Oil_price[144:223])
plt.ylabel('WTI Price')
plt.title('Oil Bull Market from Jan 2002 till Sep 2008, driven by solid US, China GDP growth driven Consumption')
plt.grid(True)
plt.show()

plt.plot(Oil_price[231:256])
plt.ylabel('WTI Price')
plt.title('Post 2008 Global Financial Crisis, Oil Bull Regime resumed in Apr 2009 till Apr 2011 driven by QE and Global recovery')
plt.grid(True)
plt.show()


plt.plot(Oil_price[144:223])
plt.ylabel('WTI Price')
plt.title('Oil Bull Market from Jan 2002 till Sep 2008')
plt.grid(True)
plt.show()


# ### Bear Regimes

plt.plot(Oil_price[222:231])
plt.ylabel('WTI Price')
plt.title('Oil Bear Regime from Aug 2008 till Mar 2009 due to the onset of the 2008 Global Financial Crisis')
plt.grid(True)
plt.show()


plt.plot(Oil_price[293:315])
plt.ylabel('WTI Price')
plt.title('Oil Bear Regime from July 2014 till Jan 2016 due to Saudi Led OPEC War on US Shale Industry by increasing supply')
plt.grid(True)
plt.show()


# ### Stagnant Regimes

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


# ### ii. Align Monthly Data with Quarterly GDP Data

# Align other data with GDP data index

aligned_data1 = Oil_price.reindex(gdp_data.index, method='ffill')
aligned_data2 = Production.reindex(gdp_data.index, method='ffill')
aligned_data3 = Import.reindex(gdp_data.index, method='ffill')
aligned_data4 = GPR.reindex(gdp_data.index, method='ffill')


# ### iii. Combine GDP and aligned data

combined_data = pd.concat([gdp_data, aligned_data1, aligned_data2, aligned_data3, aligned_data4], axis=1)

combined_data


# ### 5. Data Cleaning

# ### Fill missing values using linear interpolation 

combined_data.interpolate(method='linear', inplace=True)


# ### Cleaning Data (Outlier Removal)- removes outliers using z-scores to filter data within a certain range

z_scores = np.abs((combined_data - combined_data.mean()) / combined_data.std())
combined_data = combined_data[(z_scores < 3).all(axis=1)]

combined_data


# ### Handling Duplicates- ensures only the first occurrence of duplicate dates is kept.

combined_data = combined_data[~combined_data.index.duplicated(keep='first')]


# ### 6. Sterilized Data Sets (Combined Clean Data Sets)

combined_data


# ### 7. Exploratory Data Analysis

combined_data.plot(figsize=(15,8))
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Data for Crude Oil Price Forecasting')
plt.legend()
plt.grid(True)
plt.show()

# ### US GDP significantly grew from 1990 till 2023

import matplotlib.pyplot as plt
plt.plot(combined_data.GDP, label = 'US GDP in $ Billions')
plt.legend()
plt.show()


# ### WTI Price increased significantly from 1990 till 2023, however it has been range boundsince peaking in 2008

plt.plot(combined_data.DCOILWTICO, label = 'WTI Crude Oil Price in US$')
plt.legend()
plt.show()


bull_start_date = '2000-01-01'
bull_end_date = '2008-03-01'


# ### GDP vs US Aggregate Daily Crude Oil Demand

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

combined_data.corr()


sns.set(font_scale=1.15)
plt.figure(figsize=(8,4))
sns.heatmap(
    combined_data.corr(),        
    cmap='RdBu_r', 
    annot=True, 
    vmin=-1, vmax=1);


# ### Distributional Plots 
#      A pair plot creates a grid of scatter plots to compare the distribution of pairs of numeric variables.
#      It also features a histogram for each feature in the diagonal boxes.

sns.pairplot(combined_data, kind='scatter')


sns.pairplot(combined_data, kind='reg')


# ### 8. EDA Observations and Assessment 
#       

# ### 1. From the EDA, we can observe that since 1990, US GDP grew from USD 5.9 Trillion to 26.8 Trillion. 
# ### 2. With increased Economic Growth, WTI Oil price kept increasing from around USD 22  to USD 80.  
#        Outside US GDP, there are other factors that drive the Cost of production including CPI and the significant economic 
#        activity in China (coinciding with a massive increase in Chinese Oil Consumption).
# ### 3. Despite US GDP growing over 300% (1990-2023), Total US Oil consumption hasn't grown that significantly. 
#       This is due to the fact that Automobiles have been more energy efficient (increased mileage), and also 
#       Industrial usage of Oil has been reduced due the increased usage of Natural Gas. 
#       This explains why the Total Oil Consumption grew only moderately from around 13.7 mbpd to 18.9 mbd 
# ### 4. Until 2008, US Oil Imports have steadily increased from around 5 mbpd to over 10 mbpd
#       In 2008, we could see a significant drop in Oil imports as Oil Demand fell due to the US and Global Economic Recession 
#       In 2020, we could see a collapse in Oil demand due to the Covid Pandemic Lock downs
# ### 5. The most notable observation is US Oil production (after falling steadily over 20 years from 1990 till 2010)  started growing sharply from 2010 onwards
#       This is because of the Shale Technology caused a domestic boom, and also the higher oil prices incentivised producers to boost cheaper domestic production sharply. Thus the Increase in Domestic Oil Production has significantly leapfrogged since the Shale Revolution started in 2010
# ### 6. Based on Correlation matrix, we can see the strong linear relationship between GDP, Oil Production, Price
# 
# ### 7. The impact of Geopolitical risk is only observed over shorter periods of time, rather than over the long term, as Oil producers are incentivised in the event of a supply cuts from geopolitically sensitive countries either due to unrest or due to trade sanctions
# 
# ### 8. From the pair plots, we can observe the relationships between US GDP, and US Domestic production, and how they changed linearly since the Shale Boom in 2010, and vice-versa can be observed with US Crude imports. 
# 
# ### 9. From the pair plots, we can also observe that the GPR relationship with the Oil price tends to be clustered, and it can be due to the reason that Geopolitical risks tend to impact oil prices over the short term.
#     





#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ARIMA (Auto-regressive Integrated Moving Average) & Seasonal ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_excel('D:\OneDrive - Tata Class Edge\B2C\Retention Analysis\DS Model\Sales Forecast Data.xlsx', sheet_name='Sample Data')
df.head()


# In[3]:


df_sales = df.drop(['Sales'], axis=1)
df_sales.tail()


# In[4]:


df_sales.info()
df_sales.drop(90,axis=0,inplace=True)


# In[5]:


df_sales.set_index('Date', inplace=True)
df_sales.tail()


# In[6]:


df_sales.describe()


# In[7]:


#Visual the data
df_sales.plot(figsize=(20,10))


# In[8]:


# Testing for Stationary
from statsmodels.tsa.stattools import adfuller
test_result = adfuller(df_sales['Total'])


# In[9]:


#H0: It is not stationary
#H1: It is stationary

def adfuller_test(Sales):
    result = adfuller(Sales)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label+': '+str(value))
    if result[1] <= 0.05:
        print('Strong evidence against the null hypothesis(H0), reject the H0. Data has no unit root and is stationary')
    else:
        print('Week evidence against the null hypothesis, time series has a unit root, indicating it is non-stationary')


# In[10]:


adfuller_test(df_sales['Total'])


# ### Differencing

# In[11]:


df_sales['Sales Diff 1'] = df_sales['Total'] - df_sales['Total'].shift(1)
df_sales['Sales Diff 10'] = df_sales['Total'] - df_sales['Total'].shift(10)


# In[12]:


#df['Sales Diff 1'] p-value is closer to zero
adfuller_test(df_sales['Sales Diff 1'].dropna())


# In[13]:


df_sales['Sales Diff 1'].plot(figsize=(20,10))


# In[14]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df_sales['Total'])
plt.show()


# In[15]:


#Import Auto-Correlation & Partial Auto-Correlation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[16]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(df_sales['Sales Diff 1'].dropna(), lags = 40, ax = ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df_sales['Sales Diff 1'].dropna(), lags = 40, ax = ax2)


# In[17]:


#For Non-seasonal data
#p=1, d=1, q=1
from statsmodels.tsa.arima_model import ARIMA


# In[18]:


model = ARIMA(df_sales['Total'], order=(6,1,0))
model_fit = model.fit()


# In[19]:


model_fit.summary()


# In[20]:


df_sales.shape


# In[21]:


df_sales['forecast'] = model_fit.predict(start=80,end=90,dynamic=True)
df_sales[['Total','forecast']].plot(figsize = (20,10))


# In[22]:


df_sales.tail()


# In[23]:


import statsmodels.api as sm


# In[24]:


model = sm.tsa.statespace.SARIMAX(df_sales['Total'],order=(1,1,0),seasonal_order=(1,1,0,16))
result = model.fit()


# In[25]:


df_sales['Sforecast'] = result.predict(start = 80,end = 89, dynamic = True)
df_sales[['Total','Sforecast']].plot(figsize = (20,10))


# In[26]:


from pandas.tseries.offsets import DateOffset


# In[27]:


from datetime import date
future_dates = [df_sales.index[-1] + DateOffset(days=x)for x in range(0,77,3)]


# In[28]:


future_date_df = pd.DataFrame(index=future_dates[1:],columns=df_sales.columns)


# In[29]:


future_date_df.head()


# In[30]:


future_df = pd.concat([df_sales,future_date_df])
future_df.tail()


# In[31]:


future_df['forecast'] = result.predict(start = 90, end = 115, dynamic = True)
future_df[['Total','forecast']].plot(figsize = (20,10))


# In[32]:


future_df.head()


# In[33]:


#future_df.drop(['Sales Diff 1','Sales Diff 10','Sforecast'],axis=1,inplace=True)


# In[34]:


future_df.to_excel('D:\OneDrive - Tata Class Edge\B2C\Retention Analysis\DS Model\Final Forecast Sheet.xlsx')


# In[ ]:





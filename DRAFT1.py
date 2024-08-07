#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import warnings 
warnings.filterwarnings('ignore')
df = pd.read_csv("combined_file_draftfin.csv")

#Data Preprocessing

# shape of our dataset
# df.shape
print("No of rows in dataset:",df.shape[0])
print("No of columns in dataset:",df.shape[1])

df.columns

print(" \n Column Types")
df.dtypes


# In[6]:


# getting the numerical estimates of all the numerical column
df.describe()


# In[16]:


df.info()


# In[7]:


#Checking null values in our dataset
df.isna().sum()


# In[10]:


sns.heatmap(df.isna(),yticklabels=False,cmap='Blues')
plt.show()


# In[11]:


#Since PM 4.0 and PC 4.0 is completely empty, we can remove them from our calculations 
df.drop(columns=['PM4.0'],inplace=True)
df.drop(columns=['PC4.0'],inplace=True)


# In[12]:


df.drop(columns=['DataSet ID'],inplace=True)
df.isna().sum()


# In[15]:


#Except Timestamp all other columns have null values, we hence, replace all the null values with the mean of that particualr column

df.columns



# In[16]:


null_cols = ['Timestamp','PM0.1', 'PM0.3', 'PM0.5', 'PM1.0', 'PM2.5', 'PM5.0',
       'PM10.0', 'PC0.1', 'PC0.3', 'PC0.5', 'PC1.0', 'PC2.5', 'PC5.0',
       'PC10.0', 'Temp AQ', 'Humidity AQ', 'Pressure AQ', 'VOC AQ', 'No2 Gas',
       'Solar Panel Power', 'CO2', 'NOX AQ']
df = df[null_cols]
df[null_cols].dtypes


# In[11]:


for i in null_cols:
    if i != 'Timestamp':
        df[i] = df[i].fillna(df[i].mean())


# In[12]:


df.isna().sum()


# In[18]:



#Converting the Timstamp column from epoch to readable date
df['Timestamp'] = pd.to_datetime(df['Timestamp'],unit='ms')


# In[48]:


plt.figure(figsize=(10,10))
sns.boxplot(data=df)
plt.xticks(rotation='vertical')
plt.show()


# In[42]:


# getting the quartile one and quartile 3 values of each column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
# finally calculating the interquartile range IQR
IQR = Q3 - Q1


# In[49]:


((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()


# In[45]:


mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
mask


# In[46]:


for i in mask.columns:
    df[i].astype('float')
    temp = df[i].median()
    df.loc[mask[i], i] = temp


# In[47]:


((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()


# In[50]:


plt.figure(figsize=(5,5))
sns.boxplot(data=df)
plt.xticks(rotation='vertical')
plt.show()


# In[52]:


df.corr()


# In[2]:


df.columns


# In[ ]:


df.set_index('Timstamp', inplace=True)


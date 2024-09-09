#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


df.columns


# In[4]:


df.isna().sum()



# In[30]:


sns.heatmap(df.isna(),yticklabels=False,cmap='Blues')
plt.show()


# In[5]:


null_cols = ['Timestamp','PM0.1', 'PM0.3', 'PM0.5', 'PM1.0', 'PM2.5', 'PM5.0',
       'PM10.0', 'PC0.1', 'PC0.3', 'PC0.5', 'PC1.0', 'PC2.5', 'PC5.0',
       'PC10.0', 'Temp AQ', 'Humidity AQ', 'Pressure AQ', 'VOC AQ', 'No2 Gas',
       'Solar Panel Power', 'CO2', 'NOX AQ']
df = df[null_cols]
df[null_cols].dtypes


# In[6]:


for i in null_cols:
    if i != 'Timestamp':
        df[i] = df[i].fillna(df[i].mean())


# In[7]:


df.isna().sum()


# In[6]:


df['Timestamp'] = pd.to_datetime(df['Timestamp'],unit='ms')


# In[18]:



# getting the quartile one and quartile 3 values of each column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

# finally calculating the interquartile range IQR
IQR = Q3 - Q1


# In[17]:


((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()


# In[33]:


df.head()


# In[20]:



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(scaled_df.describe())


# In[34]:


import matplotlib.pyplot as plt
df.hist(bins=20, figsize=(20, 15))
plt.tight_layout()
plt.show()


# In[10]:


import seaborn as sns

# Calculate the correlation matrix
corr_matrix = df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Display the correlation of features with PM2.5
print(corr_matrix['PM2.5'].sort_values(ascending=False))


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

# Pairwise scatter plots
features = df.columns.difference(['PM2.5', 'Timestamp'])
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=feature, y='PM2.5')
    plt.title(f'Scatter Plot of {feature} vs PM2.5')
    plt.show()


# In[12]:



#This is to show the need of normalisation of these columns since they are in a different ranges
#We need ot normalize them so that the correlation between the PM2.5 and others can be estblished...
df.agg(['min', 'max'])


# In[ ]:


sns.pairplot(df[features.to_list() + ['PM2.5']], diag_kind='kde')
plt.show()


# In[1]:


df.columns


# In[7]:


df.plot()
plt.show()


# In[8]:


from pandas.plotting import lag_plot
lag_plot(df)
plt.show()


# In[9]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df)
plt.show()


# In[ ]:





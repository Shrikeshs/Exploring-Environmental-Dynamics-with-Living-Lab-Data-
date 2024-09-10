# ---------------------------------------
# Importing Libraries
# ---------------------------------------

from datetime import timedelta, datetime
import matplotlib.pyplot as plt

# ---------------------------------------
# Data Preparation and Timestamp Formatting
# ---------------------------------------

# Converting 'Timestamp' from milliseconds to a datetime object
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

# Creating additional time-related features from the 'Timestamp'
df['Formatted_Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Time'] = df['Timestamp'].dt.strftime('%H:%M:%S')

# ---------------------------------------
# Monthly Trend Analysis of PM2.5
# ---------------------------------------

# Calculating the monthly average of PM2.5
monthly_pm25_trend = df.groupby(df['Month'])['PM2.5'].mean()

# Plotting the monthly trend of PM2.5
plt.figure(figsize=(10, 5))
plt.plot(monthly_pm25_trend.index, monthly_pm25_trend.values, marker='o')
plt.title('Seasonal Trend of PM2.5')
plt.xlabel('Month')
plt.ylabel('PM2.5')
plt.grid(True)
plt.show()

# ---------------------------------------
# Scatter Plot: PM2.5 vs Temperature
# ---------------------------------------

# Creating a scatter plot to analyze the relationship between PM2.5 and Temperature
plt.figure(figsize=(10, 5))
plt.scatter(df['Temp AQ'], df['PM2.5'], alpha=0.6)
plt.title('Scatter Plot between PM2.5 vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('PM2.5 Concentration')
plt.grid(True)
plt.show()

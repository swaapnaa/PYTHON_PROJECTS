#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from tqdm.auto import tqdm

# Load the data
file_path = 'MPS LSOA Level Crime (Historical).csv'
data = pd.read_csv(file_path, delimiter=',', encoding='utf-8')

# Show the head of the dataframe
data_head = data.head()
print(data_head)

# Describe the dataframe without the 'datetime_is_numeric' argument
data_description = data.describe(include='all')
print(data_description)


# In[7]:


# Import matplotlib.pyplot with the common alias 'plt'
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Re-run the plotting code
# Correcting the list of monthly columns to exclude 'Total Crime'
monthly_columns = data.columns[5:-1]  # Exclude the last column which is 'Total Crime'

# Calculate the overall crime trend
overall_crime_trend = data[monthly_columns].sum()

# Convert the index to datetime to help with plotting
overall_crime_trend.index = pd.to_datetime(overall_crime_trend.index, format='%Y%m')

# Sort the index to ensure the dates are in order
overall_crime_trend.sort_index(inplace=True)

# Plot the overall crime trend
plt.figure(figsize=(15, 5))
overall_crime_trend.plot(title='Overall Crime Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Crime')
plt.show()


# In[8]:


# Calculate the total crime for each major category across all months
major_category_totals = data.groupby('Major Category')[monthly_columns].sum().sum(axis=1)

# Sort the totals in descending order to see which categories have the most crimes
major_category_totals_sorted = major_category_totals.sort_values(ascending=False)

# Plot the total crimes for each major category
plt.figure(figsize=(10, 8))
major_category_totals_sorted.plot(kind='barh', title='Total Crimes by Major Category')
plt.xlabel('Total Crimes')
plt.ylabel('Major Category')
plt.show()


# In[12]:


# Plot the stacked area chart for better visualization of the monthly crime by major category
plt.figure(figsize=(20, 10))
monthly_crime_by_category.plot(kind='area', stacked=False, title='Monthly Crime by Major Category')
plt.xlabel('Month')
plt.ylabel('Total Crimes')
plt.legend(title='Major Category')
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Adjust the layout to fit the labels
plt.show()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('MPS LSOA Level Crime (Historical).csv')
monthly_columns = data.columns[6:]

data['Total Crimes'] = data[monthly_columns].sum(axis=1)
total_crimes_by_lsoa = data.groupby('LSOA Name')['Total Crimes'].sum().reset_index()
sorted_crimes_by_lsoa = total_crimes_by_lsoa.sort_values(by='Total Crimes', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(sorted_crimes_by_lsoa['LSOA Name'].head(10), sorted_crimes_by_lsoa['Total Crimes'].head(10))
plt.xlabel('Total Crimes')
plt.ylabel('LSOA Name')
plt.title('Top 10 LSOAs with the Most Crimes')
plt.show()


# In[7]:


# Calculate the correlation between different types of crimes
crime_types = data.columns[6:]
correlation_matrix = data[crime_types].corr()
correlation_matrix


# In[11]:


import numpy as np
import seaborn as sns

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# In[12]:


# Impact of covid-19 lockdown
# Calculate the total crimes in 2019 and 2020 by selecting the columns for each year
monthly_columns_2019 = [col for col in data.columns if '2019' in col]
monthly_columns_2020 = [col for col in data.columns if '2020' in col]

total_crimes_2019 = data[monthly_columns_2019].sum().sum()
total_crimes_2020 = data[monthly_columns_2020].sum().sum()

# Display the total crimes for each year
total_crimes_2019, total_crimes_2020


# In[13]:


import matplotlib.pyplot as plt

# Calculate the total crimes in 2019 and 2020 by selecting the columns for each year
monthly_columns_2019 = [col for col in data.columns if '2019' in col]
monthly_columns_2020 = [col for col in data.columns if '2020' in col]

total_crimes_2019 = data[monthly_columns_2019].sum().sum()
total_crimes_2020 = data[monthly_columns_2020].sum().sum()

# Create a bar plot to visualize the total crimes for each year
plt.bar(['2019', '2020'], [total_crimes_2019, total_crimes_2020])
plt.title('Total Crimes in 2019 vs 2020')
plt.xlabel('Year')
plt.ylabel('Total Crimes')
plt.show()


# In[15]:


# Reshape the data to have a single 'Month' column
import pandas as pd
from tqdm.notebook import tqdm

# Load the data
file_path = 'MPS LSOA Level Crime (Historical).csv'
data = pd.read_csv(file_path)

# Melt the dataframe to convert the month columns into rows
monthly_data = data.melt(id_vars=['LSOA Code', 'LSOA Name', 'Borough', 'Major Category', 'Minor Category'], 
                         var_name='Month', value_name='Crime Count')

# Convert 'Month' to datetime
monthly_data['Month'] = pd.to_datetime(monthly_data['Month'], format='%Y%m')

# Aggregate the data by month to get the total crimes per month
monthly_totals = monthly_data.groupby('Month')['Crime Count'].sum().reset_index()

# Plotting the monthly crime totals to observe seasonal patterns
plt.figure(figsize=(15, 5))
plt.plot(monthly_totals['Month'], monthly_totals['Crime Count'])
plt.title('Monthly Crime Totals in London')
plt.xlabel('Month')
plt.ylabel('Total Crime Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[16]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Perform seasonal decomposition of the time series data
# Assuming total_crimes_df is the dataframe containing the monthly crime totals

decomposition = seasonal_decompose(monthly_totals['Crime Count'], model='additive', period=12)

# Plot the original data, trend, seasonality, and residuals
plt.figure(figsize=(14, 7))
plt.subplot(411)
plt.plot(monthly_totals['Month'], monthly_totals['Crime Count'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(monthly_totals['Month'], decomposition.trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(monthly_totals['Month'], decomposition.seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(monthly_totals['Month'], decomposition.resid, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

# Show the plots
plt.show()


# In[ ]:





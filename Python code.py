# Importing the library that will ingest the data
import pandas as pd


#Creating the function that will take the file name as an argument.
def process_co2_data(filename):
    # Load the data into a DataFrame
    df = pd.read_csv(filename, skiprows=4)
    
    # Extract the data for the years 2010-2019
    years_df = df.loc[:, 'Country Name':'2019']
    years_df.columns = [col if not col.isdigit() else str(col) for col in years_df.columns]
    
    # Transpose the DataFrame to get a country-centric view
    countries_df = years_df.transpose()
    
    # Replace empty values with 0
    countries_df = countries_df.fillna(0)
    
    # Set the column names for the countries DataFrame
    countries_df.columns = countries_df.iloc[0]
    countries_df = countries_df.iloc[1:]
    countries_df.index.name = 'Year'
    
    # Set the column names for the years DataFrame
    years_df = years_df.rename(columns={'Country Name': 'Year'})
    years_df = years_df.set_index('Year')
    
    return years_df, countries_df


#calling the function we created above
years_df, countries_df = process_co2_data('CO2E_4904492.csv')
print(years_df)
print(countries_df)

# EXPLORING THE STATISTICAL PROPERTIES OF INTEREST
# Getting a description for the data inside the columns
years_df.describe()

# Getting a description for the data inside the countries columns
countries_df.describe()

# Getting the top 20 countries with the highest CO2 emissions (kg per PPP $ of GDP) in the year 2019
years_df.nlargest(20, '2019')

# Getting the bottom 20 countries with the lowest CO2 emissions (kg per PPP $ of GDP) in the year 2019
years_df.nsmallest(20, '2019')

"""
Correlations 

Looking at the data above, for the top 20 and the bottom 20 countries in terms of CO2 emissions 
(kg per PPP $ of GDP) we see that there is no corelation either by continent or by the most powerful 
countries in the world. The data is just random and unpredictable.

"""

years_df.info()

countries_df.info()

# PLOTTING THE VISUALIZATIONS
# LINE PLOT

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('CO2E_4904492.csv', skiprows=4)

# Select 10 countries of your choice
countries = ['United States', 'China', 'United Kingdom', 'Russian Federation', 'France', 'Japan', 'Germany', 'Israel', 'United Arab Emirates', "Saudi Arabia"]

# Create a new DataFrame for the selected countries
selected_countries_df = df[df['Country Name'].isin(countries)].set_index('Country Name').loc[:, '2010':'2019']

# Transpose the DataFrame
selected_countries_df = selected_countries_df.T

# Plot the data
selected_countries_df.plot(figsize=(10, 6))
plt.xlabel('Year')
plt.ylabel('CO2 emissions (kg per PPP $ of GDP)')
plt.title('CO2 Emissions per PPP $ of GDP by Country')
plt.legend(title='Country')
plt.show()

# BAR CHARTS
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('CO2E_4904492.csv', skiprows=4)

# Select the relevant columns and rows
df_countries = df.loc[df['Country Name'].isin(["United States", "China", "United Kingdom", "Russian Federation", "France", "Japan", "Germany", "Israel", "United Arab Emirates", "Saudi Arabia"]), ['Country Name', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']]


# Set the index to be the country names
df_countries.set_index('Country Name', inplace=True)

# Set the figure size and create a new subplot
plt.figure(figsize=(10, 6))
ax = plt.subplot()

# Set the years and the number of bars per group
years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
n_bars = len(years)

# Set the bar width and the offset between the groups
bar_width = 0.8 / n_bars
offset = bar_width / 2

# Set the colors for each year
colors = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c', '#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']

# Set the x ticks to be the country names
x_ticks = df_countries.index

# Plot the bars for each year
for i, year in enumerate(years):
    ax.bar([j + offset + bar_width*i for j in range(len(x_ticks))], df_countries[year], width=bar_width, label=year, color=colors[i])

# Set the axis labels and title
ax.set_xlabel('Country')
ax.set_ylabel('CO2 emissions (kg per PPP $ of GDP)')
ax.set_title('CO2 Emissions by Country and Year')

# Set the x ticks and labels
ax.set_xticks([j + 0.4 for j in range(len(x_ticks))])
ax.set_xticklabels(x_ticks, rotation=60)

# Add a legend
ax.legend()

# Show the plot
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('CO2E_4904492.csv', skiprows=4)

# Select the relevant columns for the years 2010 and 2019
years = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
df_years = df[['Country Name'] + years]

# Select 10 countries of your choice
countries = ["United States", "China", "United Kingdom", "Russian Federation", "France", "Japan", "Germany", "Israel", "United Arab Emirates", "Saudi Arabia"]
df_countries = df_years[df_years['Country Name'].isin(countries)]

# Transpose the data to create a grouped bar chart
df_countries = df_countries.set_index('Country Name').transpose()

# Set the plot style
plt.style.use('ggplot')

# Create the plot
ax = df_countries.plot(kind='bar', figsize=(10,6))

# Set the axis labels and title
ax.set_xlabel('Year')
ax.set_ylabel('CO2 emissions (kg per PPP $ of GDP)')
ax.set_title('CO2 Emissions per PPP $ of GDP for 10 countries (2010-2019)')

# Show the plot
plt.show()


# A Data Frame With Different Indicators
import pandas as pd

# Read in the CSV files
df_CO2GDP = pd.read_csv('CO2E_4904492.csv', skiprows=4)
df_forest = pd.read_csv('FRST_5358376.csv', skiprows=4)
df_renewableElectricity = pd.read_csv('RNEW_5359592.csv', skiprows=4)
df_energyUse = pd.read_csv('PCAP_5358565.csv', skiprows=4)
df_population = pd.read_csv('TOTL_5358404.csv', skiprows=4)

# Select only the columns of interest
countries =["United States", "China", "United Kingdom", "Russian Federation", "France", "Japan", "Germany", "Israel", "United Arab Emirates", "Saudi Arabia"]
df_CO2GDP = df_CO2GDP.loc[df_CO2GDP['Country Name'].isin(countries), ['Country Name', '2014']]
df_forest = df_forest.loc[df_forest['Country Name'].isin(countries), ['Country Name', '2014']]
df_renewableElectricity = df_renewableElectricity.loc[df_renewableElectricity['Country Name'].isin(countries), ['Country Name', '2014']]
df_energyUse = df_energyUse.loc[df_energyUse['Country Name'].isin(countries), ['Country Name', '2014']]
df_population = df_population.loc[df_population['Country Name'].isin(countries), ['Country Name', '2014']]

# Merge the three dataframes into one based on 'Country Name'
df = pd.merge(df_CO2GDP, df_forest, on='Country Name')
df = pd.merge(df, df_renewableElectricity, on='Country Name')
df = pd.merge(df, df_energyUse, on='Country Name')
df = pd.merge(df, df_population, on='Country Name')

# Rename the columns
df.columns = ['year','CO2 per GDP', 'Forest_area', 'Renewable_electricity', 'Energy_use','Population_Total']

# Add a 'year' column with the value 2020
df['year'] = 2014

# Reorder the columns
df = df[['year','CO2 per GDP', 'Forest_area', 'Renewable_electricity', 'Energy_use','Population_Total']]

# Print the resulting dataframe

df


import matplotlib.pyplot as plt
import numpy as np

# Drop the year column
df_corr = df.drop('year', axis=1)

# Create a correlation matrix from the dataframe
corr_matrix = df_corr.corr()

# Create a heatmap using matshow from numpy and annotate the cells with the correlation values
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.matshow(corr_matrix, cmap='coolwarm')

for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        c = round(corr_matrix.iloc[i, j], 2)
        ax.text(j, i, str(c), va='center', ha='center')

# Add a colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Correlation', rotation=-90, va='bottom')

# Set the x and y axis tick labels to the column names
ax.set_xticklabels([''] + list(corr_matrix.columns))
ax.set_yticklabels([''] + list(corr_matrix.index))

# Rotate the x axis tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

# Set the title
ax.set_title('Correlation Heatmap For 2014')

# Show the plot
plt.show()

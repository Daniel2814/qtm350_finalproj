# Import necessary libraries
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# Establish a connection to the SQLite database
conn = sqlite3.connect("economic_data.db")
cursor = conn.cursor()

# Enable foreign key support
cursor.execute("PRAGMA foreign_keys = ON;")

print("Database connection established and foreign keys enabled.")

# Load the CSV file
gdp_growth = pd.read_csv("./data/API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_19358.csv", skiprows=4)
gdp_growth

# Select the relevant columns
columns_to_drop = ['Indicator Name', 'Indicator Code', 'Unnamed: 69']
gdp_growth = gdp_growth.drop(columns=columns_to_drop)

# Create SQLite table for GDP growth data
gdp_growth.to_sql("gdp_growth", conn, if_exists="replace", index=False)

# Load the CSV file
countries = pd.read_csv("./data/Metadata_Country_API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_19358.csv")

# Create SQLite table for countries data
countries.to_sql("countries", conn, if_exists="replace", index=False)

# List of years as strings
years = [str(year) for year in range(1960, 2025)]  # 1960–2024

# Join the years with commas and backticks (`) for SQL syntax
year_columns = ", ".join([f"`{year}`" for year in years])

# Create gdp_growth_clean table with selected columns and join with countries
query1 = f"""
DROP TABLE IF EXISTS gdp_growth_clean;
CREATE TABLE gdp_growth_clean AS
SELECT 
    gdp_growth.`Country Name` AS country_name,
    gdp_growth.`Country Code` AS country_code,
    {year_columns},
    countries.Region AS region,
    countries.IncomeGroup AS income_group
FROM gdp_growth
LEFT JOIN countries
ON gdp_growth.`Country Code` = countries.`Country Code`
WHERE
    countries.Region IS NOT NULL;
"""

conn.executescript(query1)
conn.commit()

# Create gdp_growth_long table in long format

query2 = "PRAGMA table_info(gdp_growth_clean);"
columns_info = pd.read_sql_query(query2, conn)

years_in_data = []

for col in columns_info['name']:
    # Check if the column name is an integer in the range 1960-2024
    if col.isdigit() and 1960 <= int(col) <= 2024:
        years_in_data.append(col)

union_queries = []

for year in years_in_data:
    union_queries.append(f"""
    SELECT 
        country_name,
        country_code,
        region,
        income_group,
        '{year}' AS year,
        `{year}` AS gdp_growth
    FROM gdp_growth_clean
    WHERE `{year}` IS NOT NULL
    """)

query3 = " UNION ALL ".join(union_queries)

create_table_query = f"""
DROP TABLE IF EXISTS gdp_growth_long;
CREATE TABLE gdp_growth_long AS
{query3}
"""

conn.executescript(create_table_query)
conn.commit()

print("Table gdp_growth_long created successfully.")

query4 = '''
-- Calculate the world average GDP growth for each year
WITH world_avg AS (
    SELECT 
        year, 
        AVG(gdp_growth) AS world_avg_growth
    FROM gdp_growth_long
    WHERE region IS NOT NULL  -- Ensure you're using valid regions (not countries without regions)
    GROUP BY year
)

-- Insert the world averages into the table
INSERT INTO gdp_growth_long (country_name, country_code, region, income_group, year, gdp_growth)
SELECT 
    'World' AS country_name, 
    NULL AS country_code,  -- No country code for the world
    'World' AS region,     -- Region labeled as 'World'
    'World' AS income_group,  -- No income group for the world
    world_avg.year,
    world_avg.world_avg_growth
FROM world_avg;
'''
conn.executescript(query4)
conn.commit()

# Create a new table for average GDP growth by decade
query5 = '''
SELECT
    ((CAST(year AS INTEGER) / 10) * 10) || 's' AS decade,
    COUNT(gdp_growth) AS count_growth,
    MIN(gdp_growth) AS min_growth,
    AVG(gdp_growth) AS avg_growth_decade,
    MAX(gdp_growth) AS max_growth
FROM
    gdp_growth_long
WHERE
    region != 'World'  -- Exclude the world average
GROUP BY
    decade
ORDER BY
    decade;
'''
# Execute the query and fetch the results
decade_growth = pd.read_sql_query(query5, conn)
decade_growth

# Create a new table for average GDP growth by region
query6 = '''
SELECT
    region,
    COUNT(DISTINCT country_name) AS n_countries,
    COUNT(gdp_growth) AS n_obs,
    AVG(gdp_growth) AS avg_growth,
    MIN(gdp_growth) AS min_growth,
    MAX(gdp_growth) AS max_growth
FROM
    gdp_growth_long
GROUP BY
    region
ORDER BY
    region;

'''
# Execute the query and fetch the results
region_avg = pd.read_sql_query(query6, conn)
region_avg

# Create a new table for average GDP growth by income group
query7 = '''
SELECT
    income_group,
    COUNT(DISTINCT country_name) AS n_countries,
    COUNT(gdp_growth) AS n_obs,
    MIN(gdp_growth) AS min_growth,
    AVG(gdp_growth) AS avg_growth,
    MAX(gdp_growth) AS max_growth
FROM
    gdp_growth_long
WHERE
    income_group IS NOT NULL
GROUP BY
    income_group
ORDER BY
    avg_growth DESC;
'''
# Execute the query and fetch the results
income_avg = pd.read_sql_query(query7, conn)
income_avg

# Create pandas DataFrame from gdp_growth_long table
gdp_growth_long = pd.read_sql_query("SELECT * FROM gdp_growth_long", conn)

# Close the database connection
conn.close()
print("Database connection closed.")

# Graph average GDP growth over time by region
avg_growth_by_year_region = gdp_growth_long.groupby(['year', 'region'])['gdp_growth'].mean().reset_index()

# Plot
plt.figure(figsize=(14, 8))


# Plot the average GDP growth by year and region (excluding World)
sns.lineplot(
    data=avg_growth_by_year_region[avg_growth_by_year_region['region'] != 'World'], 
    x='year', 
    y='gdp_growth', 
    hue='region'
)

# Manually add the world average line
world_avg_growth = gdp_growth_long[gdp_growth_long['region'] == 'World'].groupby('year')['gdp_growth'].mean().reset_index()

# Plot the world average line
plt.plot(world_avg_growth['year'], world_avg_growth['gdp_growth'], 
         color='black', linewidth=3, label='World Average')  # Black, thicker line for world average

# Set titles and labels
plt.title('Average GDP Growth Over Time by Region')
plt.xlabel('Year')
plt.ylabel('Average GDP Growth (%)')

# Adjust the legend to include world average
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.gca().xaxis.set_major_locator(MultipleLocator(10))

plt.grid(True)
plt.tight_layout()

plt.savefig("./figures/avg_gdp_growth_by_region.png")

# Graph average GDP growth over time by region with smoothed lines

# Make sure year is numeric
gdp_growth_long['year'] = gdp_growth_long['year'].astype(int)

# Create the plot
plt.figure(figsize=(14, 8))

# Plot with trend lines for regions (smoothing), excluding world data
sns.lmplot(
    data=gdp_growth_long[gdp_growth_long['region'].notnull() & (gdp_growth_long['region'] != 'World')],
    x='year',
    y='gdp_growth',
    hue='region',
    height=8,
    aspect=1.5,
    scatter=False,  # Don't plot individual points, just the smoothed line
    lowess=True,    # Smooth the lines
    legend=False     # Keep the legend with all regions
)

# Manually add the world average line (smoothed)
world_data = gdp_growth_long[gdp_growth_long['region'] == 'World']

# Plot the world average line, smoothed with lowess
sns.regplot(
    data=world_data,
    x='year',
    y='gdp_growth',
    scatter=False,  # No scatter points
    lowess=True,    # Smooth the world average line using lowess
    line_kws={'color': 'black', 'linewidth': 3}  # Black, thicker line for the world average
)

# Set titles and labels
plt.title('Smoothed GDP Growth Over Time by Region')
plt.xlabel('Year')
plt.ylabel('Average GDP Growth (%)')

# Adjust legend to include world average line
handles, labels = plt.gca().get_legend_handles_labels()

# Manually add the world average to the legend (only once)
handles.append(plt.Line2D([0], [0], color='black', linewidth=3))  # Add world average line handle
labels.append('World Average')  # Add world average label

# Display the legend and finalize the plot
plt.legend(handles=handles, labels=labels, title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)
plt.tight_layout()

plt.savefig("./figures/gdp_growth_by_region_trend.png")


# Graph average GDP growth over time by income group

# Group by both year and income group, calculate the mean
avg_growth_by_year_income = gdp_growth_long.groupby(['year', 'income_group'])['gdp_growth'].mean().reset_index()

# Plot the average GDP growth by year and income group (excluding World)
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=avg_growth_by_year_income[avg_growth_by_year_income['income_group'] != 'World'], 
    x='year', 
    y='gdp_growth', 
    hue='income_group'
)

# Manually add the world average line
world_avg_growth_income = gdp_growth_long[gdp_growth_long['income_group'] == 'World'].groupby('year')['gdp_growth'].mean().reset_index()

# Plot the world average line
plt.plot(world_avg_growth_income['year'], world_avg_growth_income['gdp_growth'], 
         color='black', linewidth=3, label='World Average')  # Black, thicker line for world average

# Set titles and labels
plt.title('Average GDP Growth Over Time by Income Group')
plt.xlabel('Year')
plt.ylabel('Average GDP Growth (%)')

# Get the current legend handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Define the desired order for the legend
legend_order = ['World', 'Low income', 'Lower middle income', 'Upper middle income', 'High income']

# Reorder the handles and labels based on the desired order
ordered_handles = []
ordered_labels = []

# Add world first
ordered_handles.append(plt.Line2D([0], [0], color='black', linewidth=3, label='World Average'))
ordered_labels.append('World Average')

# Add other regions in the desired order
for label in legend_order[1:]:
    if label in labels:
        idx = labels.index(label)
        ordered_handles.append(handles[idx])
        ordered_labels.append(labels[idx])

# Add the reordered legend to the plot
plt.legend(handles=ordered_handles, labels=ordered_labels, title='Income Group', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)
plt.tight_layout()

plt.savefig("./figures/avg_gdp_growth_by_income.png")


# Graph average GDP growth over time by income group with smoothed lines

# Group by year and income group, calculate the mean
avg_growth_by_year_income = gdp_growth_long.groupby(['year', 'income_group'])['gdp_growth'].mean().reset_index()

# Plot with smoothing for income groups
plt.figure(figsize=(14, 8))
sns.lmplot(
    data=gdp_growth_long[gdp_growth_long['income_group'].notnull() & (gdp_growth_long['region'] != 'World')],
    x='year',
    y='gdp_growth',
    hue='income_group',
    height=8,
    aspect=1.5,
    scatter=False,  # Don't plot individual points, just the smoothed line
    lowess=True,    # Smooth the lines
    legend=False     # Turn off the legend to customize it later
)

# Manually add the world average line (smoothed)
world_data = gdp_growth_long[gdp_growth_long['income_group'] == 'World']

# Plot the world average line, smoothed with lowess
sns.regplot(
    data=world_data,
    x='year',
    y='gdp_growth',
    scatter=False,  # No scatter points
    lowess=True,    # Smooth the world average line using lowess
    line_kws={'color': 'black', 'linewidth': 3}  # Black, thicker line for the world average
)

# Set titles and labels
plt.title('Smoothed Average GDP Growth Over Time by Income Group')
plt.xlabel('Year')
plt.ylabel('Average GDP Growth (%)')

# Get the current legend handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Define the desired order for the legend
legend_order = ['World', 'Low income', 'Lower middle income', 'Upper middle income', 'High income']

# Reorder the handles and labels based on the desired order
ordered_handles = []
ordered_labels = []

# Add world first
ordered_handles.append(plt.Line2D([0], [0], color='black', linewidth=3, label='World Average'))
ordered_labels.append('World Average')

# Add other regions in the desired order
for label in legend_order[1:]:
    if label in labels:
        idx = labels.index(label)
        ordered_handles.append(handles[idx])
        ordered_labels.append(labels[idx])

# Add the reordered legend to the plot
plt.legend(handles=ordered_handles, labels=ordered_labels, title='Income Group', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)
plt.tight_layout()

plt.savefig("./figures/avg_gdp_growth_by_income_trend.png")



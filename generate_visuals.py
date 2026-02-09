import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for visuals if it doesn't exist
output_dir = r'C:\Users\ndlan\.gemini\antigravity\brain\bbbf8f9f-9728-42bb-a014-7b825cd5bb9a\visuals'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset
csv_path = r'C:\Users\ndlan\Documents\Github files\Data Analysis\World Bank Dataset\world_bank_dataset.csv'
df = pd.read_csv(csv_path)

# Data Cleaning
df['GDP (USD)'] = pd.to_numeric(df['GDP (USD)'], errors='coerce')
df_clean = df.groupby('Country', group_keys=False).apply(lambda x: x.ffill().bfill())

# Set sleek styling
plt.style.use('ggplot')
sns.set_palette("viridis")

# 1. Life Expectancy vs GDP (Scatter with Trendline)
plt.figure(figsize=(12, 7))
sns.scatterplot(data=df_clean, x='GDP (USD)', y='Life Expectancy', hue='Year', size='Population', sizes=(50, 500), alpha=0.6)
plt.xscale('log')
plt.title('The Wealth-Health Correlation: Life Expectancy vs. GDP (Log Scale)', fontsize=14)
plt.xlabel('GDP (USD) - Log Scale', fontsize=12)
plt.ylabel('Life Expectancy (Years)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig(os.path.join(output_dir, 'gdp_vs_life_expectancy.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. CO2 Emissions vs GDP (Scatter)
plt.figure(figsize=(12, 7))
sns.scatterplot(data=df_clean, x='GDP (USD)', y='CO2 Emissions (metric tons per capita)', hue='Country', alpha=0.7)
plt.xscale('log')
plt.title('The Environmental Cost: GDP vs. CO2 Emissions', fontsize=14)
plt.xlabel('GDP (USD) - Log Scale', fontsize=12)
plt.ylabel('CO2 Emissions (metric tons per capita)', fontsize=12)
plt.savefig(os.path.join(output_dir, 'gdp_vs_co2.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Access to Electricity vs Life Expectancy
plt.figure(figsize=(12, 7))
sns.regplot(data=df_clean, x='Access to Electricity (%)', y='Life Expectancy', scatter_kws={'alpha':0.5, 'color':'teal'}, line_kws={'color':'red'})
plt.title('Energy as a Catalyst: Access to Electricity vs. Life Expectancy', fontsize=14)
plt.xlabel('Access to Electricity (%)', fontsize=12)
plt.ylabel('Life Expectancy (Years)', fontsize=12)
plt.savefig(os.path.join(output_dir, 'electricity_vs_life_expectancy.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df_clean.select_dtypes(include=['number']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Development Indicator Interdependencies (Correlation Matrix)', fontsize=14)
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Visuals generated and saved to: {output_dir}")

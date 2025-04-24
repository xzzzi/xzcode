import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("Results_21Mar2022.csv")
import seaborn as sns
columns_to_plot = [
    'mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut', 'mean_ghgs_ch4', 'mean_ghgs_n2o',
    'mean_bio', 'mean_watuse', 'mean_acid', 'sd_ghgs', 'sd_land', 'sd_watscar', 'sd_eut',
    'sd_ghgs_ch4', 'sd_ghgs_n2o', 'sd_bio', 'sd_watuse', 'sd_acid'
]
fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(columns_to_plot):
    sns.boxplot(x='sex', y=col, data=data, ax=axes[i], palette="Set2")
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
plt.tight_layout()
plt.savefig('boxplot.png',dpi=600)
plt.show()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[columns_to_plot] = scaler.fit_transform(data[columns_to_plot])
grouped_by_sex = data.groupby('sex')[columns_to_plot].mean()
grouped_by_diet = data.groupby('diet_group')[columns_to_plot].mean()
grouped_by_age = data.groupby('age_group')[columns_to_plot].mean()
from pandas.plotting import parallel_coordinates
color_map = plt.cm.get_cmap('tab20', len(grouped_by_sex))
colors = [color_map(i) for i in range(len(grouped_by_sex))]
plt.figure(figsize=(24, 9))
parallel_coordinates(grouped_by_sex.reset_index(), 'sex', color=colors)
plt.title('Parallel Coordinates for Environmental Impacts by Sex')
plt.xlabel('Environmental Factors')
plt.ylabel('Mean Values')
plt.grid(True)
plt.savefig('parallel_coordinates_by_sex.png',dpi=600)
plt.show()
age_diet_counts = data.groupby(['age_group', 'diet_group']).size().unstack(fill_value=0)
import numpy as np
import matplotlib.pyplot as plt
def create_radar_chart(data, labels, group_name, color, ax):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    data = data.tolist()
    data += data[:1]
    angles += angles[:1]
    ax.fill(angles, data, color=color, alpha=0.25)
    ax.plot(angles, data, color=color, linewidth=2)
labels = columns_to_plot
fig, ax = plt.subplots(figsize=(8, 8), dpi=80, subplot_kw=dict(polar=True))
diet_colors = {diet: np.random.rand(3,) for diet in grouped_by_diet.index}
for diet, data1 in grouped_by_diet.iterrows():
    color = diet_colors[diet]
    create_radar_chart(data1, labels, diet, color, ax)
ax.set_yticklabels([])
ax.set_xticks(np.linspace(0, 2 * np.pi, len(labels), endpoint=False))
ax.set_xticklabels(labels, rotation=45)
diet_names = list(grouped_by_diet.index)
diet_labels = [f"{diet}" for diet in diet_names]
colors = [diet_colors[diet] for diet in diet_names]
plt.title('Radar Chart for Different Diet Groups', size=15, color='black', fontweight='bold')
plt.savefig('radar_chart_by_diet_group.png', dpi=600)
plt.show()
fig, ax = plt.subplots(figsize=(8, 8), dpi=80, subplot_kw=dict(polar=True))
for age, data1 in grouped_by_age.iterrows():
    color = np.random.rand(3,)
    create_radar_chart(data1, labels, age, color, ax)
ax.set_yticklabels([])
ax.set_xticks(np.linspace(0, 2 * np.pi, len(labels), endpoint=False))
ax.set_xticklabels(labels, rotation=45)
plt.legend(grouped_by_age.index, loc='upper right', bbox_to_anchor=(1.5, 1))
plt.title('Radar Chart for Different Diet Groups', size=15, color='black', fontweight='bold')
plt.savefig('radar_chart_by_age_group.png',dpi=600)
plt.show()
import seaborn as sns
corr_data = data[columns_to_plot]
corr_matrix = corr_data.corr()
plt.figure(figsize=(16, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Environmental Impacts')
plt.savefig('correlation_matrix.png',dpi=600)
plt.show()
g = sns.PairGrid(data, hue="sex",vars = columns_to_plot)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()
g.savefig('pairplot.png',dpi=600)
import numpy as np
grouped_by_sex_diet = data.groupby(['sex', 'diet_group'])[columns_to_plot].mean().reset_index()
male_data = grouped_by_sex_diet[grouped_by_sex_diet['sex'] == 'male']
female_data = grouped_by_sex_diet[grouped_by_sex_diet['sex'] == 'female']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
columns_to_plot_mean = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut', 'mean_ghgs_ch4', 'mean_ghgs_n2o','mean_bio', 'mean_watuse', 'mean_acid']
columns_to_plot_sd = ['sd_ghgs', 'sd_land', 'sd_watscar', 'sd_eut', 'sd_ghgs_ch4', 'sd_ghgs_n2o','sd_bio', 'sd_watuse', 'sd_acid']
for i, column in enumerate(columns_to_plot_mean):
    ax = axes[i // 3, i % 3]
    male_group = male_data.groupby('diet_group')[column].mean()
    female_group = female_data.groupby('diet_group')[column].mean()
    ax.barh(male_group.index, -male_group.values, label='Male', color='#6699FF', align='center')
    ax.barh(female_group.index, female_group.values, label='Female', color='#CC6699', align='center')
    ax.set_title(f'Population Pyramid for {column}', fontsize=14)
    ax.set_xlabel('Impact Value (Normalized)', fontsize=12)
    ax.set_ylabel('Diet Group', fontsize=12)
    ax.legend()
plt.tight_layout()
plt.savefig('population_pyramid_mean.png',dpi=600)
plt.show()
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
for i, column in enumerate(columns_to_plot_sd):
    ax = axes[i // 3, i % 3]
    male_group = male_data.groupby('diet_group')[column].mean()
    female_group = female_data.groupby('diet_group')[column].mean()
    ax.barh(male_group.index, -male_group.values, label='Male', color='#6699FF', align='center')
    ax.barh(female_group.index, female_group.values, label='Female', color='#CC6699', align='center')
    ax.set_title(f'Population Pyramid for {column}', fontsize=14)
    ax.set_xlabel('Impact Value (Normalized)', fontsize=12)
    ax.set_ylabel('Diet Group', fontsize=12)
    ax.legend()
plt.tight_layout()
plt.savefig('population_pyramid_sd.png',dpi=600)
plt.show()

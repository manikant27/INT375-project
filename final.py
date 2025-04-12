import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("Admission Chance.csv") 
df.describe()
df.info()
print(df.head(5))

# Clean column names
df.columns = df.columns.str.strip()

df.fillna(df.mean(numeric_only=True), inplace=True)

df = df.drop(columns=['Serial No'])


# Objective 1
# Correlation analysis
corr_matrix = df.corr()


plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Analyzing Interdependence Among Admission Factors", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
# plt.savefig("cgpa_vs_admit1.png", dpi=300, bbox_inches='tight')
plt.show()

# Objective 2
#Analyze the Impact of GRE Score on Admission Chances


plt.figure(figsize=(8, 6))
sns.regplot(data=df, x="GRE Score", y="Chance of Admit", scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title("GRE Score vs Chance of Admission")
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admit")
plt.grid(True)
plt.tight_layout()
# plt.savefig("cgpa_vs_admit2.png", dpi=300, bbox_inches='tight')
plt.show()


# Objective 3: Distribution of University Ratings


plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df, x="University Rating", palette="viridis")


plt.title("Distribution of University Ratings")
plt.xlabel("University Rating")
plt.ylabel("Number of Applicants")

plt.tight_layout()
# plt.savefig("cgpa_vs_admit3.png", dpi=300, bbox_inches='tight')
plt.show()

avg_scores = df.groupby("University Rating")[["GRE Score", "CGPA"]].mean().reset_index()

ax = avg_scores.plot(
    x="University Rating", 
    kind="bar", 
    figsize=(9, 6), 
    colormap="Set2", 
    title="Average GRE Score and CGPA by University Rating"
)

# Add data labels on each bar
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', (bar.get_x() + bar.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10)

plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
# plt.savefig("cgpa_vs_admit4.png", dpi=300, bbox_inches='tight')
plt.show()


#Objective 4

# Objective 4: Percentage of Students with Research Experience
research_counts = df['Research'].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(research_counts, labels=['No Research', 'Has Research'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
plt.title('Research Participation')
# plt.savefig("cgpa_vs_admit5.png", dpi=300, bbox_inches='tight')
plt.show()

# Select feature columns
feature_cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA']

# normalize 1-100
df_normalized = df.copy()
for col in feature_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    df_normalized[col] = ((df[col] - min_val) / (max_val - min_val)) * 100


research_profile = df_normalized.groupby('Research')[feature_cols].mean().T
research_profile.columns = ['No Research', 'Research']

ax = research_profile.plot(kind='bar', figsize=(10, 6), colormap='coolwarm')


for p in ax.patches:
    value = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, value + 1.5, f'{value:.1f}',
            ha='center', va='bottom', fontsize=9, color='black')

plt.title("Normalized (0–100) Feature Comparison: Research vs No Research")
plt.ylabel("Average Normalized Score")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
# plt.savefig("cgpa_vs_admit6.png", dpi=300, bbox_inches='tight')
plt.show()





# Objective 5: Relationship Between CGPA and Admission

df_sorted = df.sort_values('CGPA')
plt.figure(figsize=(6, 4))
sns.lineplot(x='CGPA', y='Chance of Admit', data=df_sorted)
plt.title('CGPA vs Chance of Admit')
# plt.savefig("cgpa_vs_admit7.png", dpi=300, bbox_inches='tight')
plt.show()


# objective 6
# Distribution of TOEFL Scores
plt.figure(figsize=(10, 6))
hist = sns.histplot(df['TOEFL Score'], kde=True, bins=15, color='skyblue', edgecolor='black')


for patch in hist.patches:
    height = patch.get_height()
    if height > 0:
        plt.text(patch.get_x() + patch.get_width() / 2, height + 0.5,
                 int(height), ha='center', va='bottom', fontsize=9, color='black')

plt.title("Distribution of TOEFL Scores")
plt.xlabel("TOEFL Score")
plt.ylabel("Number of Students")
plt.grid(axis='y')
plt.tight_layout()
# plt.savefig("cgpa_vs_admit8.png", dpi=300, bbox_inches='tight')
plt.show()









## 1. Imports and Configuration

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from scipy.stats import boxcox

# Dimensionality Reduction
from sklearn.decomposition import PCA

# Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Feature selection
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Regression models
from sklearn.linear_model import LinearRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Classification evaluation
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score
)

# Regression evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Plot settings
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (10, 6)

print("All libraries loaded successfully.")

"""## 2. Data Loading and Initial Inspection

The dataset has 100,000 rows and 32 columns. Three columns are prediction targets. All three are retained in the dataframe but excluded from the feature matrix when building any individual pipeline. This prevents target leakage: none of the three targets is ever used as a feature when predicting another.

`person_id` is a row identifier with no predictive value and is dropped immediately.

"""

from google.colab import drive
drive.mount('/content/drive')

DATA_PATH = '/content/drive/MyDrive/Colab Notebooks/sleep_health_dataset.csv'

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.head()

# Drop identifier only
df.drop(columns=['person_id'], inplace=True)

print(f"Columns after dropping person_id: {df.shape[1]}")
print()
print("Data types:")
print(df.dtypes)

# Null and duplicate check
print(f"Total null values : {df.isnull().sum().sum()}")
print(f"Duplicate rows    : {df.duplicated().sum()}")
print()

# Column groups
TARGET_COLS = ['sleep_disorder_risk', 'cognitive_performance_score', 'felt_rested']

NUMERIC_CONTINUOUS = [
    'age', 'bmi',
    'sleep_duration_hrs', 'sleep_quality_score', 'rem_percentage',
    'deep_sleep_percentage', 'sleep_latency_mins', 'wake_episodes_per_night',
    'caffeine_mg_before_bed', 'alcohol_units_before_bed',
    'screen_time_before_bed_mins', 'steps_that_day', 'nap_duration_mins',
    'stress_score', 'work_hours_that_day',
    'heart_rate_resting_bpm', 'room_temperature_celsius', 'weekend_sleep_diff_hrs'
]

BINARY_COLS      = ['exercise_day', 'sleep_aid_used', 'shift_work']
CATEGORICAL_COLS = ['gender', 'occupation', 'country', 'chronotype',
                    'mental_health_condition', 'season', 'day_type']

print(f"Feature groups defined:")
print(f"  Numeric continuous : {len(NUMERIC_CONTINUOUS)}")
print(f"  Binary flags       : {len(BINARY_COLS)}")
print(f"  Categorical        : {len(CATEGORICAL_COLS)}")
print(f"  Targets            : {TARGET_COLS}")

"""## 3. Exploratory Data Analysis

The EDA covers all three targets and their relationships with the input features. Because the targets are correlated with each other, we also examine their pairwise relationships to understand the structure of the problem before modeling.

"""

# All three targets: distributions
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Target A: sleep_disorder_risk (categorical)
risk_order    = ['Healthy', 'Mild', 'Moderate', 'Severe']
risk_palette  = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
risk_counts   = df['sleep_disorder_risk'].value_counts().reindex(risk_order).dropna()

axes[0].bar(risk_counts.index, risk_counts.values, color=risk_palette, edgecolor='white')
axes[0].set_title('A: sleep_disorder_risk\n(Multiclass)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(risk_counts.values):
    axes[0].text(i, v + 300, f'{v/len(df)*100:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Target B: cognitive_performance_score (continuous)
axes[1].hist(df['cognitive_performance_score'], bins=40,
             color='steelblue', edgecolor='white', alpha=0.85)
axes[1].axvline(df['cognitive_performance_score'].mean(), color='red',
                linestyle='--', linewidth=1.5,
                label=f"Mean: {df['cognitive_performance_score'].mean():.1f}")
axes[1].set_title('B: cognitive_performance_score\n(Regression)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Score')
axes[1].set_ylabel('Frequency')
axes[1].legend(fontsize=9)

# Target C: felt_rested (binary)
rested_counts = df['felt_rested'].value_counts().sort_index()
axes[2].bar(['Not Rested (0)', 'Felt Rested (1)'],
            rested_counts.values, color=['#EF5350', '#66BB6A'], edgecolor='white')
axes[2].set_title('C: felt_rested\n(Binary Classification)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Count')
for i, v in enumerate(rested_counts.values):
    axes[2].text(i, v + 300, f'{v/len(df)*100:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Distribution of All Three Prediction Targets', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_all_targets.png', bbox_inches='tight')
plt.show()

"""### Interpretation: Target Distributions

-   **`sleep_disorder_risk` (Multiclass Classification):** This target shows significant class imbalance. The 'Healthy' category dominates with approximately 54% of the data, while 'Severe' constitutes only about 4%. This imbalance, a ratio of roughly 13:1 between the most and least frequent classes, suggests that models will likely be biased towards predicting the majority 'Healthy' class if not properly addressed (e.g., through class weighting or resampling techniques).

-   **`cognitive_performance_score` (Regression):** The distribution of cognitive performance scores appears approximately normal, centered around a mean of 59.2. Its symmetrical shape suggests that direct modeling, without any target transformation, should be appropriate for this variable.

-   **`felt_rested` (Binary Classification):** This target exhibits moderate class imbalance. Approximately 61% of individuals did not feel rested (0), while 39% did (1). While less severe than `sleep_disorder_risk`, this imbalance is still significant enough to warrant strategies like class weighting during model training to ensure the minority class ('Felt Rested') is adequately learned by classifiers.
"""

# Cross-target relationships
# Since none of the targets should be used as features for each other,
# understanding their correlation helps frame the interpretation section.

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Cognitive score by sleep disorder risk
sns.boxplot(data=df, x='sleep_disorder_risk', y='cognitive_performance_score',
            order=risk_order, palette=risk_palette, ax=axes[0])
axes[0].set_title('Cognitive Score by Sleep Disorder Risk', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Sleep Disorder Risk')
axes[0].tick_params(axis='x', rotation=15)

# Cognitive score by felt_rested
sns.boxplot(data=df, x='felt_rested', y='cognitive_performance_score',
               palette=['#EF5350', '#66BB6A'], ax=axes[1])
axes[1].set_title('Cognitive Score by Felt Rested', fontsize=12, fontweight='bold')
axes[1].set_xticklabels(['Not Rested', 'Felt Rested'])

plt.suptitle('Cross-Target Relationships', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_cross_target.png', bbox_inches='tight')
plt.show()

# Felt rested rate by risk class
print("Felt rested rate by sleep disorder risk:")
print(df.groupby('sleep_disorder_risk')['felt_rested'].mean().reindex(risk_order).round(3))

"""### Interpretation: Cross-Target Relationships

The visualizations reveal clear and intuitive relationships between the three prediction targets:

-   **Cognitive Score by Sleep Disorder Risk:** As sleep disorder risk progresses from 'Healthy' to 'Severe', there is a noticeable decline in `cognitive_performance_score`. The median cognitive score decreases consistently, with the 'Severe' group exhibiting significantly lower scores and a narrower interquartile range, indicating poorer and more consistently low cognitive function among individuals with severe sleep disorders.

-   **Cognitive Score by Felt Rested:** Individuals who `felt_rested` tend to have higher `cognitive_performance_score` values compared to those who did not. The distribution for 'Felt Rested' is shifted towards higher scores, indicating a positive correlation between feeling rested and cognitive performance.

-   **Felt Rested Rate by Sleep Disorder Risk:** The proportion of individuals who `felt_rested` decreases dramatically with increasing `sleep_disorder_risk`. Only 56.6% of 'Healthy' individuals felt rested, which drops to a mere 2.6% for those with 'Severe' risk. This strong inverse relationship confirms that higher sleep disorder risk is associated with a significantly reduced likelihood of feeling rested.

Overall, these findings underscore the interconnectedness of these sleep-related outcomes: poorer sleep architecture and higher disorder risk lead to a lower likelihood of feeling rested, which in turn correlates with diminished cognitive performance.
"""

# Sleep Architecture Features vs Sleep Disorder Risk
sleep_arch = ['sleep_duration_hrs', 'sleep_quality_score', 'rem_percentage',
              'deep_sleep_percentage', 'sleep_latency_mins', 'wake_episodes_per_night']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(sleep_arch):
    sns.boxplot(data=df, x='sleep_disorder_risk', y=col,
                order=risk_order, palette=risk_palette, ax=axes[i])
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=15)

plt.suptitle('Sleep Architecture vs Sleep Disorder Risk', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_sleep_arch_vs_risk.png', bbox_inches='tight')
plt.show()

"""### Interpretation: Sleep Architecture vs Sleep Disorder Risk

These box plots illustrate distinct patterns in sleep architecture features across different levels of `sleep_disorder_risk`:

-   **`sleep_duration_hrs` and `sleep_quality_score`:** Both show a clear monotonic decrease as the `sleep_disorder_risk` level increases. 'Healthy' individuals tend to have longer sleep durations and higher quality scores, while 'Severe' risk individuals exhibit the shortest durations and lowest quality, suggesting these are crucial indicators of sleep health.

-   **`rem_percentage` and `deep_sleep_percentage`:** Similar to sleep duration and quality, the percentages of REM and deep sleep generally decrease with higher `sleep_disorder_risk`. This indicates that the amount of time spent in restorative sleep stages is negatively impacted by sleep disorders.

-   **`sleep_latency_mins` and `wake_episodes_per_night`:** These features show an inverse trend. As `sleep_disorder_risk` increases, both `sleep_latency_mins` (time to fall asleep) and `wake_episodes_per_night` increase. This means individuals with higher risk levels take longer to initiate sleep and experience more frequent awakenings, signifying more disturbed sleep.
"""

# Psychological and lifestyle features vs sleep disorder risk
psych_life = ['stress_score', 'work_hours_that_day', 'caffeine_mg_before_bed',
              'alcohol_units_before_bed', 'screen_time_before_bed_mins', 'steps_that_day']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(psych_life):
    sns.boxplot(data=df, x='sleep_disorder_risk', y=col,
                order=risk_order, palette=risk_palette, ax=axes[i])
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=15)

plt.suptitle('Psychological and Lifestyle Features vs Sleep Disorder Risk',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('04_psych_lifestyle_vs_risk.png', bbox_inches='tight')
plt.show()

"""### Interpretation: Psychological and Lifestyle Features vs Risk

The relationship between psychological/lifestyle factors and `sleep_disorder_risk` is revealed through these box plots:

-   **`stress_score`:** This feature shows the most dramatic separation. Median stress scores rise significantly and consistently with increasing `sleep_disorder_risk`, and the interquartile ranges show minimal overlap between healthy and severe groups. This strongly suggests that stress is a primary driver of sleep disorder risk.

-   **`work_hours_that_day`:** A similar, though less pronounced, trend is observed for work hours. Higher work hours are associated with increased sleep disorder risk, indicating a potential link between workload and sleep health.

-   **`caffeine_mg_before_bed` and `alcohol_units_before_bed`:** Both caffeine and alcohol consumption before bed tend to be higher in individuals with greater `sleep_disorder_risk`. However, the distributions show considerable overlap, suggesting these are contributing factors but perhaps not as strong as stress.

-   **`screen_time_before_bed_mins`:** A slight upward trend is visible, where more screen time before bed correlates with increased risk, though the effect appears marginal.

-   **`steps_that_day`:** This feature shows a general trend: individuals with lower `sleep_disorder_risk` tend to have higher daily step counts. This observed association suggests that greater physical activity is linked to lower sleep disorder risk.
"""

# Categorical features vs Sleep disorder risk
cat_plot = ['mental_health_condition', 'chronotype', 'shift_work',
            'occupation', 'gender']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(cat_plot):
    ct = pd.crosstab(df[col], df['sleep_disorder_risk'], normalize='index') * 100
    ct = ct.reindex(columns=[c for c in risk_order if c in ct.columns])
    ct.plot(kind='bar', stacked=True, ax=axes[i],
            color=risk_palette[:len(ct.columns)], edgecolor='white')
    axes[i].set_title(f'{col} vs Risk Level (%)', fontsize=10, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].legend(title='Risk', fontsize=7, loc='upper right')

axes[5].set_visible(False)
plt.suptitle('Categorical Features vs Sleep Disorder Risk', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('05_categorical_vs_risk.png', bbox_inches='tight')
plt.show()

"""### Interpretation: Categorical Features vs Sleep Disorder Risk

The stacked bar charts highlight how categorical features relate to `sleep_disorder_risk`:

-   **`mental_health_condition`:** This feature exhibits the most striking differences. Individuals with 'Anxiety', 'Depression', or 'Both' conditions show a significantly higher proportion of 'Moderate' and 'Severe' sleep disorder risk compared to those classified as 'Healthy'. This underscores the strong interplay between mental health and sleep.

-   **`chronotype`:** 'Evening' chronotypes show a higher prevalence of 'Moderate' and 'Severe' risk compared to 'Morning' types. This aligns with research on circadian rhythms and sleep disruption.

-   **`shift_work`:** Individuals engaged in `shift_work` have a higher proportion of 'Severe' sleep disorder risk. This is consistent with the known negative impact of irregular work schedules on sleep patterns.

-   **`occupation` and `gender`:** While these categories show some variations, their impact on `sleep_disorder_risk` is less pronounced compared to `mental_health_condition`, `chronotype`, and `shift_work`.
"""

# Correlation with Cognitive performance score
numeric_for_corr = NUMERIC_CONTINUOUS + BINARY_COLS

corr_with_cog = df[numeric_for_corr + ['cognitive_performance_score']].corr()[
    'cognitive_performance_score'
].drop('cognitive_performance_score').sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#4CAF50' if v > 0 else '#F44336' for v in corr_with_cog]
ax.barh(corr_with_cog.index, corr_with_cog.values, color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Pearson Correlation with cognitive_performance_score',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Pearson r')
plt.tight_layout()
plt.savefig('06_corr_cognitive.png', bbox_inches='tight')
plt.show()

print("Top correlates with cognitive_performance_score:")
print(corr_with_cog.head(10).round(3).to_string())

"""### Interpretation: Pearson Correlation with Cognitive Performance Score

The Pearson correlation analysis reveals the linear relationships between numeric features and `cognitive_performance_score`:

-   **Strongest Positive Correlates:** `sleep_quality_score` (0.860) and `sleep_duration_hrs` (0.618) show very strong positive correlations. This indicates that better sleep quality and longer sleep duration are highly associated with higher cognitive performance.

-   **Strongest Negative Correlates:** `stress_score` (-0.593) and `wake_episodes_per_night` (-0.295) exhibit strong negative correlations. Higher stress levels and more frequent awakenings are linked to lower cognitive performance.

-   **Moderate Correlates:** `rem_percentage` (0.446), `work_hours_that_day` (-0.346), `deep_sleep_percentage` (0.280), `exercise_day` (0.255), `shift_work` (-0.254), and `alcohol_units_before_bed` (-0.240) show moderate correlations, further highlighting the influence of sleep architecture and lifestyle on cognitive function.

-   **Weak or Negligible Correlates:** Some features, like `bmi`, `heart_rate_resting_bpm`, and `weekend_sleep_diff_hrs`, exhibit very low Pearson correlation coefficients. This does not necessarily mean they are uninformative, as their relationship with cognitive performance might be non-linear or indirect, which a linear correlation measure might not capture. Mutual information, used later in feature selection, can better identify such non-linear dependencies.

### Visualization: Distributions of Numeric Continuous Features

Before performing the skewness audit numerically, let's visualize the distributions of all numeric continuous features using histograms. This will provide a visual understanding of their shape and help in identifying skewed features.
"""

n_cols = 4
n_rows = (len(NUMERIC_CONTINUOUS) + n_cols - 1) // n_cols # Calculate rows needed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(NUMERIC_CONTINUOUS):
    axes[i].hist(df[col], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[i].set_title(f'{col} (Skew: {df[col].skew():.2f})', fontsize=10, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Distributions of Numeric Continuous Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('numeric_features_distributions.png', bbox_inches='tight')
plt.show()

"""**Interpretation:** Each histogram shows one feature's distribution shape. The skewness value in the title is the key number — above 1.0 or below -1.0 means the feature is heavily skewed and needs transformation. Features like caffeine and alcohol show a spike at zero (most people consume none) with a long right tail, these are the clearest features for transformation."""

# Skewness audit
skew_df = df[NUMERIC_CONTINUOUS].skew().reset_index()
skew_df.columns = ['Feature', 'Skewness']
skew_df['Abs_Skewness'] = skew_df['Skewness'].abs()
skew_df = skew_df.sort_values('Skewness', ascending=False).reset_index(drop=True)

print("Feature skewness (sorted):")
print(skew_df.to_string(index=False))

high_skew = skew_df[skew_df['Abs_Skewness'] > 1.0]['Feature'].tolist()
mod_skew  = skew_df[(skew_df['Abs_Skewness'] > 0.5) &
                    (skew_df['Abs_Skewness'] <= 1.0)]['Feature'].tolist()

"""**Interpretation:** Features with `Abs_Skewness > 1.0` are flagged as highly skewed. Raw skewed distributions can distort regression coefficients and reduce model accuracy. Features near zero skewness are already well-shaped and need no changes.

### Detailed Skewness Analysis and Transformations

For the features identified as highly skewed (`|skew| > 1.0`) and moderately skewed (`|skew| > 0.5 && |skew| <= 1.0`), we will now perform a more detailed analysis. This involves visualizing the original distribution alongside distributions after applying common transformations: `log1p`, `square-root`, and `Box-Cox`. We will also compare their numerical skewness to evaluate the effectiveness of each transformation.
"""

from scipy.stats import skew, boxcox

for feature in (high_skew + mod_skew):
    original_data = df[feature]

    # Log1p transformation (handles zero values gracefully)wh
    log_data = np.log1p(original_data)

    # Square-root transformation (handles zero values gracefully)
    sqrt_data = np.sqrt(original_data)

    # Box-Cox transformation requires strictly positive values.
    # Since these features can have zeros, we add a small constant (1) to make them positive.
    # The lambda parameter is automatically determined by boxcox.
    boxcox_data, _ = boxcox(original_data + 1)
    boxcox_title_suffix = " (shifted by 1)"

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Distribution of {feature} and its Transformations', fontsize=16)

    axes[0, 0].hist(original_data, bins=30, edgecolor="black")
    axes[0, 0].set_title(f"Original (Skew: {skew(original_data):.2f})")

    axes[0, 1].hist(log_data, bins=30, edgecolor="black")
    axes[0, 1].set_title(f"Log1p Transform (Skew: {skew(log_data):.2f})")

    axes[1, 0].hist(sqrt_data, bins=30, edgecolor="black")
    axes[1, 0].set_title(f"Square-Root Transform (Skew: {skew(sqrt_data):.2f})")

    axes[1, 1].hist(boxcox_data, bins=30, edgecolor="black")
    axes[1, 1].set_title(f"Box-Cox Transform{boxcox_title_suffix} (Skew: {skew(boxcox_data):.2f})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

    # Skewness comparison table
    skew_table = pd.DataFrame({
        "Version": ["Original", "Log1p", "Square-root", f"Box-Cox{boxcox_title_suffix}"],
        "Skewness": [
            skew(original_data),
            skew(log_data),
            skew(sqrt_data),
            skew(boxcox_data)
        ]
    })
    print(f"Skewness Comparison for {feature}:")
    display(skew_table)
    print("\n") # Add a newline for better separation in the output

"""### **Interpretation: Skewness Audit**

The skewness audit revealed that many numeric continuous features, particularly `caffeine_mg_before_bed`, `alcohol_units_before_bed`, `nap_duration_mins`, and `screen_time_before_bed_mins`, exhibited significant positive skewness. This implies that their distributions are concentrated towards lower values, with a long tail extending towards higher values. Such skewed distributions can negatively impact the performance of various machine learning models, especially those that assume normally distributed data.

To address this, different transformations were tested, including `log1p`, `square-root`, and `Box-Cox` (with a +1 shift to handle zero values).

-   For highly skewed features, especially those containing zero values, the `Box-Cox` transformation (with a +1 shift) generally proved most effective in normalizing the distributions and reducing skewness towards zero. This transformation finds an optimal power parameter (`lambda`) that best approximates a normal distribution.
-   For moderately skewed features like `age`, `log1p` transformation was also effective in bringing the skewness closer to zero.

Therefore, a strategy of applying `Box-Cox` (with a +1 shift) to highly skewed, zero-inflated features and `log1p` to moderately skewed features is adopted to improve the normality of these distributions, which is crucial for enhancing the performance and interpretability of subsequent machine learning models.

### **EDA Summary**

This exploratory data analysis has provided several key insights that will inform the modeling strategy for the three prediction targets:

-   **Target Imbalance:** Both `sleep_disorder_risk` and `felt_rested` exhibit significant class imbalance. `sleep_disorder_risk` has an extreme 13:1 ratio between 'Healthy' and 'Severe' classes, while `felt_rested` has a moderate imbalance (61% 'Not Rested' vs 39% 'Felt Rested'). This necessitates the use of techniques like `class_weight='balanced'` in classification models to prevent them from simply predicting the majority class.

-   **Cross-Target Structure:** The targets are not independent. There are strong correlations where higher sleep disorder risk is associated with lower cognitive performance and a reduced likelihood of feeling rested. This implies that the features predictive of one target will likely also hold predictive power for the others.

-   **Key Feature Signals:** Stress score, sleep quality score, sleep duration, mental health condition, and wake episodes per night emerged as highly influential features. These show clear and intuitive relationships with sleep health outcomes, suggesting they will be strong predictors across the different models.

-   **Skewness Treatment:** Several numeric continuous features were identified as highly or moderately skewed. For highly skewed and zero-inflated features (e.g., `caffeine_mg_before_bed`, `alcohol_units_before_bed`, `nap_duration_mins`, `screen_time_before_bed_mins`), the `Box-Cox` transformation (with a +1 shift) proved most effective in reducing skewness. For moderately skewed features (`age`), the `log1p` transformation was applied. These transformations aim to normalize feature distributions, which can improve model performance.

## 4. Preprocessing (Shared Pipeline)

This section builds a single preprocessed dataframe that all three target pipelines will draw from.

Steps include:

1. Encode categorical features
2. Encode `sleep_disorder_risk` (to be used as target in Pipeline A)
3. Apply Box-Cox to moderately and highly skewed features
5. Leave binary flags unchanged
"""

df_proc = df.copy()

# Encode categorical features
le = LabelEncoder()
encoding_maps = {}

for col in CATEGORICAL_COLS:
    df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    encoding_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col}: {encoding_maps[col]}")

print("Categorical encoding complete.")

# Encode sleep_disorder_risk
RISK_MAP    = {'Healthy': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
RISK_NAMES  = ['Healthy', 'Mild', 'Moderate', 'Severe']
df_proc['risk_encoded'] = df_proc['sleep_disorder_risk'].map(RISK_MAP)

print("sleep_disorder_risk encoded: Healthy=0, Mild=1, Moderate=2, Severe=3")
print(df_proc['risk_encoded'].value_counts().sort_index()
      .rename({0:'Healthy',1:'Mild',2:'Moderate',3:'Severe'}))

"""### Interpretation: Encoding of `sleep_disorder_risk`

The `sleep_disorder_risk` feature is a categorical variable with four distinct levels: 'Healthy', 'Mild', 'Moderate', and 'Severe'. These levels inherently possess an order or hierarchy, representing increasing severity of sleep disorder.

**Why Ordinal Encoding is Appropriate Here:**

*   **Preserves Order:** The custom `RISK_MAP` (`{'Healthy': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}`) assigns numerical values that reflect this intrinsic order. This is crucial for models that are sensitive to numerical relationships (e.g., linear models, tree-based models where splits can naturally leverage this order).
*   **Efficiency:** It uses a single numerical column, which is memory-efficient and avoids the 'curse of dimensionality' that can arise with one-hot encoding for many categories.


"""

# 4.3 Transformations
# Box-Cox transformation (with +1 shift to handle zero values).
# Applied to features with extreme right skew (|skew| > 1) and large zero mass.

BOX_COX_COLS = [
    'caffeine_mg_before_bed',
    'alcohol_units_before_bed',
    'nap_duration_mins',
    'screen_time_before_bed_mins',
]

LOG1P_COLS = [
    'age',
]

# Using scipy.stats.boxcox with a +1 shift for zero-containing features
for col in BOX_COX_COLS:
    # Adding 1 to make data strictly positive for Box-Cox
    df_proc[col] = boxcox(df_proc[col] + 1)[0]

# Apply log1p for other moderately skewed features
for col in LOG1P_COLS:
    df_proc[col] = np.log1p(df_proc[col])

print("Transformations applied.")
print("Post-transform skewness:")
for col in BOX_COX_COLS + LOG1P_COLS:
    print(f" {col:<35}: Before: {df[col].skew():.3f},  After: {df_proc[col].skew():.3f}")

"""**Interpretation:**

Box-Cox finds the mathematically optimal power transformation for each feature individually, making it more effective than log1p for zero-inflated data.

From the 'After' skewness values, we can notice that all the values are closer to zero and more reduced than before, confirming the distributions are now approximately symmetric.
"""

# 4.4 Before vs after: visual check
check_cols = BOX_COX_COLS + LOG1P_COLS

fig, axes = plt.subplots(2, 5, figsize=(20, 9))

for i, col in enumerate(check_cols):
    axes[0, i].hist(df[col], bins=40, color='coral', edgecolor='white', alpha=0.85)
    axes[0, i].set_title(f'{col} BEFORE (skew: {df[col].skew():.2f})', fontsize=9, fontweight='bold')
    axes[1, i].hist(df_proc[col], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1, i].set_title(f'AFTER (skew: {df_proc[col].skew():.2f})', fontsize=9, fontweight='bold')

plt.suptitle('Transformation: Before vs After', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('07_transformation.png', bbox_inches='tight')
plt.show()

"""**Interpretation:**

The coral (Before) row shows the original shape, notice the spike at zero and long right tail for caffeine and alcohol.

The blue (After) row shows the transformed shape, which is much more symmetric. This reduces the influence of extreme outliers on model coefficients.

## 5. Multicollinearity Analysis

Multicollinearity is checked once on the shared feature matrix and applies to all three pipelines. We use two methods:

1. **Pearson correlation heatmap** — identifies strongly correlated pairs visually
2. **Variance Inflation Factor (VIF)** — quantifies how much each feature's variance is inflated by its linear relationship with other features

VIF > 5 is a moderate concern. VIF > 10 is severe and warrants dropping or combining the affected feature.
"""

# Correlation heatmap
corr = df_proc[NUMERIC_CONTINUOUS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.4, ax=ax, annot_kws={'size': 7})
ax.set_title('Pearson Correlation Matrix for Numeric Features',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('08_correlation_heatmap.png', bbox_inches='tight')
plt.show()

# Flag pairs with |r| > 0.70
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i):
        r = corr.iloc[i, j]
        if abs(r) > 0.70:
            high_corr.append({'Feature A': corr.columns[i],
                               'Feature B': corr.columns[j],
                               'Pearson r': round(r, 3)})

if high_corr:
    hc_df = pd.DataFrame(high_corr).sort_values('Pearson r', key=abs, ascending=False)
    print("Highly correlated pairs (|r| > 0.70):")
    print(hc_df.to_string(index=False))
else:
    print("No pairs exceed |r| > 0.70.")

"""**Interpretation:**

Each cell shows the Pearson correlation between two features.

Dark blue = strong positive correlation; Dark red = strong negative correlation.

Any flagged pair shares so much variance that including both in a linear model inflates coefficient uncertainty and this is exactly what the VIF analysis below quantifies.
"""

vif_data = df_proc[NUMERIC_CONTINUOUS].copy()

# Subsample to 20,000 rows for speed
vif_sample = vif_data.sample(min(20000, len(vif_data)), random_state=42)

vif_results = pd.DataFrame({
    'Feature': vif_sample.columns,
    'VIF'    : [variance_inflation_factor(vif_sample.values, i)
                for i in range(vif_sample.shape[1])]
}).sort_values('VIF', ascending=False).reset_index(drop=True)

# Add a 'Concern Level' column based on VIF thresholds
def get_concern_level(vif_score):
    if vif_score >= 10:
        return 'High — consider dropping'
    elif 5 <= vif_score < 10:
        return 'Moderate — monitor'
    else:
        return 'No concern'
vif_results['Concern Level'] = vif_results['VIF'].apply(get_concern_level)

print("VIF Scores:")
print(vif_results.to_string(index=False))
print()
print("  VIF < 5     : No concern")
print("  5 <= VIF < 10: Moderate — monitor")
print("  VIF >= 10   : High — consider dropping")

"""**Interpretation:**

VIF measures how much a feature's variance is inflated by its correlations with all other features combined.

A VIF of 257 for `BMI` means its variance is 257× what it would be if independent. This makes linear model coefficients highly unstable. The `Concern Level` column directly categorizes each feature's multicollinearity based on common thresholds.

The solution is NOT to drop all these features; instead, PCA will be used for linear models (which eliminates collinearity entirely) and tree models are left unaffected (they split one feature at a time and have no coefficients to destabilise).
"""

# VIF bar chart
fig, ax = plt.subplots(figsize=(10, 8))
bar_colors = ['#F44336' if v >= 10 else '#FF9800' if v >= 5 else '#4CAF50'
              for v in vif_results['VIF']]

ax.barh(vif_results['Feature'], vif_results['VIF'],
        color=bar_colors, edgecolor='white')
ax.invert_yaxis()
ax.axvline(5,  color='orange', linestyle='--', linewidth=1.2, label='VIF = 5')
ax.axvline(10, color='red',    linestyle='--', linewidth=1.2, label='VIF = 10')
ax.set_title('Variance Inflation Factor by Feature', fontsize=13, fontweight='bold')
ax.set_xlabel('VIF Score')
ax.legend()
plt.tight_layout()
plt.savefig('09_vif.png', bbox_inches='tight')
plt.show()

"""**Interpretation:** Red bars (VIF ≥ 10) indicate severe multicollinearity. With most features in the red zone, dropping them is not an option — it would gut the predictive feature space. PCA resolves this for linear models by creating new orthogonal components where VIF = 1 by construction.

## 5.4 Multicollinearity Treatment Strategy

The VIF scores show that most numeric features exceed the threshold of 10, with some reaching extreme values (BMI: 257, steps: 150). Dropping all of them would gut the feature matrix. Instead, a complementary strategy is applied:

**Strategy used - PCA for linear model inputs.**

PCA replaces the correlated feature space with orthogonal components, making VIF = 1 for all components by construction. Tree-based models (Random Forest, AdaBoost, Gradient Boosting, XGBoost) are immune to multicollinearity because they split one feature at a time and do not estimate coefficients — VIF is irrelevant for them.

**What we do not do:** Drop 14 features from a 28-feature matrix. That would discard genuine predictive signal and leave the models underspecified.

### Multicollinearity Summary

The initial audit of the numeric features revealed extensive multicollinearity, with most VIF scores far exceeding the problematic threshold of 10. This was particularly evident for features within sleep architecture (e.g., sleep duration, quality, REM, deep sleep) and physiological indicators (heart rate, stress), which are fundamentally interconnected in the dataset.

#### **Strategy**

**PCA for Linear Models:**

For linear models (Logistic Regression, Linear regression), the feature matrix will be transformed using PCA, retaining 95% of the explained variance. Since principal components are orthogonal by design, this process effectively eliminates multicollinearity, making VIF exactly 1 for all components. This is a crucial step for preventing coefficient instability and improving the robustness of linear models.

#### **Why no strategies for Tree-based models**
**Tree-based Models are Robust:**

For tree-based ensemble models (Random Forest, AdaBoost, Gradient Boosting, XGBoost), multicollinearity is not a concern. These models make splitting decisions based on individual features and do not rely on coefficient estimation, thus rendering VIF irrelevant.

**Finally**:

This comprehensive approach ensures that all models can leverage the rich predictive information in the dataset without being hampered by multicollinearity-induced issues.

## 6a. Shared Feature Matrix

A single feature matrix `X` is built here and reused across all three pipelines. The three target columns are excluded from it. Each pipeline then defines its own target vector `y` and runs its own train/test split and evaluation.
"""

# Build feature matrix (excludes all three targets)
FEATURE_COLS = (
    [c for c in NUMERIC_CONTINUOUS if c in df_proc.columns] +
    BINARY_COLS +
    CATEGORICAL_COLS
)

X = df_proc[FEATURE_COLS].copy()

print(f"Shared feature matrix: {X.shape}")
print(f"Features: {FEATURE_COLS}")

X.head()

"""**Interpretation:** This single matrix is shared by all three pipelines. It includes 28 features (numeric, binary, and encoded categorical). All three target columns are excluded. Each pipeline defines its own target vector and runs its own train/test split from this same X."""

# Scale the full feature matrix once
# Each pipeline will do its own train/test split but use this scaled version.
# Note: in a strict pipeline, scaling should be done inside each split.
# Here we scale once for efficiency since the dataset has no leakage risk
# from the scaling itself (no target information flows through StandardScaler).

scaler  = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

print("Feature matrix scaled.")
print(f"Shape: {X_scaled.shape}")

X_scaled.head()

"""**Interpretation:** StandardScaler transforms every feature to mean = 0 and standard deviation = 1. This is required for logistic and linear regression where features on different scales bias coefficient magnitudes.

## 6b. Dimensionality Reduction (PCA)

Before building individual model pipelines, we apply Principal Component Analysis (PCA) to understand the intrinsic dimensionality of the feature space. PCA serves two purposes here. First, it provides a diagnostic: how many components are needed to explain 95% of the total variance tells us how redundant the current 28-feature space is. Second, the 2D PCA scatter plot gives us a visual check on whether the three sleep disorder risk classes are linearly separable in the reduced space, which directly informs our model expectations for Pipeline A.

PCA is applied on the scaled feature matrix `X_scaled`. We do not replace the original feature matrix for modeling — tree-based and boosting models work better with the original interpretable features. Instead, we use PCA for analysis and visualization, and offer an optional PCA-transformed version for the linear model pipelines where dimensionality reduction genuinely reduces variance.
"""

# Fit PCA on the full scaled feature matrix
from sklearn.decomposition import PCA

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

# Find number of components for 90%, 95%, 99% variance thresholds
for thresh in [0.90, 0.95, 0.99]:
    n = np.argmax(cumulative >= thresh) + 1
    print(f"  Components for {int(thresh*100)}% explained variance: {n}")

print(f"  Total features: {X_scaled.shape[1]}")

"""**Interpretation:** These numbers show how many components are needed to retain different levels of variance. Needing 24 components out of 28 for 95% variance confirms that the features are highly correlated — the last few components add very little unique information."""

# Scree plot + cumulative variance
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

top_n = min(25, len(explained))
components = range(1, top_n + 1)

# --------------------------------------------------
# Left: Scree plot only
# --------------------------------------------------
axes[0].bar(
    components,
    explained[:top_n],
    color='steelblue',
    edgecolor='white'
)
axes[0].set_title(
    'PCA: Explained Variance per Component (Top 25)',
    fontsize=12,
    fontweight='bold'
)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')

# --------------------------------------------------
# Right: Scree + cumulative variance
# --------------------------------------------------
axes[1].bar(
    components,
    explained[:top_n],
    color='steelblue',
    alpha=0.7,
    edgecolor='white',
    label='Individual Variance'
)

axes[1].step(
    components,
    cumulative[:top_n],
    where='mid',
    color='coral',
    linewidth=2.5,
    label='Cumulative Variance'
)

for thresh, ls in zip([0.90, 0.95, 0.99], [':', '--', '-']):
    axes[1].axhline(
        thresh,
        color='gray',
        linestyle=ls,
        linewidth=1.2,
        label=f'{int(thresh*100)}% variance'
    )

axes[1].set_title(
    'PCA: Scree Plot with Cumulative Variance',
    fontsize=12,
    fontweight='bold'
)
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].legend(fontsize=9)

plt.suptitle(
    'Principal Component Analysis — Feature Space Dimensionality',
    fontsize=13,
    fontweight='bold'
)

plt.tight_layout()
plt.savefig('PCA_scree.png', bbox_inches='tight')
plt.show()

"""### Interpretation: PCA Results

The Principal Component Analysis (PCA) provided valuable insights into the structure and dimensionality of our feature space:

-   **Scree Plot and Explained Variance:**

The scree plot indicates that a significant portion of the variance in the feature dataset can be explained by a smaller number of principal components. Specifically, 95% of the cumulative variance is captured by 24 components, which is less than the original 28 features.

This confirms that there is inherent redundancy within the feature set, consistent with our VIF analysis. This redundancy, often seen in datasets where multiple features measure related aspects (e.g., different metrics of sleep quality), suggests that linear models could benefit from dimensionality reduction to improve stability, while tree-based models can utilize the full feature set without issue.

Overall, PCA confirms the meaningful structure in the dataset, identifying coherent underlying themes (sleep health, stress, lifestyle) within the features. For linear models, a PCA-transformed feature set (capturing 95% variance) will be used to ensure robustness against multicollinearity, while tree-based models will continue to use the original scaled features due to their inherent robustness.
"""

# Build PCA-transformed input for linear baselines
# Tree and boosting models will use X_scaled directly.
# Logistic Regression (Pipelines A and C) and Linear Regression (Pipeline B)
# will use X_pca_95 to reduce coefficient instability from correlated features.

n_components_95 = int(np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95)) + 1
print(f"Using {n_components_95} components (95% variance) for linear model inputs.")

pca_95 = PCA(n_components=n_components_95, random_state=42)
X_pca_95 = pd.DataFrame(
    pca_95.fit_transform(X_scaled),
    columns=[f'PC{i+1}' for i in range(n_components_95)]
)

print(f"PCA-reduced feature matrix shape: {X_pca_95.shape}")
print(f"Cumulative variance explained: {pca_95.explained_variance_ratio_.sum()*100:.2f}%")

"""**Interpretation:** `X_pca_95` will be used exclusively by Logistic Regression and Linear Regression. It has fewer dimensions than the original 28 features while retaining 95% of total variance. Every component in this matrix is uncorrelated with every other component by construction — this directly solves the multicollinearity problem for linear models without discarding any predictive information.

## 6c. Additional Dimensionality Reduction Check with t-SNE

After the PCA analysis, we observed that PCA reduced the feature space only slightly. PCA needed 24 components to preserve 95% of the variance from the original 28 features, which means the dataset does not collapse cleanly into a much smaller linear space. This suggests that the structure of the dataset may not be strongly linear.

For this reason, we add t-SNE as a complementary dimensionality reduction method. t-SNE is not used to replace the modeling pipeline. It is used mainly for visualization because it can reveal local and non-linear patterns that PCA may not show clearly in two dimensions.

The main factors that affect t-SNE are:

- **Scaling:** t-SNE is distance-based, so it should be applied to the scaled feature matrix `X_scaled`.
- **Perplexity:** this controls the balance between local and broader neighborhood structure. We test several values instead of relying on one setting.
- **Initialization:** `init='pca'` gives the algorithm a stable starting point.
- **Random state:** t-SNE is stochastic, so `random_state=42` is used for reproducibility.
- **Sample size:** t-SNE can be slow on large datasets, so a reproducible sample is used when the dataset is large.
"""

# t-SNE setup
# This section is added after PCA without changing the existing PCA or modeling code.
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
import time

# t-SNE is computationally expensive, so we cap the visualization sample.
TSNE_SAMPLE_SIZE = 5000
sample_size = min(TSNE_SAMPLE_SIZE, len(X_scaled))

X_tsne_input = X_scaled.sample(n=sample_size, random_state=42)
y_tsne_risk = df_proc.loc[X_tsne_input.index, 'risk_encoded'].astype(int)

risk_label_map = {0: 'Healthy', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
risk_colors = {0: '#4CAF50', 1: '#FFC107', 2: '#FF9800', 3: '#F44336'}

print(f"t-SNE input sample: {X_tsne_input.shape[0]:,} rows x {X_tsne_input.shape[1]} features")
print("Risk class distribution in t-SNE sample:")
print(y_tsne_risk.value_counts().sort_index().rename(risk_label_map))

"""### t-SNE Perplexity Sensitivity

Because t-SNE results depend strongly on perplexity, we test multiple perplexity values. Smaller values focus more on very local neighborhoods, while larger values try to preserve a broader neighborhood structure. The goal is not to find a perfect classifier in two dimensions, but to check whether non-linear projection gives a clearer visual structure than PCA.

"""

# Fit t-SNE embeddings using several perplexity values
perplexity_candidates = [5, 15, 30, 50]
valid_perplexities = [p for p in perplexity_candidates if p < sample_size]

# If the sample is very small, keep at least one valid perplexity.
if not valid_perplexities:
    valid_perplexities = [max(2, min(5, sample_size - 1))]

tsne_embeddings = {}
tsne_kl = {}
tsne_times = {}

for perplexity in valid_perplexities:
    start_time = time.time()

    try:
        tsne_model = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='pca',
            max_iter=1000,
            random_state=42,
            verbose=0
        )
    except TypeError:
        # Compatibility with older scikit-learn versions where max_iter was named n_iter.
        tsne_model = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='pca',
            n_iter=1000,
            random_state=42,
            verbose=0
        )

    embedding = tsne_model.fit_transform(X_tsne_input)
    tsne_embeddings[perplexity] = embedding
    tsne_kl[perplexity] = tsne_model.kl_divergence_
    tsne_times[perplexity] = time.time() - start_time

    print(
        f"Perplexity {perplexity:>2}: "
        f"KL divergence = {tsne_kl[perplexity]:.4f}, "
        f"time = {tsne_times[perplexity]:.1f}s"
    )

# Visualize t-SNE embeddings for different perplexity values
n_plots = len(valid_perplexities)
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

if n_plots == 1:
    axes = [axes]

for ax, perplexity in zip(axes, valid_perplexities):
    embedding = tsne_embeddings[perplexity]

    for risk_value in sorted(y_tsne_risk.unique()):
        mask = y_tsne_risk.values == risk_value
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.55,
            color=risk_colors.get(risk_value),
            label=risk_label_map.get(risk_value, str(risk_value))
        )

    ax.set_title(f't-SNE Projection Perplexity = {perplexity}', fontsize=11, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, title='Sleep Disorder Risk', loc='upper center', ncol=4)
plt.suptitle('t-SNE Visualization of the Scaled Feature Space', fontsize=14, fontweight='bold', y=1.08)
plt.tight_layout()
plt.savefig('PCA_vs_tSNE_perplexity.png', bbox_inches='tight')
plt.show()

"""### PCA vs t-SNE Comparison

To compare the two dimensionality reduction methods fairly, we project the same sampled data into two dimensions using PCA and t-SNE. PCA is linear and aims to preserve global variance. t-SNE is non-linear and aims to preserve local neighborhoods. Because their objectives are different, the comparison focuses on visual separability, neighborhood preservation, and whether the 2D representation helps us understand the dataset structure.

"""

# Compare 2D PCA with the best t-SNE embedding
# Best t-SNE is selected using the lowest KL divergence among tested perplexities.
pca_2d = PCA(n_components=2, random_state=42).fit_transform(X_tsne_input)
best_perplexity = min(tsne_kl, key=tsne_kl.get)
tsne_best_2d = tsne_embeddings[best_perplexity]

# Trustworthiness measures how well local neighborhoods are preserved after reduction.
n_neighbors = min(15, sample_size - 1)
pca_trust = trustworthiness(X_tsne_input, pca_2d, n_neighbors=n_neighbors)
tsne_trust = trustworthiness(X_tsne_input, tsne_best_2d, n_neighbors=n_neighbors)

# Silhouette score checks how separated the known risk classes appear in the 2D space.
# This is only a visual diagnostic, not a supervised model score.
pca_silhouette = silhouette_score(pca_2d, y_tsne_risk) if y_tsne_risk.nunique() > 1 else np.nan
tsne_silhouette = silhouette_score(tsne_best_2d, y_tsne_risk) if y_tsne_risk.nunique() > 1 else np.nan

comparison_df = pd.DataFrame({
    'Method': ['PCA 2D', f't-SNE 2D (perplexity={best_perplexity})'],
    'Type': ['Linear', 'Non-linear'],
    'Main Objective': ['Preserve global variance', 'Preserve local neighborhoods'],
    'Trustworthiness': [pca_trust, tsne_trust],
    'Silhouette by Risk Class': [pca_silhouette, tsne_silhouette],
    'KL Divergence': [np.nan, tsne_kl[best_perplexity]]
})

comparison_df

# Side-by-side PCA and t-SNE visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

plots = [
    (axes[0], pca_2d, 'PCA 2D Projection'),
    (axes[1], tsne_best_2d, f't-SNE 2D Projection Best Perplexity = {best_perplexity}')
]

for ax, embedding, title in plots:
    for risk_value in sorted(y_tsne_risk.unique()):
        mask = y_tsne_risk.values == risk_value
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.55,
            color=risk_colors.get(risk_value),
            label=risk_label_map.get(risk_value, str(risk_value))
        )

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

axes[1].legend(title='Sleep Disorder Risk', loc='best')
plt.suptitle('PCA vs t-SNE: Two-Dimensional Feature Space Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('PCA_vs_tSNE_comparison.png', bbox_inches='tight')
plt.show()

"""### **Interpretation: t-SNE Results and Comparison with PCA**

The t-SNE analysis was added because the PCA results suggested that the dataset does not reduce strongly into a small number of linear components. PCA required 24 components to retain 95% of the information from 28 original features, which means PCA only produced a small reduction in dimensionality. This supports the observation that the dataset may not have a strongly linear structure.

Compared with PCA, t-SNE gives a more flexible two-dimensional view of the dataset because it focuses on preserving local similarities between observations. If the t-SNE plots show more visible grouping or neighborhood structure than the PCA plot, it suggests that the dataset contains non-linear relationships that PCA cannot fully capture in two dimensions.

However, we are mainly using t-SNE as a visualization tool, not a replacement for the modeling pipeline. The axes do not have direct feature meanings, and the distances between far-apart clusters should not be overinterpreted. Hence, t-SNE is being useful as an additional diagnostic showing that non-linear structure may exist in the our dataset, while PCA remains useful for reducing multicollinearity in linear models.

**Final comparison:** PCA is better for creating stable, interpretable, variance-based components for linear models. t-SNE is better for visual exploration when PCA does not reveal clear structure in two dimensions. Therefore, we keep the existing PCA-based model inputs unchanged and add t-SNE only as a complementary visualization and dimensionality reduction analysis.

### **Note on Overfitting and Regularization Strategy**


Each pipeline checks the train-test gap for its models before finalizing them. The diagnosis thresholds used throughout are:

- **Gap > 0.10 (for accuracy or R²):** Overfitting. The model is memorizing training data. Correction: add depth constraints (`max_depth`, `min_samples_leaf`) for tree models, or increase regularization (reduce `C` for logistic/linear).
- **Test and train accuracies < 0.65 (or R² < 0.50):** Underfitting. The model lacks capacity. Correction: remove constraints, increase estimators, or switch to a more expressive model.
- **Gap ≤ 0.10 and acceptable test metric:** Good generalization.

For Logistic Regression and Linear Regression across all pipelines, PCA-reduced inputs (95% variance retained) are used from the start to prevent coefficient instability caused by multicollinearity, which is a form of variance inflation that mimics overfitting.

---
## Pipeline A: `sleep_disorder_risk` — Multiclass Classification

**Problem:** Predict sleep disorder risk level across four ordered classes.

**Classes:** Healthy (54.2%), Mild (33.5%), Moderate (8.3%), Severe (4.1%)

**Imbalance:** 13:1 ratio between Healthy and Severe. All three models use `class_weight='balanced'`.

**Models:**
- Logistic Regression (multinomial baseline)
- Random Forest (bagging ensemble)
- AdaBoost (sequential boosting with `algorithm='SAMME'`)

**Primary metric:** Weighted F1-score (accounts for class imbalance)
"""

# 1 Target and split
y_A = df_proc['risk_encoded'].copy()

XA_train, XA_test, yA_train, yA_test = train_test_split(
    X_scaled, y_A,
    test_size=0.20,
    random_state=42,
    stratify=y_A
)

print(f"Training: {XA_train.shape}  |  Test: {XA_test.shape}")
print()
print("Training class distribution:")
print(yA_train.value_counts().sort_index()
      .rename({0:'Healthy',1:'Mild',2:'Moderate',3:'Severe'}))

"""**Interpretation:** Stratified splitting ensures each risk class appears in the same proportion in both the training and test sets. With only 4.1% Severe cases, this prevents the test set from accidentally containing too few Severe examples to evaluate the model reliably on the most critical class."""

#  Mutual information: feature selection
mi_A = mutual_info_classif(XA_train, yA_train, random_state=42)
mi_A_s = pd.Series(mi_A, index=X_scaled.columns).sort_values(ascending=False)
# mi_A_s = pd.Series(mi_A, index=X_pca_95.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
mi_A_s.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.invert_yaxis()
ax.set_title('Feature MI Scores vs sleep_disorder_risk',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Mutual Information Score')
plt.tight_layout()
plt.savefig('A_mutual_information.png', bbox_inches='tight')
plt.show()

# Drop near-zero MI features for this pipeline
zero_mi_A = mi_A_s[mi_A_s < 0.001].index.tolist()
if zero_mi_A:
    print(f"Dropping {len(zero_mi_A)} near-zero MI features: {zero_mi_A}")
    XA_train = XA_train.drop(columns=zero_mi_A)
    XA_test  = XA_test.drop(columns=zero_mi_A)
else:
    print("All features retained.")
print(f"Feature matrix for Pipeline A: {XA_train.shape}")

"""### Why Mutual Information is used here and why it is different from PCA

Mutual Information (MI) and PCA serve completely different purposes and are not alternatives to each other.

**Mutual Information** measures how much each individual feature tells us about the target variable. A score of 0 means the feature and the target are statistically independent — knowing that feature adds nothing to predicting the target. A higher score means more predictive relevance. MI is used here as a **filter**: any feature that carries essentially zero signal for this specific target is dropped before training, because it only adds noise.

**PCA** is applied separately, only to the input of linear models (Logistic Regression and Linear Regression), to resolve multicollinearity. PCA does not score features against the target — it restructures the feature space to remove correlation between features.

**Why tree models use MI-filtered original features but NOT PCA:**

Random Forest, AdaBoost, Gradient Boosting, and XGBoost are immune to multicollinearity. They split on one feature at a time and never estimate coefficients, so correlated features simply share importance credit without destabilising the model. Applying PCA to tree models would replace interpretable original features with abstract components, losing the ability to explain which real variables matter.

**The full sequence in each pipeline is:**
1. MI screening: drops features with near-zero signal for this specific target (applied to all models)
2. PCA transformation: resolves multicollinearity (applied to linear models only)
3. MI-filtered scaled features: used directly by tree-based models

These two steps are complementary. MI removes noise; PCA removes collinearity among the useful features.
"""

# Evaluation helper
results_A  = {}
cv_A       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
MODELS_A   = ['Logistic Regression', 'Random Forest', 'AdaBoost']

def eval_classif(name, model, X_tr, y_tr, X_te, y_te, results_dict, cv):
    model.fit(X_tr, y_tr)
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    train_acc = accuracy_score(y_tr, y_pred_tr)
    test_acc  = accuracy_score(y_te, y_pred_te)
    train_f1  = f1_score(y_tr, y_pred_tr, average='weighted')
    test_f1   = f1_score(y_te, y_pred_te, average='weighted')
    gap       = train_acc - test_acc

    cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv,
                             scoring='f1_weighted', n_jobs=-1)

    diagnosis = ("Overfitting" if gap > 0.10
                 else "Underfitting" if test_acc < 0.65
                 else "Good generalization")

    results_dict[name] = {
        'model': model, 'y_pred': y_pred_te,
        'train_acc': train_acc, 'test_acc': test_acc,
        'train_f1': train_f1,  'test_f1': test_f1,
        'gap': gap, 'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(), 'diagnosis': diagnosis
    }

    print(f"{'='*60} {name} {'='*60}")
    print(f"  Train Accuracy     : {train_acc:.4f}")
    print(f"  Test Accuracy      : {test_acc:.4f}")
    print(f"  Train-Test Gap     : {gap:.4f}  -> {diagnosis}")
    print(f"  Test F1 (weighted) : {test_f1:.4f}")
    print(f"  CV F1 mean +/- std : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

#  Model 1: Logistic Regression (with PCA input)
# Logistic Regression benefits from PCA-reduced input because multicollinear
# features inflate coefficient variance without improving decision boundaries.
# We split the PCA-reduced matrix using the same stratified split as X_scaled.

XA_pca_train, XA_pca_test, _, _ = train_test_split(
    X_pca_95, y_A,
    test_size=0.20, random_state=42, stratify=y_A
)

lr_A = LogisticRegression(
    multi_class='multinomial', solver='lbfgs',
    max_iter=2000, C=1.0,
    class_weight='balanced', random_state=42
)

eval_classif('Logistic Regression', lr_A,
             XA_pca_train, yA_train, XA_pca_test, yA_test, results_A, cv_A)

# Apply PCA to the MI-reduced feature set (XA_train, XA_test)
pca_mi = PCA(n_components=0.95, random_state=42)
pca_mi.fit(XA_train)

XA_pca_mi_train = pca_mi.transform(XA_train)
XB_pca_mi_test = pca_mi.transform(XA_test)

eval_classif('Logistic Regression (PCA on MI-Reduced Features)', lr_A,
          XA_pca_mi_train, yA_train, XB_pca_mi_test, yA_test, results_A, cv_A)

"""**Interpretation:** Logistic Regression is the linear baseline. It uses PCA-reduced input to avoid unstable coefficients from multicollinearity. The metrics above show train accuracy, test accuracy, and the gap between them. A gap close to zero means the model generalises well."""

# A.5 Model 2: Random Forest
# Initial check: unconstrained Random Forest (max_depth=None) on 100k records
# typically overfits on training data due to the model memorizing deep tree paths.
# We first train without constraints to detect the gap, then apply regularization
# (max_depth=20, min_samples_leaf=10) to correct any overfitting found.

rf_A_unconstrained = RandomForestClassifier(
    n_estimators=100, max_depth=None, min_samples_split=5,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf_A_unconstrained.fit(XA_train, yA_train)
unconstrained_train_acc = accuracy_score(yA_train, rf_A_unconstrained.predict(XA_train))
unconstrained_test_acc  = accuracy_score(yA_test,  rf_A_unconstrained.predict(XA_test))
gap_unconstrained = unconstrained_train_acc - unconstrained_test_acc

print(f"Unconstrained RF: Train={unconstrained_train_acc:.4f}, Test={unconstrained_test_acc:.4f}, Gap={gap_unconstrained:.4f}")

if gap_unconstrained > 0.10:
    print("Overfitting detected (gap > 0.10). Applying regularization: max_depth=20, min_samples_leaf=10.")
    rf_A = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_split=5,
        min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
else:
    print(f"Gap = {gap_unconstrained:.4f} — within acceptable range. Keeping unconstrained model.")
    rf_A = rf_A_unconstrained

eval_classif('Random Forest', rf_A,
             XA_train, yA_train, XA_test, yA_test, results_A, cv_A)

"""**Interpretation:** The unconstrained RF result printed first reveals the raw gap. If the gap exceeds 0.10, the constrained version (max_depth=20, min_samples_leaf=10) limits tree growth, preventing individual trees from memorising training examples. The final metrics confirm whether the regularisation brought the gap within range."""

# Model 3: AdaBoost
# algorithm='SAMME' required for multiclass (>2 classes).
base_dt_A = DecisionTreeClassifier(max_depth=2, random_state=42)
ada_A = AdaBoostClassifier(
    estimator=base_dt_A, n_estimators=100,
    learning_rate=0.5, algorithm='SAMME', random_state=42
)
eval_classif('AdaBoost', ada_A,
             XA_train, yA_train, XA_test, yA_test, results_A, cv_A)

"""**Interpretation:** The unconstrained RF result printed first reveals the raw gap. If the gap exceeds 0.10, the constrained version (max_depth=20, min_samples_leaf=10) limits tree growth, preventing individual trees from memorising training examples. The final metrics confirm whether the regularisation brought the gap within range."""

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, name in zip(axes, MODELS_A):
    cm = confusion_matrix(yA_test, results_A[name]['y_pred'])
    ConfusionMatrixDisplay(cm, display_labels=RISK_NAMES).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=12, fontweight='bold')

plt.suptitle('Pipeline A — Confusion Matrices: Sleep Disorder Risk',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('A_confusion_matrices.png', bbox_inches='tight')
plt.show()

"""### **Interpretation of the Confusion Matrices**

In a confusion matrix, **rows represent the true class** and **columns represent the predicted class**. Values along the **diagonal** correspond to correct predictions, while off-diagonal values represent misclassifications.

For **Logistic Regression**, the diagonal values are relatively weaker, particularly for the **Moderate** and **Severe** classes. Many Moderate cases are predicted as Mild, and a substantial number of Severe cases are classified as Moderate. This indicates difficulty distinguishing higher-risk patients, which explains the model's lower recall and F1-scores for these classes.

The **AdaBoost** model improves the classification of Moderate cases compared to Logistic Regression, but it still struggles with Severe cases. A large number of Severe patients are predicted as Moderate, resulting in a lower recall for the highest-risk group. While overall performance is better than Logistic Regression, important high-risk cases are still frequently missed.

The **Random Forest** confusion matrix shows the strongest concentration of predictions along the diagonal, indicating the highest number of correct classifications across all risk levels. Although some confusion remains between neighboring classes such as Mild and Moderate or Moderate and Severe, the number of errors is substantially lower than in the other models. This aligns with its superior accuracy, precision, recall, and F1-score.

Overall, the confusion matrices confirm that **Random Forest provides the best balance between correctly identifying all risk categories and minimizing critical misclassifications of Moderate and Severe patients**, making it the most reliable model for this task.

"""

# Classification reports
for name in MODELS_A:
    print(f" {'='*60} {name} {'='*60}")
    print(classification_report(yA_test, results_A[name]['y_pred'], target_names=RISK_NAMES))

"""### **Interpretation of Classification Metrics**

* **Precision** measures how often the model is correct when it predicts a class. For example, a Severe precision of 0.85 means that 85% of the cases predicted as Severe were actually Severe.
* **Recall** measures how many of the actual cases in a class the model successfully identifies. A Severe recall of 0.70 means the model correctly detected 70% of all Severe cases, missing the remaining 30%.
* **F1-Score** combines precision and recall into a single metric. A high F1-score indicates that the model achieves a good balance between making correct predictions and finding most cases of a class.
* **Support** is simply the number of true instances belonging to each class in the test set.
* **Accuracy** represents the overall percentage of correctly classified samples.
* **Macro Average** gives equal importance to every class, making it useful when evaluating performance across minority and majority classes.
* **Weighted Average** accounts for class frequencies, so larger classes have a greater influence on the final score.

### **Model Comparison**

The **Logistic Regression** model achieved an accuracy of 73%, but struggled to identify Moderate and Severe cases, resulting in relatively low F1-scores for these important risk categories.

The **AdaBoost** model improved performance substantially, reaching 83% accuracy and stronger results for the Mild and Moderate classes. However, its recall for Severe cases (0.45) indicates that more than half of the highest-risk individuals were still missed.

The **Random Forest** model delivered the best overall performance, achieving 91% accuracy and the highest weighted F1-score (0.91). It also provided the strongest balance between precision and recall across all classes, including Severe cases (F1 = 0.77). Therefore, Random Forest is the most reliable model for this classification task and is selected as the best-performing model.

"""

# ── A.9 Comparison table ──────────────────────────────────────────────────────
comp_A = pd.DataFrame([{
    'Model': n,
    'Train Acc': round(results_A[n]['train_acc'], 4),
    'Test Acc' : round(results_A[n]['test_acc'],  4),
    'Gap'      : round(results_A[n]['gap'],        4),
    'Test F1 (W)': round(results_A[n]['test_f1'], 4),
    'CV F1 Mean' : round(results_A[n]['cv_f1_mean'], 4),
    'Diagnosis'  : results_A[n]['diagnosis']
} for n in MODELS_A]).sort_values('Test F1 (W)', ascending=False).reset_index(drop=True)

print("Pipeline A — Model Comparison:")
print(comp_A.to_string(index=False))

"""**Interpretation:** The Gap column is the overfitting diagnostic — above 0.10 flags a model that generalises poorly. CV F1 Mean (5-fold cross-validation) is more reliable than a single test split because it averages across five different data partitions.

We sorted by F1 (weighted) to identify the overall best performer.
"""

# A. Feature importance (Random Forest)
fi_A = pd.DataFrame({
    'Feature'   : XA_train.columns.tolist(),
    'Importance': results_A['Random Forest']['model'].feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

top = min(15, len(fi_A))
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(fi_A['Feature'].head(top), fi_A['Importance'].head(top),
        color='steelblue', edgecolor='white')
ax.invert_yaxis()
ax.set_title(f'Top {top} Feature Importances — Random Forest (sleep_disorder_risk)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig('A_feature_importance.png', bbox_inches='tight')
plt.show()

# AdaBoost learning curve
staged_err_A = [1-s for s in results_A['AdaBoost']['model']
                .staged_score(XA_test.values, yA_test)]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(staged_err_A, color='coral', linewidth=1.5)
ax.set_title('AdaBoost Staged Error — sleep_disorder_risk',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Estimators')
ax.set_ylabel('Test Error')
plt.tight_layout()
plt.savefig('A_adaboost_curve.png', bbox_inches='tight')
plt.show()

"""**Feature Importance:** The bar chart shows which features most reduced prediction uncertainty across all tree splits. Features at the top are the strongest predictors of sleep disorder risk — expect sleep quality score and stress score to dominate.

**AdaBoost Staged Error:** The curve shows test error after each estimator is added. A steeply falling curve that then flattens is ideal. If the error rises at the end, AdaBoost is starting to overfit by over-focusing on hard examples.

### Pipeline A Summary

This pipeline addresses a challenging multiclass classification problem with a significant class imbalance in `sleep_disorder_risk` (13:1 ratio between 'Healthy' and 'Severe'). All models utilize `class_weight='balanced'` to mitigate this imbalance, and evaluation is conducted on a held-out 20% test set using weighted F1-score as the primary metric.

-   **Logistic Regression (PCA input):** This model serves as a linear baseline. By training on PCA-reduced features (retaining 95% variance), it benefits from stabilized coefficient estimates due to the removal of multicollinearity. Its performance, while lower than ensemble models, provides a crucial benchmark for the inherent linear separability of the problem. The classification report highlights its struggle with minority classes, particularly 'Moderate' and 'Severe', which is expected for a linear model on a non-linearly separable problem.

-   **Random Forest:** An overfitting check is integral to this model. An unconstrained Random Forest is initially trained, and if the train-test accuracy gap exceeds 0.10, regularization (specifically, `max_depth=20` and `min_samples_leaf=10`) is applied. This strategy ensures the model generalizes well by preventing individual trees from memorizing the training data. The model demonstrates strong performance overall, significantly outperforming logistic regression.

-   **AdaBoost:** This boosting model uses shallow decision trees as base estimators. The staged error curve is a diagnostic tool, showing how test error evolves with the number of estimators. A plateau or increase in error at higher estimator counts would signal boosting overfitting. AdaBoost provides competitive performance, demonstrating its effectiveness in combining weak learners.

**Key Metric:** While overall weighted F1-score is important, the most critical aspect to monitor is the **recall for the 'Severe' class**. Given its extremely low representation (4.1%), even models with good overall metrics can fail to identify most severe cases. The classification report for 'Severe' provided the most direct assessment of the model's ability to capture this critical minority class.

---
## Pipeline B: `cognitive_performance_score` - Regression

**Problem:** Predict next-day cognitive performance as a continuous score (range approximately 0–100, mean 59.2).

**Evaluation metrics:** RMSE, MAE, R². Unlike classification, accuracy has no meaning here.
- **RMSE** (Root Mean Squared Error) — penalises large errors more than MAE
- **MAE** (Mean Absolute Error) — average absolute prediction error in score units
- **R²** — proportion of variance in cognitive performance explained by the model (1.0 is perfect)

**Models:**
- Linear Regression (baseline)
- Random Forest Regressor (bagging ensemble)
- Gradient Boosting Regressor (sequential boosting)
"""

# B. Target and split
y_B = df_proc['cognitive_performance_score'].copy()

print(f"Target stats:")
print(f"  Mean   : {y_B.mean():.2f}")
print(f"  Std    : {y_B.std():.2f}")
print(f"  Min    : {y_B.min():.2f}")
print(f"  Max    : {y_B.max():.2f}")
print(f"  Skew   : {y_B.skew():.3f}")

XB_train, XB_test, yB_train, yB_test = train_test_split(
    X_scaled, y_B, test_size=0.20, random_state=42
)

print(f"Training: {XB_train.shape}  |  Test: {XB_test.shape}")

"""**Interpretation:** The cognitive score is continuous, so no stratification is needed for the split. The mean (~59) and standard deviation printed above define the prediction challenge — a model that always predicts the mean would have R² = 0 and RMSE equal to the standard deviation, which is the baseline to beat."""

# ── B.2 Mutual information: feature selection for regression ───────────────────
mi_B = mutual_info_regression(XB_train, yB_train, random_state=42)
mi_B_s = pd.Series(mi_B, index=X_scaled.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
mi_B_s.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.invert_yaxis()
ax.set_title('Feature MI Scores vs cognitive_performance_score',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Mutual Information Score')
plt.tight_layout()
plt.savefig('B_mutual_information.png', bbox_inches='tight')
plt.show()

zero_mi_B = mi_B_s[mi_B_s < 0.001].index.tolist()
if zero_mi_B:
    print(f"Dropping {len(zero_mi_B)} near-zero MI: {zero_mi_B}")
    XB_train = XB_train.drop(columns=zero_mi_B)
    XB_test  = XB_test.drop(columns=zero_mi_B)
else:
    print("All features retained.")
print(f"Feature matrix for Pipeline B: {XB_train.shape}")

"""**Interpretation:** `mutual_info_regression` is the continuous-target version of MI scoring. Features with near-zero MI tell us nothing about cognitive performance and are dropped. This MI filtering applies to all models in this pipeline i.e both tree-based and linear.

Linear Regression additionally receives PCA-transformed input after this step to remove collinearity between the retained features. The MI chart itself shows which features are most predictive with sleep quality, sleep duration and stress ranking highest.
"""

# B. Regression evaluation helper
results_B = {}
cv_B      = KFold(n_splits=5, shuffle=True, random_state=42)
MODELS_B  = ['Linear Regression (baseline)', 'Linear Regression (PCA on Full Features)', 'Random Forest', 'Gradient Boosting']

def eval_regr(name, model, X_tr, y_tr, X_te, y_te, results_dict, cv):
    model.fit(X_tr, y_tr)
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    train_r2  = r2_score(y_tr, y_pred_tr)
    test_r2   = r2_score(y_te, y_pred_te)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
    test_mae  = mean_absolute_error(y_te, y_pred_te)
    gap       = train_r2 - test_r2

    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=cv,
                             scoring='r2', n_jobs=-1)

    diagnosis = ("Overfitting" if gap > 0.10
                 else "Underfitting" if test_r2 < 0.50
                 else "Good generalization")

    results_dict[name] = {
        'model': model, 'y_pred': y_pred_te,
        'train_r2': train_r2, 'test_r2': test_r2,
        'test_rmse': test_rmse, 'test_mae': test_mae,
        'gap': gap, 'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(), 'diagnosis': diagnosis
    }

    print(f"{'='*60} {name} {'='*60}")
    print(f"  Train R²          : {train_r2:.4f}")
    print(f"  Test R²           : {test_r2:.4f}")
    print(f"  Train-Test Gap    : {gap:.4f}  -> {diagnosis}")
    print(f"  Test RMSE         : {test_rmse:.4f}")
    print(f"  Test MAE          : {test_mae:.4f}")
    print(f"  CV R² mean +/- std: {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")

#  B. Model 1: Linear Regression (linear baseline + with PCA)
#  We also use the PCA-reduced matrix to further reduce dimensionality for the linear baseline.

XB_pca_full_train, XB_pca_full_test, _, _ = train_test_split(
    X_pca_95, y_B, test_size=0.20, random_state=42
)

lr_baseline = LinearRegression()
eval_regr('Linear Regression (baseline)', lr_baseline,
          XB_train, yB_train, XB_test, yB_test, results_B, cv_B)

# Linear Regression on PCA-transformed full features
eval_regr('Linear Regression (PCA on Full Features)', lr_baseline,
          XB_pca_full_train, yB_train, XB_pca_full_test, yB_test, results_B, cv_B)

# Apply PCA to the MI-reduced feature set (XB_train, XB_test)
pca_mi = PCA(n_components=0.95, random_state=42)
pca_mi.fit(XB_train)

XB_pca_mi_train = pca_mi.transform(XB_train)
XB_pca_mi_test = pca_mi.transform(XB_test)

eval_regr('Linear Regression (PCA on MI-Reduced Features)', lr_baseline,
          XB_pca_mi_train, yB_train, XB_pca_mi_test, yB_test, results_B, cv_B)

"""**Interpretation:**

The original Linear Regression model (baseline) performed the best, achieving a Test R² of 0.8809, and the lowest RMSE and MAE.

The PCA-transformed models, both on full features and on MI-reduced features, showed slightly lower performance in terms of R² (around 0.85) and slightly higher RMSE/MAE.

This indicates that, for this specific dataset and problem, applying PCA after MI-reduction did not improve the predictive power of the linear model compared to just using the MI-reduced features directly, although all models demonstrate good generalization without significant overfitting.

The PCA approach is primarily for mitigating multicollinearity and stabilizing coefficients, which it achieves, even if it results in a minor decrease in predictive accuracy in this instance.
"""

# B. Model 2: Random Forest Regressor (with overfitting check)
rf_B_unconstrained = RandomForestRegressor(
    n_estimators=100, max_depth=None,
    min_samples_split=5, random_state=42, n_jobs=-1
)
rf_B_unconstrained.fit(XB_train, yB_train)
u_train_r2 = r2_score(yB_train, rf_B_unconstrained.predict(XB_train))
u_test_r2  = r2_score(yB_test,  rf_B_unconstrained.predict(XB_test))
gap_u = u_train_r2 - u_test_r2

print(f"Unconstrained RF: Train R²={u_train_r2:.4f}, Test R²={u_test_r2:.4f}, Gap={gap_u:.4f}")

if gap_u > 0.10:
    print("Overfitting detected (gap > 0.10). Applying max_depth=20, min_samples_leaf=10.")
    rf_B = RandomForestRegressor(
        n_estimators=100, max_depth=20,
        min_samples_split=5, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
else:
    print(f"Gap = {gap_u:.4f} — acceptable. Keeping unconstrained model.")
    rf_B = rf_B_unconstrained

eval_regr('Random Forest', rf_B,
          XB_train, yB_train, XB_test, yB_test, results_B, cv_B)

"""**Interpretation:** The unconstrained RF result is printed first to reveal the raw train-test R² gap. With 80,000 training rows, an unconstrained forest can fit extremely fine splits that do not generalise. If the gap exceeds 0.10, depth constraints force each split to represent genuine population patterns rather than individual data points."""

# ── B.6 Model 3: Gradient Boosting Regressor ──────────────────────────────────
# Gradient Boosting fits residuals sequentially, making it well-suited for
# regression problems where the target has structured error patterns.
# subsample < 1.0 adds stochasticity, reducing overfitting.

gb_B = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.05,
    max_depth=4, subsample=0.8,
    random_state=42
)
eval_regr('Gradient Boosting', gb_B,
          XB_train, yB_train, XB_test, yB_test, results_B, cv_B)

# B. Predicted vs actual plots
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for ax, name in zip(axes, MODELS_B):
    y_pred = results_B[name]['y_pred']
    ax.scatter(yB_test, y_pred, alpha=0.3, s=5, color='steelblue')
    lims = [min(yB_test.min(), y_pred.min()),
            max(yB_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_title(f'{name} R²={results_B[name]["test_r2"]:.3f}  '
                 f'RMSE={results_B[name]["test_rmse"]:.2f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.legend(fontsize=8)

plt.suptitle('Pipeline B — Predicted vs Actual: cognitive_performance_score',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('B_predicted_vs_actual.png', bbox_inches='tight')
plt.show()

"""### **Interpretation of Predicted vs Actual Plots**

In these scatter plots, each point represents a test sample. The **x-axis shows the actual cognitive performance score**, while the **y-axis shows the model's predicted score**. The red dashed line represents **perfect predictions**, where the predicted value exactly matches the actual value.

A good regression model produces points that cluster closely around the diagonal line. The closer the points are to this line, the smaller the prediction errors. This is reflected by a **higher R² value**, which measures how much of the variation in the target variable is explained by the model, and a **lower RMSE**, which measures the average prediction error.

The **Linear Regression (baseline)** model achieves a strong performance with an R² of 0.881 and an RMSE of 7.65, indicating that it captures most of the relationship between the features and cognitive performance. However, some dispersion around the diagonal remains, especially at lower and higher score ranges.

The **PCA-based Linear Regression** performs slightly worse, with a lower R² (0.853) and higher RMSE (8.50). This suggests that reducing the feature space through PCA removed some information that was useful for prediction.

The **Gradient Boosting** model delivers the best results, achieving the highest R² (0.92) and the lowest RMSE (6.27). Its predictions form the tightest cluster around the diagonal line, indicating more accurate and consistent estimates across the full range of cognitive performance scores.

Overall, the plots confirm that **Gradient Boosting is the best-performing regression model**, providing the most accurate predictions and the lowest prediction error among the three approaches.
"""

#  B. Residual plots
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for ax, name in zip(axes, MODELS_B):
    residuals = yB_test.values - results_B[name]['y_pred']
    ax.scatter(results_B[name]['y_pred'], residuals,
               alpha=0.3, s=5, color='coral')
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
    ax.set_title(f'{name} — Residuals', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual')

plt.suptitle('Pipeline B — Residual Plots', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('B_residuals.png', bbox_inches='tight')
plt.show()

"""### **Interpretation of the Residual Plots**

Residuals represent the difference between the actual and predicted values (**Residual = Actual − Predicted**). A well-performing regression model should produce residuals that are randomly scattered around the zero line with no obvious patterns. This indicates that the model is making unbiased predictions across the full range of scores.

The **Linear Regression (baseline)** model shows a relatively wide spread of residuals, particularly at higher predicted values. The increasing variability suggests that prediction errors become larger for some observations, indicating that the model does not fully capture all relationships present in the data.

The **PCA-based Linear Regression** exhibits a similar pattern, with an even wider spread of residuals and more extreme errors. This is consistent with its lower R² and higher RMSE, suggesting that the dimensionality reduction removed some predictive information that was useful for the regression task.

The **Gradient Boosting** model displays the most desirable residual pattern. The residuals are more tightly concentrated around the zero line and show a more uniform spread across the prediction range. There is less evidence of systematic bias and fewer large prediction errors compared to the linear models.

Overall, the residual plots reinforce the earlier findings from the R² and RMSE metrics.

**Gradient Boosting produces the most accurate and reliable predictions, with smaller and more evenly distributed errors, making it the best-performing regression model for predicting cognitive performance scores.**

"""

# Comparison table
comp_B = pd.DataFrame([{
    'Model'     : n,
    'Train R²'  : round(results_B[n]['train_r2'],  4),
    'Test R²'   : round(results_B[n]['test_r2'],   4),
    'Gap'       : round(results_B[n]['gap'],        4),
    'Test RMSE' : round(results_B[n]['test_rmse'],  4),
    'Test MAE'  : round(results_B[n]['test_mae'],   4),
    'CV R² Mean': round(results_B[n]['cv_r2_mean'], 4),
    'Diagnosis' : results_B[n]['diagnosis']
} for n in MODELS_B]).sort_values('Test R²', ascending=False).reset_index(drop=True)

print("Pipeline B — Model Comparison:")
print(comp_B.to_string(index=False))

# B.10 Feature importance (Gradient Boosting)
fi_B = pd.DataFrame({
    'Feature'   : XB_train.columns.tolist(),
    'Importance': results_B['Gradient Boosting']['model'].feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

top = min(15, len(fi_B))
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(fi_B['Feature'].head(top), fi_B['Importance'].head(top),
        color='steelblue', edgecolor='white')
ax.invert_yaxis()
ax.set_title(f'Top {top} Feature Importances — Gradient Boosting (cognitive_performance_score)', fontsize=12, fontweight='bold')
ax.set_xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig('B_feature_importance.png', bbox_inches='tight')
plt.show()

"""**Interpretation:**

The feature importance chart shows how much each variable contributed to reducing prediction error in the final tree-based model. Features ranked higher have a greater influence on predicting cognitive performance.

The dominance of variables such as sleep quality score, sleep duration, and other sleep-related measures indicates that sleep health is a major driver of cognitive performance in this dataset.

The ranking is also broadly consistent with the earlier Mutual Information analysis, providing additional evidence that these features contain the most predictive information.

### **Pipeline B Summary**

This pipeline focuses on predicting `cognitive_performance_score`, a continuous target, using regression models. Key evaluation metrics include RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R² (coefficient of determination).

* **Linear Regression (Baseline):** A standard Linear Regression model is trained directly on the selected features. It achieved strong predictive performance, demonstrating that the relationship between the predictors and cognitive performance is largely linear.

* **Linear Regression (PCA):** A second Linear Regression model is trained on PCA-transformed features derived from the MI-selected variables, retaining 95% of the variance. While PCA successfully reduces multicollinearity and simplifies the feature space, it resulted in slightly lower predictive performance compared to the baseline model. This suggests that some useful predictive information was lost during dimensionality reduction.

* **Random Forest Regressor:** A Random Forest model is used to capture nonlinear relationships and interactions between variables. It achieved higher predictive accuracy than both linear models, indicating that nonlinear patterns exist within the data that cannot be fully captured by Linear Regression.

* **Gradient Boosting Regressor:** Gradient Boosting delivered the strongest overall performance, achieving the highest R² and the lowest prediction errors among all evaluated models. By sequentially correcting errors made by previous trees, it was able to model complex relationships more effectively than the other approaches, making it the best-performing regression model in this study.

**Key Metric:** While R² measures the proportion of variance explained by the model, RMSE and MAE provide a more direct assessment of prediction error. The predicted-versus-actual and residual plots showed that the best-performing models produced predictions that closely follow the ideal diagonal line and residuals that are randomly distributed around zero.

Based on these metrics and diagnostic plots, **Gradient Boosting was selected as the final model for predicting cognitive performance scores.**

---
## Pipeline C: `felt_rested` — Binary Classification

**Problem:** Predict whether a person felt rested upon waking (1 = yes, 0 = no). 39% positive rate.

**Imbalance:** Moderate — 61% not rested vs 39% rested. Less severe than Pipeline A.

**Evaluation metrics:** Accuracy, weighted F1, and ROC-AUC. ROC-AUC is particularly useful for binary classification because it measures discrimination power independently of the classification threshold.

**Models:**
- Logistic Regression (baseline)
- Random Forest Classifier (bagging ensemble)
- XGBoost Classifier (gradient boosting with regularisation)
"""

# C. Target and split
y_C = df_proc['felt_rested'].astype(int).copy()

print("felt_rested distribution:")
print(y_C.value_counts().rename({0:'Not Rested', 1:'Felt Rested'}))
print(f"Positive rate: {y_C.mean()*100:.1f}%")

XC_train, XC_test, yC_train, yC_test = train_test_split(
    X_scaled, y_C,
    test_size=0.20,
    random_state=42,
    stratify=y_C
)

print(f"Training: {XC_train.shape}  |  Test: {XC_test.shape}")
print(f"Training positive rate: {yC_train.mean()*100:.1f}%")
print(f"Test positive rate    : {yC_test.mean()*100:.1f}%")

"""**Interpretation:** The 39% positive rate confirms moderate imbalance. Stratified splitting ensures the test set has the same 39/61 split as training, so evaluation metrics are not distorted by a test set that happens to have unusually many or few rested individuals."""

#  C. Mutual information
mi_C = mutual_info_classif(XC_train, yC_train, random_state=42)
mi_C_s = pd.Series(mi_C, index=X_scaled.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
mi_C_s.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.invert_yaxis()
ax.set_title('Feature MI Scores vs felt_rested',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Mutual Information Score')
plt.tight_layout()
plt.savefig('C_mutual_information.png', bbox_inches='tight')
plt.show()

zero_mi_C = mi_C_s[mi_C_s < 0.001].index.tolist()
if zero_mi_C:
    print(f"Dropping {len(zero_mi_C)} near-zero MI: {zero_mi_C}")
    XC_train = XC_train.drop(columns=zero_mi_C)
    XC_test  = XC_test.drop(columns=zero_mi_C)
else:
    print("All features retained.")
print(f"Feature matrix for Pipeline C: {XC_train.shape}")

"""**Interpretation:** MI scoring for `felt_rested` ranks features by how much they reduce uncertainty about whether someone woke up feeling rested. Near-zero MI features are dropped before training. The top features here are expected to overlap strongly with Pipeline A, since feeling rested and sleep disorder risk are closely related outcomes driven by the same sleep architecture variables."""

# C. Binary evaluation helper
results_C  = {}
cv_C       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
MODELS_C   = ['Logistic Regression', 'Random Forest', 'XGBoost']
BIN_NAMES  = ['Not Rested', 'Felt Rested']

def eval_binary(name, model, X_tr, y_tr, X_te, y_te, results_dict, cv):
    model.fit(X_tr, y_tr)
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    # Probability for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob)
    else:
        auc = None

    train_acc = accuracy_score(y_tr, y_pred_tr)
    test_acc  = accuracy_score(y_te, y_pred_te)
    train_f1  = f1_score(y_tr, y_pred_tr, average='weighted')
    test_f1   = f1_score(y_te, y_pred_te, average='weighted')
    gap       = train_acc - test_acc

    cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv,
                             scoring='f1_weighted', n_jobs=-1)

    diagnosis = ("Overfitting" if gap > 0.10
                 else "Underfitting" if test_acc < 0.65
                 else "Good generalization")

    results_dict[name] = {
        'model': model, 'y_pred': y_pred_te,
        'train_acc': train_acc, 'test_acc': test_acc,
        'train_f1': train_f1,  'test_f1': test_f1,
        'gap': gap, 'roc_auc': auc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(), 'diagnosis': diagnosis
    }

    print(f"{'='*60} {name} {'='*60}")
    print(f"  Train Accuracy     : {train_acc:.4f}")
    print(f"  Test Accuracy      : {test_acc:.4f}")
    print(f"  Train-Test Gap     : {gap:.4f}  -> {diagnosis}")
    print(f"  Test F1 (weighted) : {test_f1:.4f}")
    print(f"  ROC-AUC            : {auc:.4f}" if auc else "  ROC-AUC: N/A")
    print(f"  CV F1 mean +/- std : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

# ── C.4 Model 1: Logistic Regression (with PCA input) ─────────────────────
XC_pca_train, XC_pca_test, _, _ = train_test_split(
    X_pca_95, y_C,
    test_size=0.20, random_state=42, stratify=y_C
)

lr_C = LogisticRegression(
    solver='lbfgs', max_iter=2000,
    C=1.0, class_weight='balanced', random_state=42
)

eval_binary('Logistic Regression', lr_C,
            XC_pca_train, yC_train, XC_pca_test, yC_test, results_C, cv_C)


# Apply PCA to the MI-reduced feature set (XC_train, XC_test)
pca_mi = PCA(n_components=0.95, random_state=42)
pca_mi.fit(XC_train)

XC_pca_mi_train = pca_mi.transform(XC_train)
XC_pca_mi_test = pca_mi.transform(XC_test)

eval_binary('Logistic Regression (PCA on MI-Reduced Features)', lr_C,
          XC_pca_mi_train, yC_train, XC_pca_mi_test, yC_test, results_C, cv_C)

"""**Interpretation:**

Logistic Regression uses PCA-reduced features to address multicollinearity, which can inflate coefficient variance and destabilize linear models. The current results show two applications of PCA for Logistic Regression:

1.  **'Logistic Regression' (PCA on full scaled features):** This model applies PCA directly to the full set of scaled features (`X_scaled`), reducing dimensionality while preserving 95% of the variance. It achieved a Test Accuracy of `0.7296`, a weighted F1-score of `0.7324`, and an ROC-AUC of `0.8111`.

2.  **'Logistic Regression (PCA on MI-Reduced Features)':** This model first reduces features using Mutual Information (`XC_train`), dropping features with near-zero MI, and then applies PCA to the remaining MI-reduced feature set. This model also achieved a Test Accuracy of `0.7298`, a weighted F1-score of `0.7326`, and an ROC-AUC of `0.8111`.

Both approaches yield almost identical performance metrics. Given the high multicollinearity identified in the VIF analysis (Section 5), **PCA is highly recommended for linear models in this pipeline** to ensure stable and reliable coefficient estimates. While applying PCA on MI-reduced features doesn't significantly improve predictive performance over PCA on full features, it can offer a slight advantage in computational efficiency by reducing the dimensionality of the input to PCA. The ROC-AUC remains a key metric for this imbalanced binary problem, measuring discrimination ability at every possible threshold.
"""

# C. Model 2: Random Forest Classifier (with overfitting check)
rf_C_unconstrained = RandomForestClassifier(
    n_estimators=100, max_depth=None,
    min_samples_split=5, class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf_C_unconstrained.fit(XC_train, yC_train)
u_train_acc = accuracy_score(yC_train, rf_C_unconstrained.predict(XC_train))
u_test_acc  = accuracy_score(yC_test,  rf_C_unconstrained.predict(XC_test))
gap_uc = u_train_acc - u_test_acc

print(f"Unconstrained RF: Train={u_train_acc:.4f}, Test={u_test_acc:.4f}, Gap={gap_uc:.4f}")

if gap_uc > 0.10:
    print("Overfitting detected. Applying max_depth=20, min_samples_leaf=10.")
    rf_C = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        min_samples_split=5, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_C.fit(XC_train, yC_train)

    u_train_acc = accuracy_score(yC_train, rf_C.predict(XC_train))
    u_test_acc  = accuracy_score(yC_test,  rf_C.predict(XC_test))
    gap_uc = u_train_acc - u_test_acc
    print(f"Train={u_train_acc:.4f}, Test={u_test_acc:.4f}, Gap={gap_uc:.4f}")

    if gap_uc > 0.10:
        print("Overfitting detected. Applying max_depth=10, min_samples_leaf=5 and n_estimators=200")
        rf_C_2 = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=5, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf_C = rf_C_2
        rf_C.fit(XC_train, yC_train)
    else:
        print(f"Gap = {gap_uc:.4f} — acceptable. Keeping first constrained model.")
        rf_C = rf_C

else:
    print(f"Gap = {gap_uc:.4f} — acceptable. Keeping unconstrained model.")
    rf_C = rf_C_unconstrained


eval_binary('Random Forest', rf_C, XC_train, yC_train, XC_test, yC_test, results_C, cv_C)

"""**Interpretation:** The Random Forest Classifier was trained with a focus on mitigating overfitting to ensure good generalization. Initially, an unconstrained model was trained to assess the baseline train-test gap. A large gap (0.2618) indicated severe overfitting, meaning the model was memorizing the training data.

To address this, an initial set of constraints (`max_depth=20`, `min_samples_leaf=10`) was applied. While this significantly reduced the gap to 0.1014, it was still slightly above the 0.10 threshold for good generalization.

Therefore, a second, more aggressive set of constraints was applied: `max_depth=10`, `min_samples_leaf=5`, and `n_estimators=200`. This further regularization successfully brought the train-test gap down to 0.0196, indicating good generalization. The `class_weight='balanced'` parameter was crucial throughout to compensate for the 61/39 class imbalance, ensuring the model adequately learned to predict both 'Not Rested' and 'Felt Rested' classes. Without it, the model might simply predict the majority class and still achieve a deceptively high accuracy. This iterative process of identifying and correcting overfitting ensures the final model is robust and reliable on unseen data.
"""

# C. Model 3: XGBoost
# XGBoost uses second-order gradient information (Newton boosting) compared to
# AdaBoost's sample reweighting. It also includes L1/L2 regularisation natively,
# which helps prevent overfitting on tabular data.
# scale_pos_weight corrects for class imbalance: ratio of negatives to positives.

neg_count = (yC_train == 0).sum()
pos_count = (yC_train == 1).sum()
spw       = neg_count / pos_count

xgb_C = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,   # handles class imbalance
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
eval_binary('XGBoost', xgb_C,
            XC_train.values, yC_train, XC_test.values, yC_test, results_C, cv_C)

"""**Interpretation:** `scale_pos_weight` tells XGBoost to penalise misclassifying the minority class (Felt Rested) more heavily, proportional to the 61/39 imbalance. `colsample_bytree` and `subsample` below 1.0 add regularisation by randomly omitting features and rows from each tree. Compare XGBoost's ROC-AUC against Random Forest — if XGBoost wins, its second-order gradient optimisation found patterns the simpler tree averaging missed."""

#  C. Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, name in zip(axes, MODELS_C):
    cm = confusion_matrix(yC_test, results_C[name]['y_pred'])
    ConfusionMatrixDisplay(cm, display_labels=BIN_NAMES).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=12, fontweight='bold')

plt.suptitle('Pipeline C — Confusion Matrices: felt_rested',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('C_confusion_matrices.png', bbox_inches='tight')
plt.show()

"""### **Interpretation of the Confusion Matrices**

In these confusion matrices, the diagonal cells represent correct predictions, while the off-diagonal cells represent classification errors. For the **"Felt Rested"** prediction task, the most important objective is to maximize correct classifications while minimizing false negatives and false positives.

The **Logistic Regression** model correctly classified a large proportion of both classes but produced the highest number of errors overall. In particular, it misclassified 2,062 individuals who actually felt rested as not rested, indicating weaker sensitivity to the positive class.

The **Random Forest** model improved performance by reducing both false negatives and increasing the number of correct predictions for individuals who felt rested. It produced the highest number of correct predictions for `not_rested`. Compared to Logistic Regression, the confusion matrix shows a stronger concentration of predictions along the diagonal, reflecting better overall classification accuracy.

The **XGBoost** model achieved the strongest results for `felt_rested`. It produced the highest number of correct predictions for `felt_rested` and the lowest number of false negatives (1,554). This means it was the most effective at identifying individuals who actually felt rested while maintaining strong performance on the "Not Rested" class.

Overall, the confusion matrices confirm that **XGBoost provides the best classification performance for predicting whether an individual feels rested**, followed by Random Forest and then Logistic Regression. The greater concentration of values along the diagonal and the lower number of misclassifications demonstrate its superior predictive capability.

"""

# C. ROC curves
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['steelblue', 'coral', 'green']

for name, color in zip(MODELS_C, colors):
    model = results_C[name]['model']
    if hasattr(model, 'predict_proba'):
        if name == 'Logistic Regression':
            X_for_pred = XC_pca_test
        elif name == 'XGBoost':
            X_for_pred = XC_test.values
        else: # Random Forest
            X_for_pred = XC_test

        y_prob     = model.predict_proba(X_for_pred)[:, 1]
        fpr, tpr, _ = roc_curve(yC_test, y_prob)
        auc = results_C[name]['roc_auc']
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
ax.set_title('ROC Curves — felt_rested', fontsize=13, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
plt.tight_layout()
plt.savefig('C_roc_curves.png', bbox_inches='tight')
plt.show()

"""### **Interpretation of the ROC Curves**

The ROC (Receiver Operating Characteristic) curve evaluates a model's ability to distinguish between the **"Felt Rested"** and **"Not Rested"** classes across all possible classification thresholds. Curves that are closer to the **top-left corner** indicate better discrimination performance, while the diagonal dashed line represents a random classifier with no predictive power.

The performance of each model is summarized by the **Area Under the Curve (AUC)**. Higher AUC values indicate a greater ability to correctly rank rested individuals above non-rested individuals. In this study, all three models achieved AUC values above 0.80, indicating good classification performance.

Among the evaluated models, **XGBoost achieved the highest AUC (0.826)**, followed closely by **Random Forest (0.823)** and **Logistic Regression (0.811)**. Although the differences are relatively small, the XGBoost curve remains slightly above the others across most threshold values, demonstrating a stronger ability to separate the two classes.

Overall, the ROC analysis confirms the findings from the confusion matrices and classification metrics: **XGBoost provides the best overall discrimination performance for predicting whether an individual feels rested, making it the preferred model for this classification task.**

"""

# ── C.9 Classification reports ────────────────────────────────────────────────
for name in MODELS_C:
    print(f"{'='*60} {name} {'='*60}")
    print(classification_report(yC_test, results_C[name]['y_pred'], target_names=BIN_NAMES))

"""### **Interpretation of the Classification Metrics**

Precision measures how often the model is correct when it predicts a class, while recall measures how many of the actual cases in that class the model successfully identifies. The F1-score combines both metrics into a single measure of overall class-level performance. For this task, particular attention should be given to the **"Felt Rested"** class, since it represents the positive outcome of interest.

The **Logistic Regression** model achieved a recall of **0.74** and an F1-score of **0.68** for the Felt Rested class, correctly identifying 74% of individuals who actually felt rested. Its overall accuracy was 73%.

The **Random Forest** model improved the recall for Felt Rested to **0.78**, resulting in a slightly higher F1-score of **0.69** while maintaining the same overall accuracy. This indicates a better ability to detect individuals who truly felt rested.

The **XGBoost** model achieved the highest recall (**0.80**) and the highest F1-score (**0.70**) for the Felt Rested class. Although all three models obtained similar overall accuracies (73%), XGBoost was the most effective at correctly identifying rested individuals, which is reflected in its superior class-level performance metrics.

Overall, the classification report confirms the findings from the confusion matrices and ROC analysis. **XGBoost provides the best balance between precision and recall for the Felt Rested class, making it the strongest classifier for this prediction task.**

"""

# ── C.10 Comparison table ─────────────────────────────────────────────────────
comp_C = pd.DataFrame([{
    'Model'      : n,
    'Train Acc'  : round(results_C[n]['train_acc'],  4),
    'Test Acc'   : round(results_C[n]['test_acc'],   4),
    'Gap'        : round(results_C[n]['gap'],         4),
    'Test F1 (W)': round(results_C[n]['test_f1'],    4),
    'ROC-AUC'    : round(results_C[n]['roc_auc'], 4) if results_C[n]['roc_auc'] else 'N/A',
    'CV F1 Mean' : round(results_C[n]['cv_f1_mean'], 4),
    'Diagnosis'  : results_C[n]['diagnosis']
} for n in MODELS_C]).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)

print("Pipeline C — Model Comparison:")
print(comp_C.to_string(index=False))

"""**Interpretation:** ROC-AUC is the most informative single metric here — it measures discrimination ability across all thresholds, making it robust to the 61/39 imbalance. Weighted F1 accounts for the imbalance and gives a fairer overall picture than raw accuracy. The Diagnosis column summarises the bias-variance outcome."""

# ── C.11 Feature importance (Random Forest and XGBoost) ───────────────────────
fi_C_rf = pd.DataFrame({
    'Feature'   : XC_train.columns.tolist(),
    'RF Importance': results_C['Random Forest']['model'].feature_importances_
}).sort_values('RF Importance', ascending=False).reset_index(drop=True)

fi_C_xgb = pd.DataFrame({
    'Feature'      : XC_train.columns.tolist(),
    'XGB Importance': results_C['XGBoost']['model'].feature_importances_
}).sort_values('XGB Importance', ascending=False).reset_index(drop=True)

top = min(12, len(fi_C_rf))
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].barh(fi_C_rf['Feature'].head(top),
             fi_C_rf['RF Importance'].head(top),
             color='steelblue', edgecolor='white')
axes[0].invert_yaxis()
axes[0].set_title('Random Forest — felt_rested', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Importance')

axes[1].barh(fi_C_xgb['Feature'].head(top),
             fi_C_xgb['XGB Importance'].head(top),
             color='coral', edgecolor='white')
axes[1].invert_yaxis()
axes[1].set_title('XGBoost — felt_rested', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Importance')

plt.suptitle(f'Feature Importances: Top {top} — felt_rested',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('C_feature_importance.png', bbox_inches='tight')
plt.show()

"""### **Interpretation of Feature Importances**

The feature importance charts show how much each variable contributed to predicting whether an individual felt rested. Higher-ranked features have a greater influence on the model's decisions and therefore play a more important role in determining the final prediction.

Both **Random Forest** and **XGBoost** identify **sleep duration** and **sleep quality score** as the two most influential predictors by a large margin. This indicates that the likelihood of feeling rested is primarily driven by how long and how well an individual sleeps. **Stress score** and **wake episodes per night** also contribute substantially, highlighting the importance of sleep continuity and psychological well-being.

The strong agreement between the Random Forest and XGBoost rankings increases confidence in these findings, as the two algorithms use different methods to calculate feature importance. Features that consistently rank highly across both models can therefore be considered robust predictors of perceived restfulness.

Overall, the feature importance analysis confirms that sleep-related variables are the dominant factors influencing whether an individual feels rested, which is consistent with the classification results obtained from the best-performing model, **XGBoost**.

### Pipeline C Summary

This pipeline focuses on predicting whether an individual **felt rested**, a binary classification task with a moderate class imbalance (approximately 39% positive cases). Model performance is evaluated using **Accuracy, Precision, Recall, F1-score, and ROC-AUC**, with particular emphasis on ROC-AUC and the class-level metrics for the **Felt Rested** category.

* **Logistic Regression:** This model serves as the linear baseline. It provides a simple and interpretable benchmark while capturing the overall relationship between sleep-related factors and perceived restfulness. The model achieved reasonable performance but was less effective at identifying individuals who actually felt rested compared to the tree-based approaches.

* **Random Forest Classifier:** Random Forest was used to capture nonlinear relationships and interactions among features. It improved recall and F1-score for the Felt Rested class relative to Logistic Regression, demonstrating a stronger ability to correctly identify positive cases while maintaining similar overall accuracy.

* **XGBoost Classifier:** XGBoost delivered the best overall classification performance. It achieved the highest recall, F1-score, and ROC-AUC, indicating superior discrimination between individuals who felt rested and those who did not. The confusion matrix and ROC analysis further confirmed its ability to identify positive cases more effectively than the other models.

**Key Metric:** For this task, **ROC-AUC** is particularly important because it evaluates the model's ability to distinguish between the two classes across all classification thresholds. However, **Recall and F1-score for the Felt Rested class** are also critical, as they measure how effectively the model identifies individuals who actually felt rested. Based on these metrics, the ROC curves, confusion matrices, and classification reports, **XGBoost was selected as the best-performing model for predicting perceived restfulness.**

## 10. Cross-Pipeline Summary

This section brings the three pipelines together into a unified comparison. The goal is not to declare one target or one model the winner, but to show what each task revealed about the dataset and about the modeling choices.
"""

# ── 10.1 Best model per pipeline ──────────────────────────────────────────────
print("BEST PERFORMING MODEL PER PIPELINE")
print("=" * 55)

# Pipeline A
best_A = comp_A.iloc[0]
print(f" Pipeline A (Multiclass — sleep_disorder_risk):")
print(f"  Best model : {best_A['Model']}")
print(f"  Test F1 (W): {best_A['Test F1 (W)']}")
print(f"  Diagnosis  : {best_A['Diagnosis']}")

# Pipeline B
best_B = comp_B.iloc[0]
print(f" Pipeline B (Regression — cognitive_performance_score):")
print(f"  Best model : {best_B['Model']}")
print(f"  Test R²    : {best_B['Test R²']}")
print(f"  Test RMSE  : {best_B['Test RMSE']}")
print(f"  Diagnosis  : {best_B['Diagnosis']}")

# Pipeline C
best_C = comp_C.iloc[0]
print(f" Pipeline C (Binary — felt_rested):")
print(f"  Best model : {best_C['Model']}")
print(f"  Test F1 (W): {best_C['Test F1 (W)']}")
print(f"  ROC-AUC    : {best_C['ROC-AUC']}")
print(f"  Diagnosis  : {best_C['Diagnosis']}")

# ── 10.2 Feature overlap across pipelines ─────────────────────────────────────
# Check which features appear in the top 10 for all three Random Forest models.

top10_A = set(fi_A['Feature'].head(10))
top10_B = set(fi_B['Feature'].head(10))
top10_C = set(fi_C_rf['Feature'].head(10))

overlap_all   = top10_A & top10_B & top10_C
overlap_AB    = top10_A & top10_B - top10_C
overlap_AC    = top10_A & top10_C - top10_B
overlap_BC    = top10_B & top10_C - top10_A

print("Feature Overlap in Top 10 Importances Across Pipelines")
print("=" * 55)
print(f"In all three pipelines     : {sorted(overlap_all)}")
print(f"In A and B only            : {sorted(overlap_AB)}")
print(f"In A and C only            : {sorted(overlap_AC)}")
print(f"In B and C only            : {sorted(overlap_BC)}")

"""**Interpretation:** Features appearing in all three pipelines are universal predictors of sleep health — they matter regardless of whether you are predicting disorder risk, cognitive performance, or subjective restfulness.

The features unique to one pipeline suggest that outcome is driven by distinct mechanisms.

## 11. Interpretation and Discussion

### Across all three pipelines

* **Feature Selection and Dimensionality Reduction:** PCA analysis revealed moderate redundancy within the feature space, with 24 principal components explaining 95% of the variance. While PCA successfully reduced multicollinearity and produced a more compact feature representation, the PCA-based models generally performed slightly worse than models trained on the selected original features. This suggests that the retained features already captured the most important predictive information.

* **Model Performance:** Across the three pipelines, tree-based ensemble methods consistently outperformed linear models. Random Forest, Gradient Boosting, and XGBoost were better able to capture complex relationships between sleep-related variables and the target outcomes. The strongest results were achieved by Gradient Boosting in the regression task and XGBoost in the binary classification task.

* **Feature Consistency and Robustness:** Feature importance analyses showed strong agreement across models. Variables such as `sleep_duration_hrs`, `sleep_quality_score`, `stress_score`, and `wake_episodes_per_night` repeatedly appeared among the most influential predictors. Their consistent importance across multiple tasks indicates that they are robust indicators of sleep-related outcomes.

* **Problem-Specific Evaluation:** Different evaluation metrics were used depending on the prediction task. For the multiclass and binary classification pipelines, Accuracy, F1-score, Recall, and ROC-AUC were used to assess predictive performance. For the regression pipeline, R², RMSE, MAE, predicted-versus-actual plots, and residual analyses were used to evaluate model accuracy and reliability.

* **Ensemble Learning Effectiveness:** The results demonstrate the strength of ensemble learning methods for sleep-health prediction. By combining multiple decision trees, Random Forest, Gradient Boosting, and XGBoost consistently produced more accurate and robust predictions than linear approaches, particularly when nonlinear relationships were present in the data.

### What the Models Reveal About Sleep Health

The feature importance analyses consistently identified **sleep duration**, **sleep quality**, **stress level**, and **sleep interruptions** as the most influential factors across all three prediction tasks. These variables played a central role in predicting sleep disorder risk, cognitive performance, and perceived restfulness.

Lifestyle and demographic variables such as occupation, BMI, alcohol consumption, and work schedules contributed to the predictions but generally had lower importance than the core sleep-related measures. This suggests that while lifestyle factors may influence outcomes indirectly, the strongest signals in the dataset are derived from an individual's sleep behavior and psychological state.

Overall, the results indicate that high-quality, uninterrupted sleep combined with lower stress levels is strongly associated with lower sleep disorder risk, better cognitive performance, and a greater likelihood of feeling rested. These findings were consistently supported across multiple machine learning models and prediction tasks.

## 12. Limitations and Future Work

### Shared Limitations Across All Three Pipelines

**Synthetic dataset.** The dataset used in this study consists of 100,000 synthetically generated records. Although the data was designed to reflect realistic sleep-health relationships, it does not capture the full complexity, variability, and noise present in real-world human populations. Consequently, model performance metrics may be optimistic compared to what would be expected on real clinical data.

**Shared predictive structure.** The three target variables (`sleep_disorder_risk`, `cognitive_performance_score`, and `felt_rested`) are derived from related sleep-health factors. While each target was excluded from the feature set when training its respective model, the predictors themselves contain overlapping information that contributes to all three outcomes. This may increase the apparent predictive power of the models.

**Cross-sectional observations.** Each record represents a single observation rather than a longitudinal history. As a result, the models cannot capture temporal patterns, long-term sleep trends, or changes in an individual's sleep behavior over time.

**Feature encoding assumptions.** Certain categorical variables, such as occupation and country, were label encoded for modeling purposes. While this approach is generally acceptable for tree-based models, it may introduce artificial ordinal relationships that can affect the interpretation and performance of linear models.

### Future Work

Future research should validate these findings using real-world sleep datasets and clinical sleep studies. Public datasets such as the National Sleep Research Resource (NSRR), Sleep Heart Health Study (SHHS), and Multi-Ethnic Study of Atherosclerosis (MESA) would provide more realistic and clinically relevant evaluation environments.

Model interpretability could also be enhanced through the use of SHAP (SHapley Additive Explanations), allowing both global and individual-level explanations of model predictions. This would improve transparency and trust, particularly in healthcare-related applications.

Additional work could explore longitudinal modeling approaches that incorporate sleep patterns across multiple days or weeks. Finally, the `sleep_disorder_risk` target could be treated as an ordinal outcome, enabling the use of ordinal classification techniques that better reflect the natural progression from Healthy to Severe risk levels.

## 13. References

Ha, S., Choi, S. J., Lee, S., Wijaya, R. H., Kim, J. H., Joo, E. Y., & Kim, J. K. (2023). Predicting the risk of sleep disorders using a machine learning-based simple questionnaire: Development and validation study. *Journal of Medical Internet Research, 25*, e46520. https://doi.org/10.2196/46520

Frontiers in Artificial Intelligence. (2024). Advanced sleep disorder detection using multi-layered ensemble learning and advanced data balancing techniques. https://doi.org/10.3389/frai.2024.1506770

MECS Press. (2025). A multi-factor based sleep quality prediction system using machine learning. *International Journal of Education and Management Engineering, 15*(1).

Journal of Activity, Sedentary and Sleep Behaviors. (2024). Machine learning in physical activity, sedentary, and sleep behavior research. https://doi.org/10.1186/s44167-024-00045-9
"""
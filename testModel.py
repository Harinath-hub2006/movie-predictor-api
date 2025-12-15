import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from xgboost import XGBRegressor
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer # Moved up for cleaner flow

# --- 1. Load Data ---
df_movies = pd.read_csv('tmdb_5000_movies.csv')
df_credits = pd.read_csv('tmdb_5000_credits.csv')

# --- 2. Merge DataFrames ---
df_credits.columns = ['movie_id', 'title', 'cast', 'crew']
df_merged = df_movies.merge(df_credits, on='title')

print(f"Total Movies after merge: {len(df_merged)}")

# --- 3. Clean Target Variable (Revenue) and Budget ---
df_cleaned = df_merged[(df_merged['revenue'] != 0) & (df_merged['budget'] != 0)].copy()
df_cleaned['revenue_log'] = np.log1p(df_cleaned['revenue'])

# --- 4. Helper Function to Extract Names from JSON-like Strings ---
def get_names(json_string):
    """Safely converts a JSON string representation of a list of dicts to a list of 'name' values."""
    if isinstance(json_string, str):
        try:
            list_of_dicts = literal_eval(json_string) 
            # Extract names, taking only the first 3 elements for simplicity/focus
            return [d['name'] for d in list_of_dicts[:3]] 
        except (ValueError, SyntaxError):
            return []
    return []

# --- 5. Apply Extraction to Key Columns ---
df_cleaned['genres_list'] = df_cleaned['genres'].apply(get_names)
df_cleaned['keywords_list'] = df_cleaned['keywords'].apply(get_names)
df_cleaned['top_cast'] = df_cleaned['cast'].apply(get_names) 

# --- 6. Director Extraction ---
def get_director(crew_json):
    if isinstance(crew_json, str):
        try:
            crew_list = literal_eval(crew_json)
            for member in crew_list:
                if member['job'] == 'Director':
                    return member['name']
        except (ValueError, SyntaxError):
            pass
    return np.nan

df_cleaned['director'] = df_cleaned['crew'].apply(get_director)

# --- 7. Time Series Feature Engineering ---
df_cleaned['release_date'] = pd.to_datetime(df_cleaned['release_date'], errors='coerce')
df_cleaned['release_month'] = df_cleaned['release_date'].dt.month
df_cleaned['release_year'] = df_cleaned['release_date'].dt.year

# --- 8. One-Hot Encoding for 'original_language' (THE FIX) ---
# This converts 'en', 'fr', etc. into numerical columns (e.g., language_en, language_fr)
language_dummies = pd.get_dummies(df_cleaned['original_language'], 
                                   prefix='language', 
                                   dummy_na=False)

# Add language dummies to the main DataFrame
df_cleaned = pd.concat([df_cleaned, language_dummies], axis=1)

# Drop ALL original complex/unneeded columns and the original 'original_language'
df_final = df_cleaned.drop(columns=[
    'homepage', 'genres', 'keywords', 'tagline', 'cast', 'crew', 
    'overview', 'original_title', 'status', 'revenue', 
    'original_language', 
    'spoken_languages', 
    'production_companies', # <--- NEW FIX: Drop this uncleaned JSON column
    'production_countries', # <--- Good idea to drop this too, as it's also a JSON string
    'id', 'movie_id' 
])

# --- 9. Multi-Label Encoding for Genre and Keyword Lists ---
mlb = MultiLabelBinarizer()

# Apply to Genres
genre_dummies = pd.DataFrame(mlb.fit_transform(df_final['genres_list']), 
                             columns=[f'Genre_{c}' for c in mlb.classes_], 
                             index=df_final.index)

# Apply to Keywords
keyword_dummies = pd.DataFrame(mlb.fit_transform(df_final['keywords_list']), 
                               columns=[f'Keyword_{c}' for c in mlb.classes_], 
                               index=df_final.index)

# --- 10. Simple Encoding for Director (Top K) ---
TOP_DIRECTORS = df_final['director'].value_counts().head(20).index
df_final['director_is_top'] = df_final['director'].apply(lambda x: 1 if x in TOP_DIRECTORS else 0)

# Concatenate all list-based features and drop the temporary list columns
df_final = pd.concat([df_final, genre_dummies, keyword_dummies], axis=1).drop(columns=['genres_list', 'keywords_list', 'top_cast', 'director', 'release_date'])

# Final check for missing numerical values (e.g., in runtime, vote_average) and fill with the mean
df_final = df_final.fillna(df_final.mean(numeric_only=True))

# Select Features (X) and Target (y)
X = df_final.drop(columns=['title', 'revenue_log']) 
y = df_final['revenue_log']

# --- 11. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 12. Model Training (XGBoost Regressor - Original Model) ---
print("Starting Model Training...")
# Switching to XGBoost for better handling of outliers and non-linearity
from xgboost import XGBRegressor # Ensuring import is available
model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1) 
model.fit(X_train, y_train)

# --- 13. Prediction ---
y_pred_log = model.predict(X_test)

# --- 14. Evaluation ---
# Convert predictions and test data back from log scale (np.expm1 is the inverse of np.log1p)
y_pred_actual = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

print("\n--- Model Evaluation ---")
print(f"Number of Features Used: {X.shape[1]}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

# =========================================================
# === STEP 15: FEATURE SELECTION FOR DEPLOYMENT ===
# =========================================================

# 1. Extract and sort feature importance
importances = model.feature_importances_
feature_names = X.columns
feature_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# 2. Select the top N features (e.g., Top 20)
TOP_N = 20
top_20_features = feature_series.head(TOP_N).index.tolist()

print(f"\n--- Top {TOP_N} Predictive Features ---")
print(top_20_features)
print(f"Combined Importance of Top {TOP_N}: {feature_series.head(TOP_N).sum():.4f}")

# 3. Create a simplified feature matrix (X_simplified)
X_simplified = X[top_20_features]
y_simplified = y

# 4. Split and Retrain a SIMPLIFIED Model (M_simple)
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simplified, y_simplified, test_size=0.2, random_state=42
)

# 4. Split and Retrain a SIMPLIFIED Model (M_simple - XGBoost)
# ...
M_simple = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
M_simple.fit(X_train_simple, y_train_simple)

# Optional check: Compare RMSE of simple model vs. complex one
y_pred_simple_log = M_simple.predict(X_test_simple)
rmse_simple = np.sqrt(mean_squared_error(np.expm1(y_test_simple), np.expm1(y_pred_simple_log)))
print(f"Simplified Model RMSE (Top {TOP_N} Features): ${rmse_simple:,.2f}")

# =========================================================
# === STEP 16: CUSTOM PREDICTION WITH SIMPLIFIED FEATURES ===
# =========================================================

# 1. DEFINE YOUR HYPOTHETICAL MOVIE'S DATA using ONLY the Top N features.
# You must look at the 'top_20_features' list printed above!

# Example Top Features (these are hypothetical, use your script's output):
# ['budget', 'popularity', 'vote_count', 'runtime', 'vote_average', 'Genre_Action', 'language_en', ...]

hypothetical_input = {
    # Numerical Features (Must match your top feature list):
    'budget': 180000000.0,
    'popularity': 65.0,
    'vote_count': 2500.0,
    'runtime': 130.0,
    'vote_average': 8.0,
    
    # One-Hot Encoded Features (Set to 1 or 0):
    'Genre_Action': 1.0,  # Is it an action movie? (1=Yes)
    'language_en': 1.0,   # Is it in English? (1=Yes)
    # Set all other Top 20 features to 0.0 or their defined value
    
    # Add the remaining 12 features from your actual TOP_N list here...
}

# 2. CREATE A DATAFRAME for the new movie using ONLY the Top N columns
X_predict_simple = pd.DataFrame(0, index=[0], columns=top_20_features) 

# Update the defined features for the hypothetical movie
for feature, value in hypothetical_input.items():
    if feature in X_predict_simple.columns:
        X_predict_simple.loc[0, feature] = value
    else:
        # Note: If you defined features outside the Top 20, they are ignored here.
        pass

# 3. RUN THE PREDICTION using the SIMPLIFIED model (M_simple)
predicted_revenue_log_simple = M_simple.predict(X_predict_simple)
predicted_revenue_actual_simple = np.expm1(predicted_revenue_log_simple)

# 4. DISPLAY THE RESULT
print("\n--- Simplified Custom Prediction ---")
print(f"Features Used for Input: {len(X_predict_simple.columns)}")
print(f"Predicted Revenue: ${predicted_revenue_actual_simple[0]:,.2f}")

# --- 1. Extract and Plot Feature Importance for Top 20 ---
importances = M_simple.feature_importances_
feature_names = X_simplified.columns # Use the columns from the simplified X
feature_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
# Use the feature_series values and index directly
sns.barplot(x=feature_series.values, y=feature_series.index, palette="viridis")

plt.title(f'Top {len(feature_series)} Features Driving Movie Revenue Prediction', fontsize=16)
plt.xlabel('Feature Importance Score (Combined Importance: 0.9241)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance_top20.png') # Save the plot
plt.show()

# --- 2. Actual vs. Predicted Scatter Plot (Using Simplified Model Results) ---
y_test_actual_simple = np.expm1(y_test_simple) # Use the test set from the simple split
y_pred_actual_simple = np.expm1(y_pred_simple_log)

plt.figure(figsize=(10, 10))
# Plot the ideal line
plt.plot([y_test_actual_simple.min(), y_test_actual_simple.max()], 
         [y_test_actual_simple.min(), y_test_actual_simple.max()], 'k--', lw=2, label='Perfect Prediction Line')
# Plot the scatter points
plt.scatter(y_test_actual_simple, y_pred_actual_simple, alpha=0.3)

plt.xscale('log') 
plt.yscale('log') 

plt.title('Actual vs. Predicted Revenue (Log Scale) - Simple Model', fontsize=16)
plt.xlabel('Actual Revenue (Log Scale)', fontsize=12)
plt.ylabel('Predicted Revenue (Log Scale)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig('actual_vs_predicted_top20.png') # Save the plot
plt.show()

# --- 3. Revenue Distribution Plot (Same as before, still needed for context) ---
df_plot = df_merged[(df_merged['revenue'] != 0)].copy()

plt.figure(figsize=(12, 5))

# Subplot 1: Highly skewed original revenue
plt.subplot(1, 2, 1)
sns.histplot(df_plot['revenue'], kde=True, bins=50)
plt.title('Original Revenue Distribution (Highly Skewed)', fontsize=12)
plt.xlabel('Revenue (Billions of Dollars)', fontsize=10)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,9))

# Subplot 2: Normal-looking log-transformed revenue
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df_plot['revenue']), kde=True, bins=50)
plt.title('Log-Transformed Revenue Distribution (Better for ML)', fontsize=12)
plt.xlabel('Log(1 + Revenue)', fontsize=10)

plt.tight_layout()
plt.savefig('revenue_distribution.png')
plt.show()

# =========================================================
# === STEP 17: MODULAR PREDICTION FUNCTION ===
# =========================================================

def predict_movie_revenue(raw_input_data: dict, model, feature_list: list, df_template_cols: pd.Index) -> float:
    """
    Takes a dictionary of simple features, converts it to the required 20-feature vector,
    and returns the predicted actual revenue.

    Args:
        raw_input_data: Dictionary of user-friendly features (e.g., {'budget': 200000000.0, 'Genre_Action': 1.0}).
        model: The trained scikit-learn model (M_simple).
        feature_list: The list of 20 feature names the model was trained on (top_20_features).
        df_template_cols: The full Index of the simplified feature columns (X_simplified.columns).

    Returns:
        Predicted revenue in actual dollar amount.
    """
    
    # 1. Create a template DataFrame filled with zeros (shape = 1 row, 20 columns)
    X_predict = pd.DataFrame(0.0, index=[0], columns=df_template_cols)
    
    # 2. Map the raw input to the template
    for feature, value in raw_input_data.items():
        if feature in X_predict.columns:
            # Ensure the value is float type for consistency
            X_predict.loc[0, feature] = float(value)
        # Note: If an input feature is missing from the top_20 list, it is safely ignored.

    # 3. Run the prediction (outputs log-transformed revenue)
    predicted_revenue_log = model.predict(X_predict)
    
    # 4. Convert back to actual dollars
    predicted_revenue_actual = np.expm1(predicted_revenue_log)
    
    return predicted_revenue_actual[0]

# =========================================================
# === STEP 18: USER-EDITABLE MODULAR PREDICTION TEST ===
# =========================================================

# The Top 20 features from your previous output (ensure this list is accurate):
TOP_20_FEATURES = ['vote_count', 'budget', 'release_year', 'vote_average', 'Keyword_comedy', 'popularity', 'runtime', 'Keyword_berlin', 'Keyword_eroticism', 'release_month', 'Keyword_key', 'Keyword_venice', 'Genre_Science Fiction', 'Genre_Comedy', 'Keyword_suspense', 'Keyword_woman director', 'Genre_Crime', 'Keyword_mannequin', 'Keyword_arbitrary law', 'Keyword_partner']


# --- DEFINE YOUR CUSTOM MOVIE HERE ---
# Use the feature names from the TOP_20_FEATURES list as the dictionary keys.
# If a feature is not listed here, it defaults to 0.0 in the prediction function.

your_custom_movie = {
    # 1. Continuous Numerical Inputs:
    'budget': 40000000.0,      # Example: $100 Million Budget
    'popularity': 85.0,         # Example: Average popularity score
    'vote_count': 2400.0,       # Example: 3000 user votes
    'runtime': 107.0,           # Example: 105 minutes
    'vote_average': 8.2,        # Example: 7.0/10 rating
    'release_year': 1999.0,     # Example: Future release year
    'release_month': 8.0,       # Example: July (Summer Blockbuster)
    
    # 2. Binary/Categorical Inputs (Set to 1.0 if TRUE, or omit if 0.0):
    'Genre_Comedy': 0.0,        # It IS a Comedy
    'Genre_Science Fiction': 1.0, # It is NOT Sci-Fi (you can also just omit this line)
    'Genre_Crime': 1.0,
    'Keyword_woman director': 0.0, # It DOES have a woman director (from the top 20 list)
    'Keyword_eroticism': 0.0,
    # The rest of the Top 20 features (like Keyword_berlin, Keyword_venice, etc.) will be 0.0
}


# --- RUN PREDICTION (No changes needed below this line) ---
final_prediction = predict_movie_revenue(
    raw_input_data=your_custom_movie,
    model=M_simple,
    feature_list=TOP_20_FEATURES,
    df_template_cols=X_simplified.columns
)

print("\n--- TEST: User-Defined Custom Prediction ---")
print(f"Input Budget: ${your_custom_movie.get('budget', 0.0):,.2f}")
print(f"Features Provided: {len(your_custom_movie)}")
print(f"Predicted Revenue: ${final_prediction:,.2f}")

# --- CRITICAL: Ensure this list matches the output of your XGBoost training ---
TOP_20_FEATURES = ['vote_count', 'budget', 'release_year', 'vote_average', 'Keyword_comedy', 'popularity', 'runtime', 'Keyword_berlin', 'Keyword_eroticism', 'release_month', 'Keyword_key', 'Keyword_venice', 'Genre_Science Fiction', 'Genre_Comedy', 'Keyword_suspense', 'Keyword_woman director', 'Genre_Crime', 'Keyword_mannequin', 'Keyword_arbitrary law', 'Keyword_partner']

# Create a folder for deployment assets inside MOV_PROJECT
os.makedirs('deployment_assets', exist_ok=True)

# 1. Save the Trained Simplified Model (M_simple is now XGBoost)
# Note: M_simple must be accessible in the environment when you run this line.
joblib.dump(M_simple, 'deployment_assets/movie_predictor_model.pkl')
print("\n[SUCCESS] XGBoost Model saved to deployment_assets/movie_predictor_model.pkl")

# 2. Save the List of Top 20 Feature Names
feature_data = {
    'features': TOP_20_FEATURES
}
with open('deployment_assets/feature_list.json', 'w') as f:
    json.dump(feature_data, f)
print("[SUCCESS] Feature list saved to deployment_assets/feature_list.json")
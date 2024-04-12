import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to load data
@st.cache_data
def load_data():
    data_path = 'sleep.csv'  # Specific dataset path
    return pd.read_csv(data_path)

# Load the dataset and avoid mutation warning by cloning
sleep_data = load_data().copy()

# Data preprocessing configuration
categorical_features = ['Gender', 'Smoking status']
numeric_features = ['Age', 'Caffeine consumption', 'Alcohol consumption', 'Exercise frequency', 'Daily Steps']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Applying preprocessing
X = preprocessor.fit_transform(sleep_data)
encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
features = numeric_features + list(encoded_cat_features)
X_df = pd.DataFrame(X, columns=features)
y = sleep_data['Sleep efficiency']

# Train the KMeans model
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_df)
sleep_data['Cluster'] = clusters

# Train the Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)
rf_regressor.fit(X_train, y_train)

def recommend_improvements(new_data, model, preprocessor, kmeans, data_df, cluster_centers):
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])  # Ensure new_data is a DataFrame

    new_data_processed = pd.DataFrame(preprocessor.transform(new_data), columns=features)
    current_cluster = kmeans.predict(new_data_processed)[0]
    current_efficiency_prediction = model.predict(new_data_processed)[0]
    
    mean_efficiencies = data_df.groupby('Cluster')['Sleep efficiency'].mean()
    better_clusters = mean_efficiencies[mean_efficiencies > mean_efficiencies.iloc[current_cluster]]
    target_cluster = better_clusters.idxmax() if not better_clusters.empty else current_cluster
    
    target_centroid = pd.DataFrame(cluster_centers, columns=features).iloc[target_cluster]
    recommendations_raw = (target_centroid - new_data_processed.iloc[0])
    actionable_features = ['Caffeine consumption', 'Alcohol consumption', 'Exercise frequency', 'Daily Steps']
    recommendations = {}
    
    for feature in actionable_features:
        change = recommendations_raw[feature]
        if feature in ['Alcohol consumption', 'Caffeine consumption'] and change > 0:
            recommendations[feature] = f"Reduce by {abs(change):.2f} units"
        elif feature in ['Exercise frequency', 'Daily Steps'] and change < 0:
            recommendations[feature] = f"Increase by {abs(change):.2f} units"
    
    new_data_modified = new_data_processed.copy()
    for feature, recommendation in recommendations.items():
        adjustment = float(recommendation.split()[-2])
        new_data_modified.at[0, feature] += adjustment if 'Increase' in recommendation else -adjustment

    potential_efficiency_prediction = model.predict(new_data_modified)[0]
    improvement = potential_efficiency_prediction - current_efficiency_prediction  # Calculate improvement
    
    return {
        "Current Efficiency": current_efficiency_prediction,
        "Potential Efficiency After Changes": potential_efficiency_prediction,
        "Improvement": improvement,
        "Current Cluster": current_cluster,
        "Target Cluster": target_cluster,
        "Recommendations": recommendations
    }


# Streamlit user interface setup
st.title('Sleep Efficiency Improvement Recommendations')
age = st.number_input('Age', min_value=18, max_value=100, value=30)
caffeine = st.number_input('Caffeine consumption (cups/day)', min_value=0, max_value=20, value=3)
alcohol = st.number_input('Alcohol consumption (units/day)', min_value=0, max_value=10, value=2)
exercise = st.number_input('Exercise frequency (times/week)', min_value=0, max_value=7, value=4)
steps = st.number_input('Daily Steps', min_value=1000, max_value=30000, value=8000)
gender = st.selectbox('Gender', ['Male', 'Female'])
smoking = st.selectbox('Smoking status', ['No', 'Yes'])

new_user_data = {
    'Age': age,
    'Caffeine consumption': caffeine,
    'Alcohol consumption': alcohol,
    'Exercise frequency': exercise,
    'Daily Steps': steps,
    'Gender': gender,
    'Smoking status': smoking
}

if st.button('Recommend Improvements'):
    results = recommend_improvements(new_user_data)
    st.write('Current Efficiency:', results['Current Efficiency'])
    st.write('Potential Efficiency After Changes:', results['Potential Efficiency After Changes'])
    st.write('Improvement:', results['Improvement'])
    st.write('Recommendations:', results['Recommendations'])

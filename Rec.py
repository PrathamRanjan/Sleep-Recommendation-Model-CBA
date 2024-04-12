import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

@st.cache(allow_output_mutation=True)  # Caching with mutation allowed
def load_data():
    data_path = 'sleep.csv'  # Adjust this path as needed
    return pd.read_csv(data_path)

# Load and preprocess data
sleep_data = load_data()
categorical_features = ['Gender', 'Smoking status']
numeric_features = ['Age', 'Caffeine consumption', 'Alcohol consumption', 'Exercise frequency', 'Daily Steps']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X = preprocessor.fit_transform(sleep_data)
encoded_cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
features = numeric_features + list(encoded_cat_features)
X_df = pd.DataFrame(X, columns=features)
y = sleep_data['Sleep efficiency']

# Setup and train KMeans and RandomForest models
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_df)
sleep_data['Cluster'] = clusters

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)
rf_regressor.fit(X_train, y_train)

def recommend_improvements(new_data, model, preprocessor, kmeans, data_df, cluster_centers):
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])  # Convert dict to DataFrame

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
    
    potential_efficiency_prediction = model.predict(new_data_processed + recommendations_raw)[0]
    improvement = potential_efficiency_prediction - current_efficiency_prediction
    
    return {
        "Current Efficiency": current_efficiency_prediction,
        "Potential Efficiency After Changes": potential_efficiency_prediction,
        "Improvement": improvement,
        "Current Cluster": current_cluster,
        "Target Cluster": target_cluster,
        "Recommendations": recommendations
    }

# Streamlit UI setup
st.title('Sleep Efficiency Improvement Recommendations')
with st.form("my_form"):
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    caffeine = st.number_input('Caffeine consumption before you sleep', min_value=0, max_value=20, value=3)
    alcohol = st.number_input('Alcohol consumption before you sleep', min_value=0, max_value=10, value=2)
    exercise = st.number_input('Exercise frequency (times/week)', min_value=0, max_value=7, value=4)
    steps = st.number_input('Daily Steps', min_value=1000, max_value=30000, value=8000)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    smoking = st.selectbox('Smoking status', ['No', 'Yes'])
    submitted = st.form_submit_button("Recommend Improvements")
    if submitted:
        new_user_data = {
            'Age': age,
            'Caffeine consumption': caffeine,
            'Alcohol consumption': alcohol,
            'Exercise frequency': exercise,
            'Daily Steps': steps,
            'Gender': gender,
            'Smoking status': smoking
        }
        results = recommend_improvements(new_user_data, rf_regressor, preprocessor, kmeans, sleep_data, kmeans.cluster_centers_)
        st.write('Current Efficiency:', results['Current Efficiency'])
        st.write('Potential Efficiency After Changes:', results['Potential Efficiency After Changes'])
        st.write('Improvement:', results['Improvement'])
        st.write('Recommendations:', results['Recommendations'])

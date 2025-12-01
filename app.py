import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from folium.plugins import FastMarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import streamlit.components.v1 as components

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction App")

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv('home_data.csv')
    data.drop(['id', 'date'], axis=1, inplace=True)
    data['bathrooms'] = data['bathrooms'].astype(int)
    data['floors'] = data['floors'].astype(int)
    data.rename(columns={'yr_built': 'age', 'yr_renovated': 'renovated'}, inplace=True)
    data['age'] = 2023 - data['age']
    data['renovated'] = data['renovated'].apply(lambda x: 0 if x == 0 else 1)
    for col in ['sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_lot15']:
        data[col] = data[col] / data[col].max()
    return data

data = load_data()

# Train/Test Split
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
@st.cache_resource
def train_models():
    # Linear (Polynomial) Regression
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    pred_poly = pipe.predict(X_test)

    # Ridge Regression
    ridge = Ridge(alpha=0.001)
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    return pipe, ridge, rf, pred_poly, pred_ridge, pred_rf

pipe, ridge, rf, pred_poly, pred_ridge, pred_rf = train_models()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 EDA", "🤖 Model Evaluation", "🗺️ Map", "🎯 Predict", "📁 Raw Data"])

# --- Tab 1: EDA ---
with tab1:
    st.subheader("📊 Exploratory Data Analysis")

    st.markdown("### 🔥 Price Correlation with Features")
    corr_df = data.corr()['price'].sort_values(ascending=False).reset_index()
    corr_df.columns = ['Feature', 'Correlation']
    st.bar_chart(corr_df.set_index('Feature').iloc[1:])  # Skip self-correlation

    st.markdown("### 📈 Feature vs Price (Interactive)")
    features_to_plot = ['sqft_living', 'bedrooms', 'bathrooms', 'sqft_above', 'sqft_basement']
    selected_feature = st.selectbox("Select Feature", features_to_plot)
    chart = alt.Chart(data).mark_circle(size=60).encode(
        x=selected_feature,
        y='price',
        tooltip=['price', selected_feature]
    ).interactive().properties(height=400)
    st.altair_chart(chart, use_container_width=True)

# --- Tab 2: Model Evaluation ---
with tab2:
    st.subheader("📈 Model Performance Metrics")

    results = {
        "Model": ["Linear Regression", "Ridge Regression", "Random Forest"],
        "R² Score": [
            r2_score(y_test, pred_poly),
            r2_score(y_test, pred_ridge),
            r2_score(y_test, pred_rf)
        ],
        "MAE": [
            mean_absolute_error(y_test, pred_poly),
            mean_absolute_error(y_test, pred_ridge),
            mean_absolute_error(y_test, pred_rf)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, pred_poly)),
            np.sqrt(mean_squared_error(y_test, pred_ridge)),
            np.sqrt(mean_squared_error(y_test, pred_rf))
        ]
    }

    results_df = pd.DataFrame(results)
    st.dataframe(results_df.set_index("Model"))

    st.markdown("### 📊 R² Score Comparison")
    st.bar_chart(results_df.set_index("Model")["R² Score"])

    st.markdown("### 📊 MAE Comparison")
    st.bar_chart(results_df.set_index("Model")["MAE"])

# --- Tab 3: Map ---
with tab3:
    st.subheader("🗺️ House Locations Map")
    map_data = data[['lat', 'long']].dropna()
    m = folium.Map(location=[47.5480, -121.9836], zoom_start=9)
    FastMarkerCluster(map_data.values.tolist()).add_to(m)
    components.html(m._repr_html_(), height=500)

# --- Tab 4: Prediction ---
with tab4:
    st.subheader("🎯 Predict House Price")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            bedrooms = st.slider("Bedrooms", 1, 10, 3)
            bathrooms = st.slider("Bathrooms", 1, 5, 2)
            sqft_living = st.slider("sqft_living", 500, 5000, 2000)
            sqft_lot = st.slider("sqft_lot", 1000, 15000, 10000)
            floors = st.selectbox("Floors", [1, 2, 3])
            waterfront = st.selectbox("Waterfront", [0, 1])
            view = st.selectbox("View", list(range(5)))
        with col2:
            condition = st.selectbox("Condition", list(range(1, 6)))
            grade = st.slider("Grade", 1, 13, 8)
            sqft_above = st.slider("sqft_above", 500, 4000, 2000)
            sqft_basement = st.slider("sqft_basement", 0, 2000, 0)
            year_built = st.slider("Year Built", 1900, 2023, 1990)
            renovated = st.selectbox("Renovated?", [0, 1])

        submitted = st.form_submit_button("Predict")

        if submitted:
            age = 2023 - year_built
            features = np.array([[bedrooms, bathrooms, sqft_living / 5000, sqft_lot / 15000,
                                  floors, waterfront, view, condition, grade,
                                  sqft_above / 4000, sqft_basement / 2000, age, renovated,
                                  98001, 47.5480, -121.9836,
                                  sqft_living / 5000, sqft_lot / 15000]])
            prediction = rf.predict(features)
            st.success(f"💰 Estimated House Price: ${prediction[0]:,.2f}")

# --- Tab 5: Raw Data ---
with tab5:
    st.subheader("📁 Raw Dataset (First 100 Rows)")
    st.dataframe(data.head(100))

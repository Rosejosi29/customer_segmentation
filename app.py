import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Customer Segmentation Group 19", layout="centered")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('customer_segmentation_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

kmeans, scaler = load_model()

# Title and description
st.title("🛍️ Customer Segmentation App – Group 19")
st.markdown("""
This app uses **K-Means clustering** to segment customers based on:
- **Annual Income (k$)**  
- **Spending Score (1-100)**

Enter a customer's details below to see which segment they belong to, along with business recommendations.
""")

# Input sliders
col1, col2 = st.columns(2)
with col1:
    income = st.slider("💰 Annual Income (k$)", min_value=15, max_value=140, value=60, step=1)
with col2:
    spending = st.slider("💳 Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# Prediction button
if st.button("🔮 Predict Customer Segment", type="primary"):
    # Scale the input using the same scaler
    input_data = np.array([[income, spending]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    
    # Show result
    st.success(f"### This customer belongs to **Cluster {cluster}**")
    
    # Interpretation based on project's analysis
    cluster_labels = {
        0: "💎 **Premium Spenders** – High income, high spending. Target with loyalty programs and exclusive offers.",
        1: "🛒 **Mid-Range Shoppers** – Average income, average spending. Engage with seasonal campaigns and referral rewards.",
        2: "🎯 **Discount Seekers** – Low income, high spending. Respond well to coupons, flash sales, and value bundles.",
        3: "👑 **Potential Activators** – High income, low spending. Need personalised promotions or product recommendations.",
        4: "📉 **Budget Conscious** – Low income, low spending. Focus on essential products and basic plans."
    }
    st.info(cluster_labels.get(cluster, "Explore this segment further."))
    
    # Visualisation: show all clusters and highlight the new point
    try:
        # Load original dataset to plot all points
        df = pd.read_csv("Mall_Customers.csv")
        # Predict clusters for all points to colour them
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
        X_scaled = scaler.transform(X)
        df['Cluster'] = kmeans.predict(X_scaled)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = sns.scatterplot(
            data=df, 
            x='Annual Income (k$)', 
            y='Spending Score (1-100)',
            hue='Cluster', 
            palette='viridis', 
            alpha=0.6,
            s=80,
            ax=ax
        )
        # Highlight the new customer point
        ax.scatter(income, spending, color='red', s=200, edgecolor='black', 
                   marker='X', label='Your Customer', zorder=5)
        ax.set_title("Customer Segments (Clusters)")
        ax.legend()
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("⚠️ Dataset 'Mall_Customers.csv' not found. Can't show the full cluster plot, but prediction is still valid.")
    
    # Show centroid coordinates for the cluster
    centroids = kmeans.cluster_centers_
    st.caption(f"*Centroid of Cluster {cluster}: Income ≈ {centroids[cluster][0]*scaler.scale_[0] + scaler.mean_[0]:.1f}k$, Spending ≈ {centroids[cluster][1]*scaler.scale_[1] + scaler.mean_[1]:.1f}*")

# Sidebar with team info
st.sidebar.header("👥 Group 19 Members")
st.sidebar.markdown("""
- RUTH  
- TUNDE  
- MUSA  
- ROSELINE  
- MELODY  
- MARGARET  
- PROMISE  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("📅 **Date:** 4th June 2026")
st.sidebar.markdown("📘 **Course:** AI/Machine Learning – Capstone Project")
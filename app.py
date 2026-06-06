import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set page config (must be first)
st.set_page_config(page_title="Customer Segmentation – Group 19", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for advanced styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    /* Metric card styling */
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .metric-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    /* Success message */
    .stSuccess {
        border-left-color: #4CAF50;
    }
    /* Expander headers */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler with caching
@st.cache_resource
def load_models():
    model = joblib.load('customer_segmentation_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

kmeans, scaler = load_models()

# Cluster definitions with detailed marketing profiles
CLUSTER_PROFILES = {
    0: {
        "name": "💰 Premium Spenders",
        "color": "#2E86C1",
        "summary": "High income, high spending – your most valuable segment.",
        "behavior": "Luxury oriented, brand loyal, seek exclusive experiences.",
        "marketing": "VIP loyalty programs, early access, premium bundles, personal shoppers."
    },
    1: {
        "name": "🛒 Mid-Range Shoppers",
        "color": "#28B463",
        "summary": "Average income, average spending – stable and large base.",
        "behavior": "Value seekers, compare options, respond to discounts.",
        "marketing": "Seasonal campaigns, referral rewards, email promotions, mid-tier products."
    },
    2: {
        "name": "🏷️ Discount Seekers",
        "color": "#E67E22",
        "summary": "Low income, high spending – deal hunters.",
        "behavior": "Price sensitive, impulse buyers when discounts are high.",
        "marketing": "Flash sales, BOGO offers, coupon codes, loyalty points."
    },
    3: {
        "name": "⭐ Potential Activators",
        "color": "#8E44AD",
        "summary": "High income, low spending – untapped potential.",
        "behavior": "Not yet engaged, need personalised incentives.",
        "marketing": "Personalised recommendations, first-purchase discounts, free samples."
    },
    4: {
        "name": "📉 Budget Conscious",
        "color": "#E74C3C",
        "summary": "Low income, low spending – essential buyers.",
        "behavior": "Need-based, price critical, trust oriented.",
        "marketing": "Bulk discounts, subscription for staples, basic product lines."
    }
}

def predict_batch(data_df):
    """Predict clusters for a dataframe containing 'Annual Income (k$)' and 'Spending Score (1-100)'"""
    X = data_df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    X_scaled = scaler.transform(X)
    clusters = kmeans.predict(X_scaled)
    return clusters

def add_cluster_details(df, clusters):
    """Add detailed columns to dataframe"""
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters
    df_with_clusters['Segment'] = df_with_clusters['Cluster'].map(lambda c: CLUSTER_PROFILES[c]["name"])
    df_with_clusters['Summary'] = df_with_clusters['Cluster'].map(lambda c: CLUSTER_PROFILES[c]["summary"])
    df_with_clusters['Behaviour'] = df_with_clusters['Cluster'].map(lambda c: CLUSTER_PROFILES[c]["behavior"])
    df_with_clusters['Target_Marketing'] = df_with_clusters['Cluster'].map(lambda c: CLUSTER_PROFILES[c]["marketing"])
    return df_with_clusters

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4290/4290854.png", width=80)
    st.title("📊 Controls")
    input_method = st.radio("Choose input method:", ["📝 Single Customer Entry", "📂 Batch CSV Upload"])
    st.markdown("---")
    st.markdown("### 📌 About")
    st.info("This tool segments customers using K-Means clustering (income + spending). Get detailed marketing strategies for each segment.")
    st.markdown("---")
    st.markdown("**Group 19 Members**")
    st.markdown("RUTH, TUNDE, MUSA, ROSELINE, MELODY, MARGARET, PROMISE")
    st.caption("Capstone Project – June 2026")

# Main area
st.title("🎯 Advanced Customer Segmentation Engine")
st.markdown("""
    Upload or enter customer data to receive **personalised marketing strategies** based on income and spending patterns.
    The model assigns each customer to one of five segments, each with a tailored target market structure.
""")

# Single Customer Entry
if input_method == "📝 Single Customer Entry":
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("💰 Annual Income (k$)", min_value=0.0, max_value=200.0, value=60.0, step=1.0)
    with col2:
        spending = st.number_input("💳 Spending Score (1-100)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    
    if st.button("🔮 Predict Segment", use_container_width=True):
        input_df = pd.DataFrame([[income, spending]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
        clusters = predict_batch(input_df)
        enriched = add_cluster_details(input_df, clusters)
        cluster = clusters[0]
        profile = CLUSTER_PROFILES[cluster]
        
        # Display results in cards
        st.success(f"### 🎯 Customer assigned to **{profile['name']}** (Cluster {cluster})")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Annual Income", f"${income}k")
        with col_b:
            st.metric("Spending Score", f"{spending}/100")
        with col_c:
            st.metric("Segment Cluster", cluster)
        
        st.markdown("---")
        st.subheader("📋 Detailed Customer Profile")
        st.markdown(f"**Summary:** {profile['summary']}")
        st.markdown(f"**Spending Behaviour:** {profile['behavior']}")
        st.markdown(f"**🎯 Recommended Marketing Strategy:** {profile['marketing']}")
        
        # Show centroid info
        centroid = kmeans.cluster_centers_[cluster]
        orig_centroid_income = centroid[0] * scaler.scale_[0] + scaler.mean_[0]
        orig_centroid_spending = centroid[1] * scaler.scale_[1] + scaler.mean_[1]
        st.caption(f"📊 Segment centroid: Income ≈ {orig_centroid_income:.1f}k$, Spending ≈ {orig_centroid_spending:.1f}")

# Batch CSV Upload
else:
    st.markdown("### 📂 Upload Customer Data (CSV)")
    st.markdown("Your CSV must contain columns: **`Annual Income (k$)`** and **`Spending Score (1-100)`**")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Check required columns
        required = ['Annual Income (k$)', 'Spending Score (1-100)']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}. Please ensure your CSV has {required}")
        else:
            # Show preview
            st.subheader("🔍 Data Preview (first 5 rows)")
            st.dataframe(df.head(), use_container_width=True)
            
            # Run prediction
            with st.spinner("Segmenting customers..."):
                clusters = predict_batch(df)
                enriched_df = add_cluster_details(df, clusters)
            
            # Display summary metrics
            st.subheader("📊 Segmentation Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card"><span class="metric-number">{}</span><br>Total Customers</div>'.format(len(enriched_df)), unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card"><span class="metric-number">{}</span><br>Unique Segments</div>'.format(enriched_df['Cluster'].nunique()), unsafe_allow_html=True)
            with col3:
                top_segment = enriched_df['Segment'].mode()[0] if not enriched_df.empty else "N/A"
                st.markdown('<div class="metric-card"><span class="metric-number">{}</span><br>Most Common Segment</div>'.format(top_segment[:15]), unsafe_allow_html=True)
            with col4:
                avg_income = enriched_df['Annual Income (k$)'].mean()
                st.markdown('<div class="metric-card"><span class="metric-number">${:.0f}k</span><br>Avg Income</div>'.format(avg_income), unsafe_allow_html=True)
            
            # Cluster distribution chart (bar chart with seaborn)
            st.subheader("📈 Customer Distribution Across Clusters")
            cluster_counts = enriched_df['Cluster'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = [CLUSTER_PROFILES[i]['color'] for i in cluster_counts.index]
            bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of Customers')
            ax.set_title('How customers are spread across segments')
            for bar, count in zip(bars, cluster_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()
            
            # Display full results table
            st.subheader("📋 Customer Segmentation Results")
            st.dataframe(enriched_df, use_container_width=True, height=400)
            
            # Download button
            csv_output = enriched_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Enriched CSV",
                data=csv_output,
                file_name="segmented_customers.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Optional: Show a few detailed profiles for the most common cluster
            most_common_cluster = enriched_df['Cluster'].mode()[0]
            st.subheader(f"🎯 Spotlight: {CLUSTER_PROFILES[most_common_cluster]['name']} (Cluster {most_common_cluster})")
            st.markdown(f"**Summary:** {CLUSTER_PROFILES[most_common_cluster]['summary']}")
            st.markdown(f"**Recommended Marketing Strategy:** {CLUSTER_PROFILES[most_common_cluster]['marketing']}")

# Footer
st.markdown("---")
st.caption("🔍 Customer Segmentation using K-Means Clustering | Group 19 Capstone Project")

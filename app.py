import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Page config
st.set_page_config(page_title="Customer Segmentation – Group 19", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for cleaner layout
st.markdown("""
<style>
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .recommendation-card {
        background-color: #f9f9fb;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .metric-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_models():
    model = joblib.load('customer_segmentation_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

kmeans, scaler = load_models()

# Cluster profiles with clean, structured recommendations
CLUSTER_PROFILES = {
    0: {
        "name": "💰 Premium Spenders",
        "color": "#2E86C1",
        "summary": "High income, high spending – your most valuable segment.",
        "behavior_points": [
            "Luxury oriented and brand loyal",
            "Seek exclusive experiences",
            "Willing to pay premium for quality"
        ],
        "marketing_points": [
            "VIP loyalty programs with tiered rewards",
            "Early access to new collections",
            "Personal shopper services",
            "Invite‑only events"
        ]
    },
    1: {
        "name": "🛒 Mid-Range Shoppers",
        "color": "#28B463",
        "summary": "Average income, average spending – stable and large base.",
        "behavior_points": [
            "Value seekers – compare prices",
            "Respond to moderate discounts",
            "Trust peer reviews and referrals"
        ],
        "marketing_points": [
            "Seasonal campaigns (e.g., summer sale, Black Friday)",
            "Referral rewards (give $10, get $10)",
            "Email newsletters with curated mid‑range products",
            "Loyalty points on every purchase"
        ]
    },
    2: {
        "name": "🏷️ Discount Seekers",
        "color": "#E67E22",
        "summary": "Low income, high spending – deal hunters.",
        "behavior_points": [
            "Highly price sensitive",
            "Impulse buyers when discounts are steep",
            "Frequently use coupons and promo codes"
        ],
        "marketing_points": [
            "Flash sales (24‑48 hour deep discounts)",
            "BOGO (buy one, get one free) offers",
            "Push notifications for clearance items",
            "Loyalty points multiplier on discounted items"
        ]
    },
    3: {
        "name": "⭐ Potential Activators",
        "color": "#8E44AD",
        "summary": "High income, low spending – untapped potential.",
        "behavior_points": [
            "Not yet engaged – may be new customers",
            "Need personalised incentives to spend",
            "Respond to education and product showcases"
        ],
        "marketing_points": [
            "Personalised product recommendations (based on browsing)",
            "First‑purchase discount (e.g., 20% off)",
            "Free samples or trial periods",
            "Educational content about product benefits"
        ]
    },
    4: {
        "name": "📉 Budget Conscious",
        "color": "#E74C3C",
        "summary": "Low income, low spending – essential buyers.",
        "behavior_points": [
            "Need‑based purchasing only",
            "Price is the main decision factor",
            "Trust and reliability over flashy marketing"
        ],
        "marketing_points": [
            "Bulk discounts for everyday essentials",
            "Subscription plans for staple products (save 10%)",
            "Community loyalty programs (points for repeat purchases)",
            "Clear, honest pricing with no hidden fees"
        ]
    }
}

def predict_batch(data_df):
    X = data_df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    X_scaled = scaler.transform(X)
    return kmeans.predict(X_scaled)

def enrich_dataframe(df, clusters):
    df_out = df.copy()
    df_out['Cluster'] = clusters
    df_out['Segment'] = df_out['Cluster'].map(lambda c: CLUSTER_PROFILES[c]["name"])
    df_out['Summary'] = df_out['Cluster'].map(lambda c: CLUSTER_PROFILES[c]["summary"])
    # For detailed recommendations we'll use the points directly in display
    return df_out

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4290/4290854.png", width=70)
    st.title("⚙️ Controls")
    input_mode = st.radio(
        "Choose input method:",
        ["✏️ Single Customer", "📝 Batch (Manual Entry)", "📂 Batch (CSV Upload)"]
    )
    st.markdown("---")
    st.markdown("### 👥 Group 19 Members")
    st.write("RUTH, TUNDE, MUSA, ROSELINE, MELODY, MARGARET, PROMISE")
    st.caption("Capstone Project – June 2026")

# Main title
st.title("🎯 Customer Segmentation & Marketing Engine")
st.markdown("Group customers based on income & spending, then receive **targeted marketing strategies**.")

# ==========================================
# MODE 1: SINGLE CUSTOMER
# ==========================================
if input_mode == "✏️ Single Customer":
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("💰 Annual Income (k$)", min_value=0.0, max_value=200.0, value=60.0, step=1.0)
    with col2:
        spending = st.number_input("💳 Spending Score (1-100)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    
    if st.button("🔮 Predict Segment", use_container_width=True):
        input_df = pd.DataFrame([[income, spending]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
        clusters = predict_batch(input_df)
        cluster = clusters[0]
        profile = CLUSTER_PROFILES[cluster]
        
        st.success(f"### 🎯 Customer Segment: {profile['name']} (Cluster {cluster})")
        
        # Metrics row
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Income", f"${income}k")
        col_b.metric("Spending Score", f"{spending}/100")
        col_c.metric("Segment", cluster)
        
        st.markdown("---")
        st.subheader("📌 Segment Summary")
        st.info(profile['summary'])
        
        st.subheader("🛒 Spending Behaviour")
        for point in profile['behavior_points']:
            st.markdown(f"- {point}")
        
        st.subheader("🎯 Recommended Marketing Strategy")
        for point in profile['marketing_points']:
            st.markdown(f"- {point}")
        
        # Centroid comparison
        centroid = kmeans.cluster_centers_[cluster]
        orig_income = centroid[0] * scaler.scale_[0] + scaler.mean_[0]
        orig_spending = centroid[1] * scaler.scale_[1] + scaler.mean_[1]
        st.caption(f"📊 Segment centre: Income ≈ {orig_income:.1f}k$, Spending ≈ {orig_spending:.1f}")

# ==========================================
# MODE 2: BATCH MANUAL (TEXT AREA)
# ==========================================
elif input_mode == "📝 Batch (Manual Entry)":
    st.markdown("### Enter multiple customers (one per line)")
    st.markdown("Format each line as: `Income, Spending`  (e.g., `60, 50`)")
    batch_text = st.text_area("Customer list", height=200, placeholder="60, 50\n90, 85\n25, 80\n88, 15\n26, 20")
    
    if st.button("🔮 Process Batch", use_container_width=True):
        if not batch_text.strip():
            st.error("Please enter at least one customer.")
        else:
            lines = batch_text.strip().split('\n')
            customers = []
            errors = []
            for idx, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split(',')
                    if len(parts) != 2:
                        errors.append(f"Line {idx}: '{line}' – use exactly two numbers separated by comma")
                        continue
                    inc = float(parts[0].strip())
                    spend = float(parts[1].strip())
                    if inc < 0 or inc > 200:
                        errors.append(f"Line {idx}: Income {inc} should be between 0 and 200")
                    if spend < 0 or spend > 100:
                        errors.append(f"Line {idx}: Spending {spend} should be between 0 and 100")
                    customers.append((inc, spend))
                except ValueError:
                    errors.append(f"Line {idx}: '{line}' – invalid numbers")
            
            if errors:
                for err in errors:
                    st.warning(err)
            
            if customers:
                df = pd.DataFrame(customers, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
                clusters = predict_batch(df)
                enriched = enrich_dataframe(df, clusters)
                
                # Summary metrics
                st.success(f"Processed {len(customers)} customers")
                col1, col2 = st.columns(2)
                col1.metric("Unique Segments", enriched['Cluster'].nunique())
                col2.metric("Most Common Segment", enriched['Segment'].mode()[0] if not enriched.empty else "N/A")
                
                # Show results table
                st.subheader("📋 Segmentation Results")
                st.dataframe(enriched[['Annual Income (k$)', 'Spending Score (1-100)', 'Segment', 'Summary']], use_container_width=True)
                
                # Download button
                csv = enriched.to_csv(index=False)
                st.download_button("💾 Download Results CSV", csv, "segmentation_results.csv", "text/csv")
                
                # For each customer, show expandable details (clean)
                st.subheader("🔍 Detailed Recommendations (click to expand)")
                for idx, row in enriched.iterrows():
                    with st.expander(f"Customer {idx+1}: Income ${row['Annual Income (k$)']}k, Spending {row['Spending Score (1-100)']}"):
                        profile = CLUSTER_PROFILES[row['Cluster']]
                        st.markdown(f"**Segment:** {profile['name']}")
                        st.markdown(f"**Summary:** {profile['summary']}")
                        st.markdown("**Behaviour:**")
                        for bp in profile['behavior_points']:
                            st.markdown(f"- {bp}")
                        st.markdown("**Marketing Strategy:**")
                        for mp in profile['marketing_points']:
                            st.markdown(f"- {mp}")

# ==========================================
# MODE 3: CSV UPLOAD
# ==========================================
else:
    st.markdown("### Upload CSV file")
    st.markdown("Required columns: **`Annual Income (k$)`** and **`Spending Score (1-100)`**")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        required = ['Annual Income (k$)', 'Spending Score (1-100)']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.subheader("Preview (first 5 rows)")
            st.dataframe(df.head(), use_container_width=True)
            
            with st.spinner("Segmenting customers..."):
                clusters = predict_batch(df)
                enriched = enrich_dataframe(df, clusters)
            
            st.success("Segmentation complete!")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", len(enriched))
            col2.metric("Unique Segments", enriched['Cluster'].nunique())
            col3.metric("Most Common Segment", enriched['Segment'].mode()[0] if not enriched.empty else "N/A")
            
            # Cluster distribution chart
            st.subheader("📊 Segment Distribution")
            counts = enriched['Cluster'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = [CLUSTER_PROFILES[i]['color'] for i in counts.index]
            bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of Customers')
            ax.set_title('How customers are distributed across segments')
            for bar, count in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()
            
            # Full table
            st.subheader("📋 Detailed Results (first 100 rows)")
            st.dataframe(enriched[['Annual Income (k$)', 'Spending Score (1-100)', 'Segment', 'Summary']].head(100), use_container_width=True)
            
            # Download
            csv_data = enriched.to_csv(index=False)
            st.download_button("💾 Download Full Results CSV", csv_data, "segmented_customers.csv", "text/csv")
            
            # Optional: show top segment's marketing strategy
            top_cluster = enriched['Cluster'].mode()[0]
            st.subheader(f"🎯 Spotlight on {CLUSTER_PROFILES[top_cluster]['name']}")
            st.markdown(f"**Summary:** {CLUSTER_PROFILES[top_cluster]['summary']}")
            st.markdown("**Recommended Marketing Actions:**")
            for mp in CLUSTER_PROFILES[top_cluster]['marketing_points']:
                st.markdown(f"- {mp}")

# Footer
st.markdown("---")
st.caption("Powered by K‑Means Clustering | Group 19 Capstone Project – Customer Segmentation")

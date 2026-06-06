import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
This app helps businesses understand their customers by grouping them into segments
based on their **Annual Income (k$)** and **Spending Score (1-100)**.

Enter multiple customers (one per line) below to see their cluster and a detailed marketing profile.
""")

# Batch input: text area for multiple customers
st.subheader("📋 Enter Customer Data")
st.markdown("Format each line as: `Income, Spending` (e.g., `60, 50`)")

input_text = st.text_area(
    "Customer list (one per line)",
    height=150,
    placeholder="60, 50\n90, 85\n25, 80\n88, 15\n26, 20"
)

# Prediction button
if st.button("🔮 Predict Customer Segments", type="primary"):
    if not input_text.strip():
        st.error("Please enter at least one customer in the text area.")
    else:
        # Parse input lines
        lines = input_text.strip().split('\n')
        customers = []
        valid_lines = []
        errors = []
        
        for idx, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(',')
                if len(parts) != 2:
                    errors.append(f"Line {idx}: '{line}' - Expected two values separated by comma")
                    continue
                income = float(parts[0].strip())
                spending = float(parts[1].strip())
                # Validate ranges
                if income < 15 or income > 140:
                    errors.append(f"Line {idx}: Income {income} is outside valid range (15-140)")
                if spending < 1 or spending > 100:
                    errors.append(f"Line {idx}: Spending {spending} is outside valid range (1-100)")
                customers.append((income, spending))
                valid_lines.append(idx)
            except ValueError:
                errors.append(f"Line {idx}: '{line}' - Invalid numbers")
        
        if errors:
            for err in errors:
                st.warning(err)
        
        if customers:
            # Prepare batch prediction
            input_array = np.array(customers)
            input_scaled = scaler.transform(input_array)
            clusters = kmeans.predict(input_scaled)
            
            # Cluster descriptions with detailed summaries
            cluster_details = {
                0: {
                    "name": "Premium Spenders",
                    "summary": "High income, high spending. These customers are your most valuable segment. They have disposable income and enjoy spending on quality products.",
                    "behavior": "They respond well to premium products, luxury offers, and exclusive experiences. They are brand-loyal if treated well.",
                    "target_market": "Loyalty programs, VIP events, early access to new collections, premium bundles, and personalised concierge services."
                },
                1: {
                    "name": "Mid-Range Shoppers",
                    "summary": "Average income, average spending. This is your largest and most stable customer base.",
                    "behavior": "They seek value for money. They are practical shoppers who compare options before purchasing.",
                    "target_market": "Seasonal campaigns, referral rewards, cashback offers, and mid-tier product bundles. Engage with email newsletters and social media promotions."
                },
                2: {
                    "name": "Discount Seekers",
                    "summary": "Low income, high spending. They spend a significant portion of their income on shopping, often driven by deals.",
                    "behavior": "Highly price-sensitive. They are attracted to discounts, flash sales, coupons, and value packs.",
                    "target_market": "Frequent discount alerts, BOGO (buy one get one) offers, loyalty points on every purchase, and budget-friendly product lines."
                },
                3: {
                    "name": "Potential Activators",
                    "summary": "High income, low spending. They have the means but are not yet engaged to spend.",
                    "behavior": "They may be new customers, or they haven't found products that excite them. They need persuasion.",
                    "target_market": "Personalised product recommendations, first-purchase discounts, free samples, and educational content about product benefits. Re-engagement campaigns."
                },
                4: {
                    "name": "Budget Conscious",
                    "summary": "Low income, low spending. They are very careful with their money and only buy essentials.",
                    "behavior": "They prioritise needs over wants. Price is the main decision factor.",
                    "target_market": "Essential products at lowest prices, bulk discounts, subscription for staples, and community loyalty programs. Focus on trust and reliability."
                }
            }
            
            # Prepare results table
            results_data = []
            for i, (income, spending) in enumerate(customers):
                cluster = clusters[i]
                details = cluster_details[cluster]
                results_data.append({
                    "Income (k$)": income,
                    "Spending Score": spending,
                    "Cluster": cluster,
                    "Segment Name": details["name"],
                    "Summary": details["summary"],
                    "Spending Behaviour": details["behavior"],
                    "Target Market Strategy": details["target_market"]
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Display results
            st.success(f"✅ Processed {len(customers)} customer(s)")
            st.subheader("📊 Customer Segmentation Results")
            
            # Show table
            st.dataframe(results_df, use_container_width=True)
            
            # Optional: Show detailed breakdown per customer (expandable)
            for idx, row in results_df.iterrows():
                with st.expander(f"Customer {idx+1}: Income {row['Income (k$)']}k$, Spending {row['Spending Score']}"):
                    st.markdown(f"**Segment:** {row['Segment Name']} (Cluster {row['Cluster']})")
                    st.markdown(f"**📌 Summary:** {row['Summary']}")
                    st.markdown(f"**🛒 Spending Behaviour:** {row['Spending Behaviour']}")
                    st.markdown(f"**🎯 Target Market Structure:** {row['Target Market Strategy']}")
            
            # Add download button for results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="customer_segmentation_results.csv",
                mime="text/csv"
            )
        else:
            st.error("No valid customers to process. Please check the format and ranges.")

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
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ How to use")
st.sidebar.markdown("""
1. Enter one customer per line: `income, spending`
2. Click **Predict Customer Segments**
3. View the results table and detailed profiles
4. Download results as CSV if needed
""")

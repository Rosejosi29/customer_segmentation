# Customer Segmentation — K‑Means Clustering (Group 19)

A group capstone project: apply K‑Means clustering to segment mall customers and deploy an interactive predictor with Streamlit.

##  Group Members
- RUTH
- TUNDE
- MUSA
- ROSELINE
- MELODY
- MARGARET
- PROMISE

## 1. Dataset

**Mall Customers dataset** from Kaggle:  
[https://www.kaggle.com/datasets/shwetabh123/mall-customers](https://www.kaggle.com/datasets/shwetabh123/mall-customers)

Download `Mall_Customers.csv` and place it in this folder.  
Columns: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`

For clustering we use only **Annual Income (k$)** and **Spending Score (1-100)**.

## 2. Setup

bash
pip install -r requirements.txt

## Contents of requirements.txt:

pandas
matplotlib
seaborn
scikit-learn
joblib
streamlit

## 3. Train the model

bash
python train.py
This script:

Loads and explores the data.

Selects the two features.

Scales them using StandardScaler.

Uses the Elbow Method to find the optimal number of clusters (K=5).

Trains a K‑Means model with 5 clusters.

Saves the model and scaler as:
customer_segmentation_model.pkl
scaler.pkl

Output example – cluster means:
         Annual Income (k$)  Spending Score (1-100)
Cluster                                            
0                 55.30                49.52
1                 86.54                82.13
2                 25.73                79.36
3                 88.20                17.11
4                 26.30                20.91

## 4. Run the app locally

Streamlit (web interface):

bash
streamlit run app.py
The app opens in your browser. It lets you:

Slide to choose annual income and spending score.

Predict which customer segment (cluster) the customer belongs to.

View business recommendations and a scatter plot of all clusters.

## 5. Deploy 
Streamlit Community Cloud (free)
Push this folder to a public GitHub repository.

Push this folder to a GitHub repo.
Go to share.streamlit.io → New app → pick repo → main file app.py → Deploy.

## Results & Business Recommendations
The model identified five customer segments:

Segment	Characteristics	Recommendation
Cluster 0	Average income & spending	Seasonal campaigns, referral programs
Cluster 1	High income, high spending	Loyalty programs, premium offers
Cluster 2	Low income, high spending	Discounts, coupons, value bundles
Cluster 3	High income, low spending	Personalised promotions, product recs
Cluster 4	Low income, low spending	Budget‑focused plans, essential products

These insights enable targeted marketing strategies to improve customer satisfaction and sales.

Group 19 – Capstone Machine Learning Project
Course: AI/Machine Learning | Date: 4th June 2026

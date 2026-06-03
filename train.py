import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

#Loading and Cleaning the Dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
df.shape
df.info()

#Selecting Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Elbow Method to Determine Optimal Number of Clusters
inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#Training the K-Means Model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster']

#centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
  s=300, c='black', marker='X', label='Centroids')

#Visualizing the Clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
   hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Customer Segmentation Result')
plt.show()

#Analyzing Cluster Characteristics
print(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())

#Saving the Model
joblib.dump(kmeans, 'customer_segmentation_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(" Model and scaler saved as 'customer_segmentation_model.pkl' and 'scaler.pkl'")
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset
def load_data(file_path):
    # Load the customer data from a CSV file (update the file path)
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return data

# Preprocess the data
def preprocess_data(data):
    # Handle missing values, if any (e.g., remove rows with missing values)
    data = data.dropna()
    
    # Select numerical columns for clustering (update this based on your dataset)
    numerical_data = data.select_dtypes(include=[np.number])

    # Feature scaling: Standardize the features to have zero mean and unit variance
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_data)
    
    print(f"Preprocessing complete: {scaled_data.shape} scaled features.")
    return scaled_data

# Determine the optimal number of clusters (Elbow Method)
def optimal_k(X):
    # We will try different values of K (1 to 10 clusters) and calculate the inertia
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()

# Perform K-Means clustering
def perform_kmeans(X, k):
    # Fit KMeans with the specified number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    # Get the cluster labels
    labels = kmeans.labels_
    
    # Add the labels to the original data
    return labels, kmeans

# Visualize the clusters (using PCA for dimensionality reduction)
def visualize_clusters(X, labels):
    # Use PCA to reduce the data to 2D for visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X)
    
    # Plot the clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=labels, palette='Set2', s=100, alpha=0.6)
    plt.title("Customer Segments")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster', loc='upper right')
    plt.show()

# Evaluate the model (Silhouette Score)
def evaluate_clustering(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
# Main function to run the whole pipeline
def main():
    # Load the dataset (update this with the actual dataset file path)
    file_path = "customer_data.csv"  # Update with your file path
    data = load_data(file_path)
    
    # Preprocess the data
    scaled_data = preprocess_data(data)
    
    # Find the optimal number of clusters using the Elbow method
    optimal_k(scaled_data)
    
    # Ask the user for the number of clusters (from the elbow plot or manually decided)
    k = int(input("Enter the optimal number of clusters (based on the elbow method): "))
    
    # Perform KMeans clustering
    labels, kmeans = perform_kmeans(scaled_data, k)
    
    # Add the cluster labels to the original data for reference
    data['Cluster'] = labels
    
    # Show the first few rows of the resulting data with cluster labels
    print("First few rows of clustered data:")
    print(data.head())

    # Visualize the clusters
    visualize_clusters(scaled_data, labels)
    
    # Evaluate the clustering performance using the silhouette score
    evaluate_clustering(scaled_data, labels)
    
    # Optionally, save the results to a new CSV file
    data.to_csv("segmented_customers.csv", index=False)
    print("Clustered data saved to 'segmented_customers.csv'.")

# Entry point of the script
if __name__ == "__main__":
    main()

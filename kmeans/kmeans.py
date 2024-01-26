import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
import threading


# Load the dataset
file_path = 'Netflix_User_Ratings_Cleaned.csv'
data = pd.read_csv(file_path)
print("finished with dataframe")

# Prepare the data for K-means
pivot_table = data.pivot_table(values='Rating', index='CustId', columns='MovieId').fillna(-1)
print("finished with pivot table")
original_rows, original_cols = pivot_table.shape
print(f"Original Pivot Table - Rows: {original_rows}, Columns: {original_cols}")

pivot_array = pivot_table.values
clusters = 1000
sse = {}
sill = {}
threads = []

def run_kmeans(clusters):
    print(f"Starting fitting with {clusters}")
    kmeans2 = KMeans(n_clusters=clusters)

    # Fit K-means to the data
    kmeans2.fit(pivot_array)
    print(kmeans2.inertia_)
    sse[clusters] = kmeans2.inertia_
    sill[clusters] =  silhouette_score(pivot_array, kmeans2.labels_)
    print(sill[clusters])
    print(f"Done with fitting {clusters} clusters")

while clusters <= 50000:
    thread = threading.Thread(target=run_kmeans, args=(clusters,))
    threads.append(thread)
    thread.start()
    clusters *= 2

# Join all threads to ensure completion
for thread in threads:
    thread.join()
    
    
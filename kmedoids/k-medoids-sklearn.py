import pandas as pd
import math
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples

# Pivot table for Rating and Date combined

# Converting the data merged Rating & Date columns format using pivot_table
df = pd.read_csv(
    '/Dataset/Netflix_User_Ratings_Normalized.csv')

# Combine Rating and Date columns into merged column to generate unique rows by customer id
df['Rating-Date'] = df['Rating'].astype(str) + '@' + df['Date']
df.drop('Rating', axis=1, inplace=True)
df.drop('Date', axis=1, inplace=True)

pivot_table_result = pd.pivot_table(df, values=['Rating-Date'], index='CustId', columns='MovieId',
                                    aggfunc={'Rating-Date': 'first'})

# Resetting index for a cleaner output
pivot_table_result.reset_index(inplace=True)

# Renaming columns for a cleaner output
pivot_table_result.columns.name = None

# Flattening the pivot table from multiple level column format to single level
pivot_table_result.columns = [f'{col[0]}_{col[1]}' for col in pivot_table_result.columns]

print("Pivot table\n: ", pivot_table_result.head(5))

# Running K Medoids

df = pivot_table_result
df.set_index('CustId_')  # For efficient querying

customer_data = df[['CustId_']]

# Row to customer ID dictionary for efficient retrievals
custid_to_row = {row['CustId_']: row for _, row in df.iterrows()}


# A custom distance metric function (inspiration:Penalized Euclidean Distance function)
def custom_distance(point1, point2):
    row1 = custid_to_row.get(point1[0])
    row2 = custid_to_row.get(point2[0])

    # Handle missing rows
    if row1 is None or row2 is None:
        return float('inf')

    sum = 0
    c = 0.0001
    for i, j in zip(row1, row2):
        if pd.notna(i) and pd.notna(j) and isinstance(i, str) and isinstance(j, str):
            # check to eliminate Nan and non-string values
            # i -> -1.11@2005-03-07; rating -> -1.11; date -> 2005-03-07
            split_str1 = i.split('@')
            split_str2 = j.split('@')

            rating1 = float(split_str1[0])
            rating2 = float(split_str2[0])

            date_obj1 = datetime.strptime(split_str1[1], "%Y-%m-%d")
            date_obj2 = datetime.strptime(split_str2[1], "%Y-%m-%d")

            time_difference = abs(date_obj2 - date_obj1)
            total_days = time_difference.days

            # total_days acts as a time penalty
            # c is added to handle false equidistant measures
            sum += ((((rating1 - rating2) + c) ** 2) * (total_days + c))

    distance = math.sqrt(sum)
    return distance


# K is passed as number of clusters; here K = 50
kmedoids = KMedoids(n_clusters=50, random_state=0, metric=custom_distance).fit(customer_data)

labels = kmedoids.labels_
medoid_indices = kmedoids.medoid_indices_
inertia = kmedoids.inertia_

# Print the results
print("Cluster labels:", labels)
print("Medoid indices:", medoid_indices)
print("Inertia: ", inertia)

# Silhouette Score

distance_matrix = pairwise_distances(customer_data, metric=custom_distance)
np.fill_diagonal(distance_matrix, 0)

silhouette_vals = silhouette_samples(distance_matrix, labels, metric='precomputed')

# Computing the overall silhouette score
silhouette_avg = np.mean(silhouette_vals)

print(f"Silhouette Score: {silhouette_avg}")
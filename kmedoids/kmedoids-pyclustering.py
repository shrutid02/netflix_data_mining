# Abandoned the implementation of K-Medoids using py-clustering library due to lack of support
# for inertia and assigned cluster labels, custom written evaluation methods takes even longer time for execution
# other libraries like sklearn.extra calculates SSE during K Medoids calculation itself, saving additional computation

# Install:- pip install pyclustering

from pyclustering.cluster.kmedoids import kmedoids

import pandas as pd
import math
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_distances
import random

# Converting the data merged Rating & Date columns format using pivot_table
df = pd.read_csv('/Dataset/Netflix_User_Ratings_Normalized.csv')

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


# Running K Medoids - pyclustering

df = pivot_table_result

customer_data = df[['CustId_']].values.tolist()
custid_to_row = {row['CustId_']: row for _, row in df.iterrows()}


# Custom distance metric function
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


def kmedoids_sse(data, medoids):
    distances = pairwise_distances(data, metric=custom_distance)

    sse = 0.0
    for i in range(len(data)):
        closest_medoid = min(medoids, key=lambda j: distances[i, j])
        sse += distances[i, closest_medoid]

    return sse


# Number of clusters
k = 50

# Initial medoid indices (they can also be initialized them randomly)
initial_medoids = random.sample(range(0, k + 1), k)

# K-Medoids instance with custom distance metric
kmedoids_instance = kmedoids(customer_data, initial_medoids, tolerance=0.25, ccore=False, metric=custom_distance)
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()
flat_clusters = [i for cluster in clusters for i in cluster]

print(medoids)
print(clusters)
print(flat_clusters)

sse = kmedoids_sse(customer_data, medoids)
print(sse)
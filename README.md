# Netflix User Pattern Mining

The project was done as part of the CSC 522 coursework. [Final project report link](https://github.com/shrutid02/netflix_data_mining/files/14067948/CSC_522_Final_Report.pdf).

## 💡 Project Overview
The project revolves around using the Netflix prize dataset to glean insights about users and their ratings. 


One approach is to use clustering algorithms to group movies based on their rating profiles and cluster users based on their rating behaviors. Clustering algorithms such as K-means, K-medoids, co-clustering, and others were used throughout the years in the Netflix data set competition. Clustering users based on movie ratings could enhance Netflix’s recommendation system for more personalized content suggestions. 

However, through our research, we haven’t found any papers written about **how time difference in rating factors into clustering accuracy**. We found that strange as a user’s preferences for movies could change over multiple years. Through our project, we plan on **experimenting with time difference in clustering and identify if time difference could impact clustering accuracy.**

## 📈 Dataset 
We are using the [Netflix Prize dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) available on Kaggle. This dataset consists of four .txt files of half GB that contain data about movieId, customerId, date, and the ratings customers have given to that movie. There are multiple data groups for each movie. The first line of each data group of a movie starts with the movie ID followed by a colon. Each subsequent line corresponds to a rating from a customer and a date on which the rating is given in the following format: CustomerID, Rating, Date. Similarly, multiple such groups are stored in each file for other movie IDs.

-  MovieIDs range from 1 to 17770 sequentially.
-  CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users. • Ratings are on a five star (integral) scale from 1 to 5.
-  Dates have the format YYYY-MM-DD

## 🔍 Machine Learning Algorithms (Methodology)
We deployed the **K-means** clustering algorithm for general clustering.
We deployed the **K-medoids** algorithm for clustering users  with an additional **time-based penalty** to penalize similar ratings which have a significant time gap.

To include the time factor in K-medoids, we need to modify the distance metric for calculating the nearest neighbors. We **explored two Python libraries** for implementing the K-medoids algorithm: **sklearn** extra and **pyclustering**, and finalized the former, owing to the variety of functionalities and customizations it provides. The K-medoids algorithm by the sklearn extra library provides a ‘metric‘ parameter that accepts a ‘callable‘ as the distance measure. We use this parameter to pass the custom distance between two users, which is based on both, ratings and time difference. We also require other parameters like ‘cluster labels‘ and ‘inertia‘ for result evaluation. 

## 📊 Results

Km-medoids vs K-means SSE (Sum of Squared error)
![Image 26-01-24 at 1 01 PM](https://github.com/shrutid02/netflix_data_mining/assets/42238433/4b00f939-2576-4af4-b41f-11cadaf2fe3e)

We notice that the error for K-medoids surpasses that of K-means with an increasing number of clusters. A similar trend was observed for another evaluation metric- **silhouette scores** for different cluster sizes. Consequently, we observe that introducing a time-based penalty produces inferior results compared to clustering users based solely on ratings. We also tried **tuning the weight(w)** of the time penalty in the custom distance calculations and even at **w=0.1, K-Means performed better** which further supports our observation. The intention behind adding a time-based penalty was to account for evolving movie interests, but if it increases the error in clustering, solely relying on user ratings would be the favorable clustering method for the Netflix prize data set.

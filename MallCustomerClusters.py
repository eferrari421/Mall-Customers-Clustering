#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:10:05 2020

@author: erikferrari
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

mall = pd.read_csv('Mall_Customers.csv')


#DATA CLEANING


mall.columns

mall.head()

mall.dtypes

mall.info()

mall.shape

mall.nunique()

summary = mall.describe()

#drop duplicates if there are any
mall.drop_duplicates(inplace=True)

#check for any null values
mall.isnull().sum()
#no null values

#rename columns for simplicity
mall = mall.rename(columns={'CustomerID':'customerid', 'Gender':'gender', 'Age':'age', 'Annual Income (k$)':'annual_income', 'Spending Score (1-100)':'spending_score'})


#DATA VISUALIZATION


#let's look at the distributions of the numeric variables
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=mall['age'], shade=True, color='b')

plt.title('Distribution by Annual Income')
plt.xlabel('Annual Income')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=mall['annual_income'], shade=True, color='b')

plt.title('Distribution by Spending Score')
plt.xlabel('Spending Score')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=mall['spending_score'], shade=True, color='b')

#check the distribution of gender
sns.catplot(x='gender', data=mall, kind='count', palette="Blues")
plt.title('Distribution of Gender')

#look at numeric distributions by gender
age_male = pd.DataFrame(mall.age.loc[mall['gender']=='Male'])
age_female = pd.DataFrame(mall.age.loc[mall['gender']=='Female'])
plt.title('Age by Churn')
plt.xlabel('Age')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=age_male['age'], label='Male')
sns.kdeplot(data=age_female['age'], label='Female')

annual_income_male = pd.DataFrame(mall.annual_income.loc[mall['gender']=='Male'])
annual_income_female = pd.DataFrame(mall.annual_income.loc[mall['gender']=='Female'])
plt.title('Annual Income by Churn')
plt.xlabel('Annual Income')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=annual_income_male['annual_income'], label='Male')
sns.kdeplot(data=annual_income_female['annual_income'], label='Female')

spending_score_male = pd.DataFrame(mall.spending_score.loc[mall['gender']=='Male'])
spending_score_female = pd.DataFrame(mall.spending_score.loc[mall['gender']=='Female'])
plt.title('Spending Score by Churn')
plt.xlabel('Spending Score')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=spending_score_male['spending_score'], label='Male')
sns.kdeplot(data=spending_score_female['spending_score'], label='Female')

#let's look at the relationships between the numerical variables
plt.title('Relationship Between Age and Annual Income')
sns.regplot(x='age', y='annual_income', data=mall, color='b')
plt.xlabel('Age')
plt.ylabel('Annual Income')

plt.title('Relationship Between Age and Spending Score')
sns.regplot(x='age', y='spending_score', data=mall, color='b')
plt.xlabel('Age')
plt.ylabel('Spending Score')

plt.title('Relationship Between Annual Income and Spending Score')
sns.regplot(x='annual_income', y='spending_score', data=mall, color='b')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')

#let's look at these same relationships, seperated by gender
sns.lmplot(x='age', y='annual_income', data=mall, hue='gender')
plt.title('Relationship Between Age and Annual Income By Gender')
plt.xlabel('Age')
plt.ylabel('Annual Income')

sns.lmplot(x='age', y='spending_score', data=mall, hue='gender')
plt.title('Relationship Between Age and Spending Score By Gender')
plt.xlabel('Age')
plt.ylabel('Spending Score')

sns.lmplot(x='annual_income', y='spending_score', data=mall, hue='gender')
plt.title('Relationship Between Annual Income and Spending Score By Gender')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')

#heatmap for correlations between variables
plt.title('Correlation Between All Variables')
sns.heatmap(data=mall.corr(), square=True , annot=True, cbar=True, cmap='Blues')


#K-MEANS CLUSTERING


#AGE AND SPENDING SCORE
age_and_spending = mall[['age', 'spending_score']].iloc[:,:].values
kmeans_kwargs = {'init':'random', 'n_init':10, 'max_iter':300, 'random_state':42}
sse_1 = []
for k in range(1,11):
    kmeans_1 = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans_1.fit(age_and_spending)
    sse_1.append(kmeans_1.inertia_)

#look at scree plot to determine elbow(optimal number of clusters)
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse_1)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Area")
plt.show()

#optimal number of clusters is 4
kmeans_2 = KMeans(n_clusters=4, **kmeans_kwargs)
y_kmeans_2 = kmeans_2.fit_predict(age_and_spending)

#visualize clusters
plt.scatter(age_and_spending[y_kmeans_2 == 0, 0], age_and_spending[y_kmeans_2 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(age_and_spending[y_kmeans_2 == 1, 0], age_and_spending[y_kmeans_2 == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(age_and_spending[y_kmeans_2 == 2, 0], age_and_spending[y_kmeans_2 == 2, 1], s = 100, c = 'blue', label = 'Cluster 3')
plt.scatter(age_and_spending[y_kmeans_2 == 3, 0], age_and_spending[y_kmeans_2 == 3, 1], s = 100, c = 'pink', label = 'Cluster 4')
plt.scatter(kmeans_2.cluster_centers_[:, 0], kmeans_2.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend()

#ANNUAL INCOME AND SPENDING SCORE
annual_income_and_spending = mall[['annual_income', 'spending_score']].iloc[:,:].values
sse_2 = []
for k in range(1,11):
    kmeans_3 = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans_3.fit(annual_income_and_spending)
    sse_2.append(kmeans_3.inertia_)

#look at scree plot to determine elbow(optimal number of clusters)
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse_2)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Area")
plt.show()

#optimal number of clusters is 5
kmeans_4 = KMeans(n_clusters=5, **kmeans_kwargs)
y_kmeans_4 = kmeans_4.fit_predict(annual_income_and_spending)

#visualize clusters
plt.scatter(annual_income_and_spending[y_kmeans_4 == 0, 0], annual_income_and_spending[y_kmeans_4 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(annual_income_and_spending[y_kmeans_4 == 1, 0], annual_income_and_spending[y_kmeans_4 == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(annual_income_and_spending[y_kmeans_4 == 2, 0], annual_income_and_spending[y_kmeans_4 == 2, 1], s = 100, c = 'blue', label = 'Cluster 3')
plt.scatter(annual_income_and_spending[y_kmeans_4 == 3, 0], annual_income_and_spending[y_kmeans_4 == 3, 1], s = 100, c = 'pink', label = 'Cluster 4')
plt.scatter(annual_income_and_spending[y_kmeans_4 == 4, 0], annual_income_and_spending[y_kmeans_4 == 4, 1], s = 100, c = 'black', label = 'Cluster 5')
plt.scatter(kmeans_4.cluster_centers_[:, 0], kmeans_4.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()

#AGE, ANNUAL INCOME, AND SPENDING SCORE
age_annual_income_and_spending = mall[['age', 'annual_income', 'spending_score']].iloc[:,:].values
sse_3 = []
for k in range(1,11):
    kmeans_5 = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans_5.fit(age_annual_income_and_spending)
    sse_3.append(kmeans_5.inertia_)

#look at scree plot to determine elbow(optimal number of clusters)
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse_3)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Area")
plt.show()

#optimal number of clusters is 6
kmeans_6 = KMeans(n_clusters=6, **kmeans_kwargs)
y_kmeans_6 = kmeans_6.fit_predict(age_annual_income_and_spending)

#visualize clusters
ax = plt.axes(projection='3d')
ax.scatter3D(age_annual_income_and_spending[y_kmeans_6 == 0, 0], age_annual_income_and_spending[y_kmeans_6 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
ax.scatter3D(age_annual_income_and_spending[y_kmeans_6 == 1, 0], age_annual_income_and_spending[y_kmeans_6 == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
ax.scatter3D(age_annual_income_and_spending[y_kmeans_6 == 2, 0], age_annual_income_and_spending[y_kmeans_6 == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
ax.scatter3D(age_annual_income_and_spending[y_kmeans_6 == 3, 0], age_annual_income_and_spending[y_kmeans_6 == 3, 1], s = 100, c = 'pink', label = 'Cluster 4')
ax.scatter3D(age_annual_income_and_spending[y_kmeans_6 == 4, 0], age_annual_income_and_spending[y_kmeans_6 == 4, 1], s = 100, c = 'black', label = 'Cluster 5')
ax.scatter3D(age_annual_income_and_spending[y_kmeans_6 == 5, 0], age_annual_income_and_spending[y_kmeans_6 == 5, 1], s = 100, c = 'orange', label = 'Cluster 6')
plt.title('Clusters of customers')
plt.xlabel('Age')
plt.ylabel('Annual Income')
ax.set_zlabel('Spending Score')
plt.legend()

















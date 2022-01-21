#!/usr/bin/env python
# coding: utf-8

# # Name : Nikita Wani
# 
# The Sparks Foundation : GRIP
# 
# Task 2 : Prediction Using Unsupervised ML
#  
# From the given 'Iris' dataset, predict the optimum number of clusters and represent it visually .

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head()


# In[3]:


iris_df.tail()


# In[4]:


iris_df.isnull().any().sum()


# In[5]:


iris_df.columns


# # scatter plot based on the Seapl height width and Petal height and width

# In[6]:


sns.stripplot(data=iris_df)


# # Grouping variables in Seaborn countplot with Seapl height width and Petal height and width

# In[7]:


sns.countplot(data=iris_df)


# # Plotting a line graph using elbow method

# In[9]:


x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss,marker='.')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # Visualizing Data

# In[10]:


sns.pairplot(data=iris_df)


# In[11]:


sns.jointplot(kind = "kde", data = iris_df)
plt.show()


# In[12]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# # Cluster visualizaton of Sepal length and width

# In[13]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'black', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'violet', label = 'Iris-versicolour')

plt.legend()


# In[14]:


plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'grey', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# # Plot All K-Means Clusters of Sepal length and width

# In[15]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'black', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'violet', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'grey', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# # Cluster visualizaton of Petal length and width

# In[16]:


plt.scatter(x[y_kmeans == 0, 2], x[y_kmeans == 0, 3],
            s = 100,c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 2], x[y_kmeans == 1, 3],
            s = 100,c = 'blue', label = 'Iris-versicolour')

plt.legend()
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()


# In[20]:


plt.scatter(x[y_kmeans == 2, 2], x[y_kmeans == 2, 3],
            s = 100,c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3],
            c = 'black', label = 'Centroids')

plt.legend()
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()


# # Plot All K-Means Clusters of Petal length and width

# In[19]:


plt.scatter(x[y_kmeans == 0, 2], x[y_kmeans == 0, 3],
            s = 100,c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 2], x[y_kmeans == 1, 3],
            s = 100,c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 2], x[y_kmeans == 2, 3],
            s = 100,c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3],
            c = 'black', label = 'Centroids')

plt.legend()
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()


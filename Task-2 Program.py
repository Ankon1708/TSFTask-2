# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 20:45:19 2021

@author: ankon
"""
#%%Importing modules and the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('TASK-2 DATA.csv')
'''
#print(df.to_string()) 
#print(type(df))
#print("Info on training examples:",df.info())

ndarr=df.to_numpy()
#print(type(ndarr))
#print(ndarr[1])
ndarr=ndarr[:,1:5]
#print(ndarr)
'''
print("Info on training examples:",df.info())

#Converting dataframe to a 2-D numpy array
ndarr=df.to_numpy()
#Eliminating features  Id and Species
ndarr=ndarr[:,1:5]

#Feature Scaling
scaler = StandardScaler()
scaled_ndarr = scaler.fit_transform(ndarr)

#%%Elbow method
#Array to hold Sum of Squared Errors for each value of k(2 to 10)
SSE=[]
for k in range(1,11):
    iriskm=KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300)
    iriskm.fit(scaled_ndarr)
    SSE.append(iriskm.inertia_)

#Plotting graph SSE v/s No. of Clusters
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), SSE)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
    
#Confirmation of elbow using KneeLocator(package kneed)
kl = KneeLocator(
    range(1, 11), SSE, curve="convex", direction="decreasing")
print("Elbow:",kl.elbow)


#%%K-means with obtained optimal value of No. of clusters

K=kl.elbow

iriskm=KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=300)
iriskm.fit(scaled_ndarr)

print("Labels:\n",iriskm.labels_,"\nSSE:",iriskm.inertia_,"\nNo. of iterations run:",iriskm.n_iter_)

centers=scaler.inverse_transform(iriskm.cluster_centers_)
print("Cluster Centers:\n",centers)


#%%Interpreting the results
#Performing PCA to obtain two principal components of maximal amount of variance

pca_ndarr=PCA(n_components=2).fit(scaled_ndarr)
print("Variance Ratio:",round(sum(pca_ndarr.explained_variance_ratio_)*100,3),"%")

#print("PCs of data:\n",pca_ndarr.transform(ndarr).shape)
reduceddf=pd.DataFrame(pca_ndarr.transform(ndarr),index=df.index,columns=['PC1','PC2'])
reduceddf['Cluster']=iriskm.labels_
reducedcenters=pca_ndarr.transform(centers)
print("Reduced Centers:\n",reducedcenters)


#%%Visualization

#print(reduceddf[reduceddf['Cluster']==0])
clusters=reduceddf.Cluster.unique()
print(clusters)
colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])

for x in clusters:
    temp=reduceddf[reduceddf['Cluster']==x]
    plt.scatter(temp['PC1'],temp['PC2'],c=colors[x])
plt.scatter(reducedcenters[:,0],reducedcenters[:,1],marker='X',c="black")
plt.show()
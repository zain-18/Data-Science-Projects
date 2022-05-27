from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def Plot_Hie_Silhouette(X_data,linkage,Range):
    s_score=[]
    for i in range(2,int(Range)):
        labels=AgglomerativeClustering(n_clusters=i,linkage=linkage).fit_predict(X_data)
        s_score.append(silhouette_score(X_data,labels))
    
    plt.figure(figsize=(10,5))
    sns.set_context('poster')
    plt.plot(range(2,10),s_score)
    plt.title('Silhouette Score with '+linkage+' Linkage')
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette_score')
    plt.show()
    
def Plot_Kmean_Silhouette(X_data,Range):
    s_score=[]
    for i in range (2,int(Range)):
        kmeans=KMeans(n_clusters=i).fit_predict(X_data)
        s_score.append(silhouette_score(X_data,kmeans))
        
    plt.figure(figsize=(12,5))
    sns.set_context('poster')
    plt.plot(range(2,int(Range)),s_score)
    plt.xlabel('K')
    plt.ylabel('Silhouette_score')
    plt.show()
def Elbow(X_data):
    Cost=[]
    for i in range(1,10):
        kmeans=KMeans(n_clusters=i).fit(X_data)
        Cost.append(kmeans.inertia_)
    plt.figure(figsize=(12,5))
    sns.set_context('poster')
    plt.plot(range(1,10),Cost)
    plt.xlabel('K')
    plt.ylabel('Cost')
    plt.show()
    
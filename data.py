import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from R_square_clustering import r_square
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import numpy as np




euro = pd.read_csv('eurostat-2013.csv')

""" 
Question 1
- tps00001
- tsc00004
- tet00002
"""


euro['tsc00004 (2012)'] /= euro['tps00001 (2013)']
euro['tet00002 (2013)'] /= euro['tps00001 (2013)']
euroClean = euro.drop(['tps00001 (2013)'], axis = 1)

print (euroClean)


#Question 2

stdScaler = StandardScaler()
eurofit = stdScaler.fit_transform(euroClean[euroClean.columns[2:]])
print(eurofit)


#Question 3
pca = PCA(svd_solver='full')
coord = pca.fit_transform(eurofit)
print(pca.explained_variance_ratio_)
n = np.size(eurofit, 0)
p = np.size(eurofit, 1)
eigval = float(n-1)/n*pca.explained_variance_
fig = plt.figure()

plt.plot(np.arange(1, p+1), eigval)
plt.title('Scree plot')
plt.ylabel('Eigen values')
plt.xlabel('Factor number')
plt.savefig('acp_eigen_values')
plt.close()

print(pca.components_)

sqrt_eigval = np.sqrt(eigval)
corvar = np.zeros((p,p))


for k in range(p):
    corvar[:,k] = pca.components_[k,:] * sqrt_eigval[k]

print(corvar)

fig, axes = plt.subplots(figsize = (12, 12))
axes.set_xlim(-7, 7)
axes.set_ylim(-5, 5)

for i in range(n):
    plt.annotate(euro['Code'].values[i], (coord[i, 0], coord[i, 1]))

plt.plot([-7, 7], [0, 0], color = 'silver', linestyle = '-', linewidth = 1)
plt.plot([0, 0], [-5, 5], color = 'silver', linestyle = '-', linewidth = 1)
plt.savefig('acp_instances_1_2')
plt.close(fig)



fig, axes = plt.subplots(figsize = (12, 12))
axes.set_xlim(-4, 4)
axes.set_ylim(-5, 5)

for i in range(n):
    plt.annotate(euro['Code'].values[i], (coord[i, 2], coord[i, 3]))

plt.plot([-4, 4], [0, 0], color = 'silver', linestyle = '-', linewidth = 1)
plt.plot([0, 0], [-5, 5], color = 'silver', linestyle = '-', linewidth = 1)
plt.savefig('acp_instances_3_4')
plt.close(fig)

#Question 4
''' 
1_2 -> x teimf00118, y tec00115
3_4 -> x tsdsc260, y tet00002

Groupes de pays ->cf graphe 
'''

def correlation_circle(df,nb_var,x_axis,y_axis):

    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    # label with variable names
    for j in range(nb_var):
        # ignore two first columns of df: Nom and Code^Z
        plt.annotate(df.columns[j+2],(corvar[j,x_axis],corvar[j,y_axis]))
    # axes
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)

correlation_circle(euroClean, 9, 0, 1)
correlation_circle(euroClean, 9, 2, 3)



print('--------------------------------------------------------------------------')
print('--------------------------------------------------------------------------')

#Question 5

y = euro["Code"]
est = KMeans(n_clusters=4, random_state=0).fit(eurofit)
# print centroids associated with several countries
lst_countries=['EL','FR','DE','US']
# centroid of the entire dataset
# est: KMeans model fit to the dataset
print(est.cluster_centers_)
for name in lst_countries:
    num_cluster = est.labels_[y.loc[y==name].index][0]
    print('Num cluster for '+name+': '+str(num_cluster))
    print('\tlist of countries: '+', '.join(y.iloc[np.where(est.labels_==num_cluster)].values))
    print('\tcentroid: '+str(est.cluster_centers_[num_cluster]))


print('--------------------------------------------------------------------------')
print('--------------------------------------------------------------------------')

lst_k=range(2,9)
lst_rsq = []

for k in lst_k:
        est = KMeans(n_clusters = k)
        est.fit(eurofit)
        lst_rsq.append(r_square(eurofit, est.cluster_centers_, est.labels_, k))

fig = plt.figure()
plt.plot(lst_k,lst_rsq,'bx-')
plt.xlabel('k')
plt.ylabel('RSQ')
plt.title('The Elbow Method showing the optimal k')
plt.show()
plt.close()
print('--------------------------------------------------------------------------')
print('--------------------------------------------------------------------------')
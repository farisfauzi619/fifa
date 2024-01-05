import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df_fifa = pd.read_csv('fifa.csv')

df_fifa.rename(columns={
    'population_share': 'population',
    'tv_audience_share': 'tv',
    'gdp_weighted_share': 'gdp'
}, inplace=True)

x = df_fifa.drop(['country', 'confederation'], axis=1)

st.header("Isi Dataset")
st.write(df_fifa)

clusters=[]
for i in range(1,11):
  km=KMeans(n_clusters=i).fit(x)
  clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('clusters')
ax.set_ylabel('inertia')

#panah elbow
ax.annotate('Possible elbow point', xy=(2, 750), xytext=(3, 1000),xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible elbow point', xy=(3, 350), xytext=(3, 1000),xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :", 2,10,3,1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(x)
    x['Labels'] = kmean.labels_
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='population', y='tv', size='gdp', hue='Labels', markers=True, data=x, palette=sns.color_palette('hls', n_clust))

    for label in x['Labels']:
        mean_population = x[x['Labels'] == label]['population'].mean()
        mean_tv = x[x['Labels'] == label]['tv'].mean()
        mean_gdp = x[x['Labels'] == label]['gdp'].mean()

        plt.annotate(label,
                 (mean_population, mean_tv),
                 xytext=(mean_population, mean_tv),  # Use the same values for xytext
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold',
                 color='black')
        
    st.header('Cluster Plot')
    st.pyplot()
    st.write(x)
    
k_means(clust)
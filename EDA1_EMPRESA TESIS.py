"""
EDA1_EMPRESA TESIS
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#columnas =['PedidoId', 'Cantidad', 'MontoProd', 'MontoFinalProd', 'Producto','ID CLIENTE', 'Ciudad', 'Fecha']
df=pd.read_excel("C:/Users/HP/Desktop/TESIS/TESIS EXCEL/DatosNDA5.xlsx")


# In[ ]:


#df['Fecha']=pd.to_datetime(df['Fecha'] )


# In[3]:


df


# In[4]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df.columns


# In[ ]:


df


# In[ ]:


df['Departamento'].value_counts()


# In[ ]:


import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

col_features= ['Cantidad','MontoFinal']
x = MinMaxScaler().fit_transform(df[col_features])

kmeans= KMeans(init='k-means++', n_clusters=4, random_state=15).fit(x)
df['kmeans'] = kmeans.labels_


# In[ ]:


df


# In[ ]:


sns.scatterplot(data=df, x="Cantidad", y="MontoFinal", hue='kmeans', palette='Accent')


# In[ ]:





# In[ ]:


len(df['Id_Cliente'].unique())


# In[ ]:


len(df['PedidoId'].unique())


# In[ ]:


df


# In[5]:


dfFecha=df


# In[ ]:





# In[ ]:





# In[6]:


dfFecha=df.groupby(by="Fecha").sum()

#filt = (df['Cantidad']<1000) &(df['MontoFinalProd']<5000)
#df=df.loc[filt]
dfFecha=dfFecha.reset_index()
dfFecha


# In[7]:


dfFecha.isna().sum()


# In[ ]:





# In[10]:


sns.lineplot(data=dfFecha, x="Fecha", y="MontoFinal")


# In[12]:


sns.boxplot(y=df['MontoFinal'])


# In[ ]:


df.describe()


# In[13]:


sns.boxplot(y=df['Cantidad'])


# In[ ]:





# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

col_features= ['Cantidad','MontoFinalProd']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[col_features])

kmeans = KMeans(init="random", n_clusters=4, n_init=10, max_iter=300, random_state=42)

kmeans.fit(scaled_features)

df['kmeans2'] = kmeans.labels_


# In[ ]:


sns.scatterplot(data=df, x="Cantidad", y="MontoFinalProd", hue='kmeans2', palette='Accent_r')


# In[ ]:


print(kmeans.inertia_)
print(kmeans.cluster_centers_)
print(kmeans.n_iter_)



# In[ ]:


kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}


sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from scipy import stats

z = np.abs(stats.zscore(df['MontoFinalProd']))
#print(z)
#type(z)
dfz= pd.DataFrame(z)
dfz


# In[ ]:


#threshold = 3
#print(np.where(z > 3))
df=pd.concat([df, dfz], axis=1)
df


# In[ ]:


df=df.rename(columns={0:'z'})


# In[ ]:


#retirando los outliers
filtz=df['z']<3
df=df.loc[filtz]
df=df.drop(columns=['z'])
df=df.reset_index(drop= True)
df


# In[ ]:


sns.scatterplot(data=df, x="Cantidad", y="MontoFinalProd")


# In[ ]:


sns.boxplot(y=df['MontoFinalProd'])


# In[ ]:


sns.boxplot(y=df['Cantidad'])


# In[ ]:


import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

col_features= ['Cantidad','MontoFinalProd']
x = MinMaxScaler().fit_transform(df[col_features])

kmeans= KMeans(init='k-means++', n_clusters=4, random_state=15).fit(x)
df['kmeans'] = kmeans.labels_


# In[ ]:





# In[ ]:


df


# In[ ]:


sns.scatterplot(data=df, x="Cantidad", y="MontoFinalProd", hue='kmeans', palette='Accent_r')


# In[ ]:


#df.to_excel(r'C:/Users/HP/Desktop/TESIS/EMPRESA_NDA/DATOS/DATOS_NDA.xlsx',sheet_name='hoja1')


# In[ ]:


#import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#col_features= ['Cantidad','MontoFinalProd']
#scaler = StandardScaler()
#scaled_features = scaler.fit_transform(col_features)

#kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)

#kmeans.fit(scaled_features)


kmeans.inertia_
kmeans.cluster_centers_
kmeans.n_iter_

kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}


sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


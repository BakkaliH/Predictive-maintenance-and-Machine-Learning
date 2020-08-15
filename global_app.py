#!/usr/bin/env python
# coding: utf-8

# In[127]:

#To see the results of the app
#Go to Anaconda Prompt
#Print 'cd Desktop'
#Print 'streamlit run global_app.py'
#When you open the app, you can see all your results in the frontend(right part)
#You can get your prediction values by playing in the backend(dont forget jour means day)
#Let's enjoy a little bit
import pandas as pd
import numpy as np
import streamlit as st
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
from datetime import datetime, date, time
from subprocess import call
from IPython.display import Image
from sklearn import preprocessing
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
from sklearn import metrics
import matplotlib.pyplot as plt

st.write("""
#  Prediction App
Downtime prediction application
""")

st.sidebar.header('User input parameters')
def user_input_features():
    jour = st.sidebar.slider('jour',1,31,9) 
    Machine_M1 = st.sidebar.slider('Machine_M1',0,1,0) 
    Machine_M2 = st.sidebar.slider('Machine_M2',0,1,0)
    Prob_P1 = st.sidebar.slider('Prob_P1',0,1,0)
    Prob_P2 = st.sidebar.slider('Prob_P2',0,1,0)
    Prob_P5 = st.sidebar.slider('Prob_P5',0,1,0)
    Prob_P8 = st.sidebar.slider('Prob_P8',0,1,0)
   
    data = {
        'jour': jour,
            'Machine_M2': Machine_M2,
           'Machine_M1': Machine_M1,
           "Prob_P1": Prob_P1,
        "Prob_P5": Prob_P5,
        "Prob_P2": Prob_P2,
        "Prob_P8": Prob_P8,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

features0 = pd.read_csv("C:/Users/Dell/Desktop/bablaa.csv")
st.dataframe(features0)
Total1= features0["Temps d'arret"].sum()
Total2= features0['temps prevu'].sum()

st.subheader("the total downtime is:")
st.write(Total1)
st.subheader("the total time provided by the maintenance agents before our procedure is:")
st.write(Total2)

feat=features0.describe(include='all')
st.dataframe(feat)


sns.kdeplot(features0["temps prevu"],shade = True, label = "temps prevu",bw = 5)
sns.kdeplot(features0["Temps d'arret"], shade = True,label = "Temps d'arret",bw = 5)
st.pyplot()

featjour=features0.groupby(['jour']).sum()
featjour=featjour.drop(['Annee','mois'],axis=1)
featjour = featjour.sort_values(by="Temps d'arret", ascending=False)
featjour = featjour.style.background_gradient(cmap='Reds')
st.subheader("Actual and planned downtime per day")
st.write(featjour)

featMach=features0.groupby(['Machine']).sum()
featMach=featMach.drop(['Annee','mois','jour'],axis=1)
featMach = featMach.sort_values(by="Temps d'arret", ascending=False)
featMach = featMach.style.background_gradient(cmap='Reds')
st.subheader("Actual and planned downtime Machine")
st.write(featMach)

featProb=features0.groupby(['Prob']).sum()
featProb=featProb.drop(['Annee','mois','jour'],axis=1)
featProb = featProb.sort_values(by="Temps d'arret", ascending=False)
featProb = featProb.style.background_gradient(cmap='Reds')
st.subheader("Actual and planned downtime Problem")
st.write(featProb)

features1 = pd.read_csv("C:/Users/Dell/Desktop/bablaa.csv")
features1 = pd.get_dummies(features1)
features1 = features1.drop(features1.columns.difference(["Temps d'arret",'jour']),axis = 1)
x = features1.iloc[:,[0,1]].values
wcss =[]
distances = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans = KMeans(n_clusters = i).fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 20), wcss)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('wcss')
st.pyplot()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 0')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title("Downtime clusters")
plt.xlabel('Day')
plt.ylabel("Downtime")
plt.legend()
st.pyplot()

pred = kmeans.predict(x)
frame = pd.DataFrame(x)
frame['cluster'] = pred
Nbre = frame['cluster'].value_counts()
st.subheader("Number of observation per cluster")
st.write(Nbre)

features = pd.read_csv("C:/Users/Dell/Desktop/bablaa.csv")
features = pd.get_dummies(features)
features = features.drop(features.columns.difference(["Temps d'arret",'jour','Machine_M2','Machine_M1','Prob_P1','Prob_P5','Prob_P8','Prob_P2']),axis = 1)
labels = np.array(features["Temps d'arret"])
features= features.drop("Temps d'arret", axis = 1)
features_list = list(features.columns)
features= np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
important_indices = [features_list.index('jour'),features_list.index('Machine_M2'),features_list.index('Machine_M1'),features_list.index('Prob_P1'),features_list.index('Prob_P5'),features_list.index('Prob_P8'),features_list.index('Prob_P2')]

train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
rf_most_important.fit(train_important, train_labels)
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape

prediction = rf_most_important.predict(df)


st.subheader("The mean absolute error")
st.write(round(np.mean(errors), 2), 'min.')

st.subheader('Accuracy')
st.write(round(np.mean(accuracy), 2), '%.')

st.subheader('Prediction')
st.write(round(np.mean(prediction), 2),'min.')


# In[ ]:





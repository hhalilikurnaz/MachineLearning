import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram,linkage


#Veri Setini yüklüyorum
try:
    nba_df=pd.read_csv("nba_2013.csv")
    print("NBA 2013 Verisi Başarıyla yüklendi")

except FileNotFoundError:
    print("Hata : 'nba_2013.csv' dosyası bulunamadı .")
    exit()


print(nba_df.head(5))

print("\n Veri Seti Bilgisi")
print(nba_df.info())
#Temel İstatistikler 
print("\n Özet")
print(nba_df.describe())


#missing value control
print("\n missing value check")
print(nba_df.isnull().sum())

#Veri setinin içinde nonintager veriler var bundan dolayı kümele algoritması için sadece integer verileri seçiyorum ki işim kolay olsun 

features_df=nba_df.drop(['player','team'],axis=1)

#eksik verileri dolduruyorum 

for col in features_df.columns:
    if features_df[col].isnull().any():
        features_df[col]=features_df[col].fillna(features_df[col].mean())
print("\nEksik değerler ortalama ile doldurduldu")
print(features_df.isnull().sum().sum()) #tüm değerlerde missing value 0 mı diye bakıyoruz check ediyorum 

#Veri PreProceessing
scaler=StandardScaler()
scaled_features=scaler.fit_transform(features_df)
#DataFrame'e çevirme 
scaled_features_df=pd.DataFrame(scaled_features,columns=features_df.columns)
print("\nVeri başarıyla ölçeklendirildi. İlk 5 satır:")
print(scaled_features_df.head())
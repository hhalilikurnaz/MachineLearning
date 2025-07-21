from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage


import matplotlib.pyplot as plt

X,_=make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=42)
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.title("Örnek Veri")
plt.show()

linkage_methods={"ward","single","avarage","complete"} #bağlama metodlarım
'''
1. ward → "Varyansı minimize et"
Bilim insanı gibi çalışan yöntem.

Küme içindeki toplam kareler toplamı (intra-cluster variance) en az olacak şekilde kümeleri birleştirir.
Genelde iyi sonuçlar verir.
Ama sadece Öklidyen mesafe ile çalışır!
 Kullanım: linkage='ward'

2. single → "En yakın iki noktayı birleştir"
 Aşırı romantik yöntem… En yakınları hep birleştiriyor.

İki küme arasındaki en kısa mesafeye bakar (minimum distance).
Ama dikkat: “zincirleme bağlama” (chaining) riski var. Yani uzun ince dallar oluşturabilir.
Kullanım: linkage='single'

3. complete → "En uzak noktaya göre karar ver"
 Güvenlikçi: “Ben riski alacak adam değilim!”

İki küme arasındaki en uzak noktayı baz alır (maximum distance).
Grupları sıkı tutar, zincirleme problemi yaşamaz ama çok "konservatif" kalabilir.
 Kullanım: linkage='complete'

4. average → "Ortalama mesafeye göre"
 Dengeli yaklaşım. Ne çok cesur, ne çok temkinli.

İki küme arasındaki tüm noktalar arası mesafenin ortalaması alınır.
Hem single’dan dengeli, hem complete’ten esnek.
 Kullanım: linkage='average'


'''
plt.figure()
for i, linkage_methods in enumerate(linkage_methods,1):
    model=AgglomerativeClustering(n_clusters=4,linkage=linkage_methods)
    cluster_labels=model.fit_predict(X)
    #neden traşn test split yapmıyoruz ?
    #herbir kümede train olucak etiketler olucaktı ama test etiketlerim yok ondan dolayı direkt fit yapıyoruz

    plt.subplot(2,4,i)
    plt.title(f"{linkage_methods.capitalize()} Linkage Dendogram")
    dendrogram(linkage(X,method=linkage_methods),no_labels=True)
    plt.xlabel("Veri Noktaları")
    plt.ylabel("uzaklık")
    plt.show()

    plt.subplot(2,4,i+4)
    plt.scatter(X[:,0],X[:,1],c=cluster_labels,cmap="viridis")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()



import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, cluster
from sklearn.preprocessing import StandardScaler

# --- Veri Kümesi Tanımlamaları ---
n_samples = 1500

noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples)
no_structure = np.random.rand(n_samples, 2), None # tuple yapısını koruyoruz

clustering_names = ["MiniBatchKMeans", "SpectralClustering", "Ward", "AgglomerativeClustering", "DBSCAN", "Birch"]

# Yeterli renge sahip bir numpy array tanımlayalım veya dinamik renkler kullanalım
# Mevcut durumda 6 algoritma ve genelde 2 küme için yetecektir.
# Daha fazla küme oluşursa otomatik renk seçimi yapacağız.
plot_base_colors = np.array(["b", "g", "r", "c", "m", "y", "k,""#FFA500", "#800080", "#00FFFF"]) # Ek renkler

datasets_list = [noisy_circles, noisy_moons, blobs, no_structure] # 'datasets' değişken adı çakışmasından kaçınmak için değiştirdim

# --- Grafik Oluşturma ---
# Tüm alt grafiklerin sığacağı tek bir figure (pencere) oluşturuyorum

plt.figure(figsize=(24, 16)) 

plot_num = 1 # Alt grafik konumunu takip etmek için sayaç (subplot numaralandırması 1'den başlar)

for i_dataset, dataset in enumerate(datasets_list): # datasets_list kullanıyoruz
    X, y = dataset # X: özellikler, y: gerçek etiketler (kıyaslama için)
    X = StandardScaler().fit_transform(X) # Veriyi ölçeklendir

    # Kümeleme Algoritmalarını Tanımlama 
    two_means = cluster.MiniBatchKMeans(n_clusters=2, random_state=42, n_init=10)
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage="ward")
    spectral = cluster.SpectralClustering(n_clusters=2, n_init=10)
    dbscan = cluster.DBSCAN(eps=0.2) # eps değerini veri setinize göre ayarlayabilirsiniz
    average_linkage = cluster.AgglomerativeClustering(n_clusters=2, linkage="average")
    birch = cluster.Birch(n_clusters=2) # n_clusters burada bir varsayılan değeri, Birch küme sayısını dinamik olarak da belirleyebilir.

    clustering_algorithms = [two_means, ward, spectral, dbscan, average_linkage, birch]

    for name, algo in zip(clustering_names, clustering_algorithms):
        # Her bir algoritmayı veri kümesine uyguluyorum
        try:
            # fit işlemi, algoritmaların dahili olarak kümeleme yapmasını sağlar.
            # Bazı algoritmalar y_pred'i doğrudan 'labels_' attribute'u ile döndürür.
            # SpectralClustering için predict() methodu da kullanılabilir ama fit sonrası labels_ de mevcuttur.
            # DBSCAN gürültü noktalarını -1 olarak etiketler.
            algo.fit(X) 
            if hasattr(algo, "labels_"):
                y_pred = algo.labels_.astype(int)
            else:
                # Nadiren bu kısma düşeriz, çünkü çoğu sklearn kümeleyici labels_ attribute'unu kullanır.
                y_pred = algo.predict(X) 
        except Exception as e:
            print(f"Hata oluştu: {name} için {e}")
            # Hata durumunda, grafiğin boş kalmaması için tüm noktaları aynı küme olarak işaretleyelim.
            y_pred = np.zeros(X.shape[0], dtype=int) 

        # --- Alt Grafik Çizimi ---
        # plt.subplot(satır_sayısı, sütun_sayısı, mevcut_grafik_indeksi)
        plt.subplot(len(datasets_list), len(clustering_algorithms), plot_num)
        
        # Sadece ilk satırdaki grafiklerin üzerine algoritma adını başlık olarak yaz
        if i_dataset == 0:
            plt.title(name, size=14)
        
        # Kümeleme sonuçlarını renklendirme
        unique_labels = np.unique(y_pred)
        n_clusters_found = len(unique_labels)
        
        # Eğer unique_labels arasında -1 (DBSCAN gürültüsü) varsa, bunu da ayrı bir renk olarak ele alalım
        if -1 in unique_labels:
            n_clusters_found -= 1 # Gürültü noktalarını küme sayısına dahil etmeyiz genelde

        if n_clusters_found > len(plot_base_colors):
            # Eğer bulunan küme sayısı, önceden tanımlı renklerden fazlaysa, bir renk haritası kullan
            colors_for_plot = plt.cm.get_cmap('viridis', n_clusters_found + (1 if -1 in unique_labels else 0)) 
        else:
            colors_for_plot = plot_base_colors # Tanımlı renkleri kullan

        # Noktaları çizerken, y_pred'deki -1 etiketini (gürültü) gri renkle göstermek isteyebilirsiniz.
        # Bunun için mapping yapmanız gerekebilir.git
        mapped_y_pred = np.copy(y_pred)
        if -1 in unique_labels:
            # -1 olan gürültü noktalarını genellikle farklı bir renkle (örneğin siyah/gri) göstermek isteriz.
            # Eğer renk haritası kullanıyorsak, bunu en sona veya özel bir konuma atayabiliriz.
            # Şimdilik, sadece negatif olmayan indeksleri kullanarak rengi belirliyoruz, -1 olanlar renk dizisinin dışına çıkar.
            # Bu durumda, -1 olan noktalar için özel bir renk ataması yapmalıyız.
            
            # Gürültü noktalarını farklı bir renkle (örneğin siyah) çizmek için ayrı bir scatter çağrısı yapabiliriz.
            noise_mask = (y_pred == -1)
            # Küme noktalarını çiz
            plt.scatter(X[~noise_mask, 0], X[~noise_mask, 1], 
                        c=colors_for_plot[mapped_y_pred[~noise_mask]], s=10)
            # Gürültü noktalarını çiz
            plt.scatter(X[noise_mask, 0], X[noise_mask, 1], 
                        c='k', s=10, alpha=0.5) # Gürültü için siyah (k) ve biraz şeffaf
        else:
            plt.scatter(X[:,0], X[:,1], c=colors_for_plot[y_pred], s=10)
            

        plt.xticks(()) # X ekseni etiketlerini gizle
        plt.yticks(()) # Y ekseni etiketlerini gizle

        plot_num += 1 # Bir sonraki alt grafik için sayacı artır

# TÜM DÖNGÜLER BİTTİKTEN SONRA SADECE BİR KEZ plt.show() ÇAĞIRILMALI
plt.tight_layout() # Alt grafiklerin düzgün hizalanmasını sağlar
plt.show() # Oluşturulan tüm grafikleri tek bir pencerede göster
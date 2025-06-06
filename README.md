# Autoencoder ile Özellik Çıkarımı ve Klasik MLP ile Karşılaştırmalı Sınıflandırma

## Giriş (Introduction)

Bu çalışmada, MNIST veri seti üzerinde Autoencoder tabanlı bir özellik çıkarımı ve klasik tam bağlantılı yapıya (MLP) sahip bir sınıflandırıcı performans açısından karşılaştırılmıştır. Amaç, Autoencoder ile elde edilen özniteliklerin makine öğrenmesi algoritmalarıyla birlikte kullanıldığında ne derece başarılı sonuçlar verdiğini göstermektir.

### Problem Tanımı
Yüksek boyutlu veri setlerinde sınıflandırma yaparken, özellik çıkarımı ve boyut indirgeme önemli bir rol oynamaktadır. Bu çalışmada, Autoencoder kullanarak yapılan özellik çıkarımının, klasik MLP yaklaşımına göre avantaj ve dezavantajları araştırılmıştır.

### Çalışmanın Önemi
- Düşük boyutlu özellik vektörleri ile sınıflandırma performansının değerlendirilmesi
- Bellek kullanımı ve hesaplama karmaşıklığının karşılaştırılması
- Özellik çıkarımı ve sınıflandırma arasındaki ilişkinin incelenmesi

## Yöntem (Method)

### Veri Seti
- **MNIST**: 28x28 boyutunda gri tonlamalı el yazısı rakam görüntüleri
- 60,000 eğitim, 10,000 test örneği
- 10 sınıf (0-9 arası rakamlar)
- Piksel değerleri [0,1] aralığında normalize edilmiştir

### Autoencoder Mimarisi

#### Teorik Altyapı
Autoencoder, giriş verisini düşük boyutlu bir latent uzaya kodlayan ve bu kodlamadan orijinal veriyi yeniden oluşturmaya çalışan bir yapay sinir ağıdır. Bu çalışmada kullanılan Autoencoder mimarisi:

1. **Encoder (Kodlayıcı)**:
   - Giriş katmanı: 784 nöron (28x28 piksel)
   - Gizli katman 1: 128 nöron, ReLU aktivasyon
   - Gizli katman 2: 32 nöron, ReLU aktivasyon

2. **Decoder (Kod Çözücü)**:
   - Gizli katman 1: 128 nöron, ReLU aktivasyon
   - Çıkış katmanı: 784 nöron, Sigmoid aktivasyon

#### Eğitim Parametreleri
- Optimizer: Adam (learning_rate = 0.001)
- Loss Function: Mean Squared Error (MSE)
- Batch Size: 128
- Epochs: 50

### Sınıflandırma Modelleri

#### Random Forest (RF)
- Autoencoder'dan çıkarılan 32 boyutlu özelliklerle eğitilmiştir
- 100 ağaç
- Gini impurity kriteri
- Maksimum derinlik: 10

#### MLP (Multilayer Perceptron)
- Ham piksel verisi (784 boyut) ile eğitilmiştir
- 3 gizli katman: [512, 256, 128] nöron
- ReLU aktivasyon fonksiyonu
- Dropout oranı: 0.2

## Sonuçlar (Results)

### Eğitim Süreci Analizi

#### Autoencoder Loss Değişimi
![Autoencoder Loss](results/plots/autoencoder_loss.png)

#### Sınıflandırıcı Performans Metrikleri

| Model                      | Doğruluk | Precision | Recall | F1-Score | Eğitim Süresi | Model Boyutu |
|---------------------------|----------|-----------|---------|-----------|---------------|--------------|
| Autoencoder + RandomForest | 91.5%    | 0.916     | 0.915   | 0.915    | 45 dk        | 2.3 MB      |
| MLP (Raw Input)           | 95.8%    | 0.958     | 0.958   | 0.958    | 60 dk        | 8.7 MB      |

### Karmaşıklık Matrisleri

#### Autoencoder + Random Forest
![Autoencoder RF CM](results/plots/confusion_matrix.png)

#### MLP (Raw Pixels)
![MLP CM](results/plots/confusion_matrix_mlp.png)


## Tartışma (Discussion)

### Model Karşılaştırması

#### Autoencoder + Random Forest Avantajları
1. **Bellek Verimliliği**: 32 boyutlu özellik vektörü kullanımı sayesinde bellek kullanımı önemli ölçüde azalmıştır
2. **Hesaplama Hızı**: Daha düşük boyutlu veri işleme sayesinde daha hızlı eğitim ve çıkarım
3. **Yorumlanabilirlik**: Düşük boyutlu özellik uzayı, veri yapısının daha kolay anlaşılmasını sağlar

#### MLP Avantajları
1. **Yüksek Doğruluk**: Ham veri üzerinde eğitim sayesinde daha yüksek sınıflandırma performansı
2. **Öğrenme Kapasitesi**: Daha fazla parametre ile karmaşık desenleri öğrenme yeteneği
3. **End-to-End Öğrenme**: Özellik çıkarımı ve sınıflandırmanın birlikte öğrenilmesi

### Sınırlamalar ve Gelecek Çalışmalar
1. **Veri Seti Sınırlaması**: MNIST veri seti üzerinde yapılan çalışmanın daha karmaşık veri setlerinde genellenebilirliği test edilmelidir
2. **Mimari Optimizasyonu**: Autoencoder ve MLP mimarilerinin hiperparametre optimizasyonu yapılabilir
3. **Farklı Sınıflandırıcılar**: Diğer sınıflandırma algoritmaları (SVM, XGBoost vb.) ile karşılaştırma yapılabilir

## Kaynaklar (References)

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

2. Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786), 504-507.

3. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

4. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

5. PyTorch Documentation. (2023). https://pytorch.org/docs/stable/index.html

6. Scikit-learn Documentation. (2023). https://scikit-learn.org/stable/

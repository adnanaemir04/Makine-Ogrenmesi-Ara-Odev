# Müşteri Ayrılma Tahmini - Makine Öğrenmesi Ara Ödevi

## Projenin Amacı

Bu projede müşterilerin abonelikten ayrılıp ayrılmayacağını (`churn`)
tahmin etmek amacıyla temel bir makine öğrenmesi sınıflandırma akışı
uygulanmıştır.

Proje; veri inceleme, eksik değer işlemleri, feature engineering,
train-validation-test ayrımı, One-Hot Encoding, StandardScaler,
model eğitimi ve sınıflandırma metrikleriyle değerlendirme adımlarını
içermektedir.

## Veri Seti

Projede 200 satırlık örnek bir müşteri veri seti kullanılmıştır.

Sütunlar:

- `yas`
- `gelir`
- `abonelik_suresi`
- `destek_talebi_sayisi`
- `sehir`
- `uyelik_tipi`
- `churn`

`churn` hedef değişkenidir:

- `0`: Müşteri kalır
- `1`: Müşteri ayrılır

## Uygulanan İşlemler

1. Veri pandas DataFrame olarak CSV'den okunmuştur.
2. İlk satırlar ve veri setinin boyutu incelenmiştir.
3. `churn` dağılımı incelenmiştir.
4. Eksik değer kontrolü yapılmıştır.
5. Sayısal eksik değerler medyan ile doldurulmuştur.
6. Kategorik eksik değerler mod ile doldurulmuştur.
7. `destek_talebi_var_mi` adlı yeni bir özellik üretilmiştir.
8. Veri `%60 train`, `%20 validation`, `%20 test` olarak ayrılmıştır.
9. Bölme işleminde `stratify` kullanılmıştır.
10. Kategorik değişkenlere One-Hot Encoding uygulanmıştır.
11. Sayısal değişkenlere StandardScaler uygulanmıştır.
12. Logistic Regression ve KNN modelleri eğitilmiştir.
13. Validation sonuçlarına göre model seçilmiştir.
14. Seçilen model test verisi üzerinde değerlendirilmiştir.
15. Confusion matrix, accuracy, precision, recall ve F1-score hesaplanmıştır.

## Modeller

### Logistic Regression

İkili sınıflandırma problemi için temel ve yorumlanabilir bir model
olarak kullanılmıştır.

### KNN

Müşterileri özellikleri arasındaki mesafeye göre sınıflandıran ikinci
model olarak kullanılmıştır. KNN mesafe hesabı yaptığı için sayısal
özelliklerde standardizasyon uygulanmıştır.

## Nasıl Çalıştırılır?

Python 3 yüklü olduğundan emin olun.

Gerekli paketleri kurun:

```bash
pip install -r requirements.txt
```

Daha sonra:

```bash
python customer_churn.py
```

komutunu çalıştırın.

## Sonuç

Program çalıştırıldığında iki modelin validation performansları
karşılaştırılmakta ve F1-score'u daha yüksek olan model seçilmektedir.

Seçilen model daha sonra test verisi üzerinde değerlendirilerek
accuracy, precision, recall, F1-score ve confusion matrix sonuçları
gösterilmektedir.

Sonuçlar veri setinin rastgele oluşturulması ve kullanılan
train/validation/test bölünmesine bağlı olarak program çalıştırıldığında
ekrana yazdırılmaktadır.

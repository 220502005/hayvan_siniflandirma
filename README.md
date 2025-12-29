#  Yapay Zeka Destekli Hayvan Görüntü Sınıflandırma

Bu projede, derin öğrenme tabanlı bir **Vision Transformer (ViT)** modeli kullanılarak hayvan görüntülerinin otomatik olarak sınıflandırılması amaçlanmıştır.  
Eğitilen model, **Streamlit** kullanılarak geliştirilen kullanıcı dostu bir web arayüzü ile entegre edilmiştir.

Proje, *Yapay Zeka ve Bulut Bilişim Teknolojileri* dersi kapsamında hazırlanmıştır.

---

##  Proje Amacı

Bu projenin temel amacı:

- Görüntü sınıflandırma problemini derin öğrenme yöntemleri ile çözmek  
- Önceden eğitilmiş bir Vision Transformer modelini kullanmak  
- Model çıktısını kullanıcı dostu bir web arayüzü üzerinden sunmaktır  

---

##  Kullanılan Teknolojiler ve Kütüphaneler

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Vision Transformer (ViT)**
- **Streamlit**
- **Matplotlib**
- **Scikit-learn**
- **PIL (Python Imaging Library)**

---

##  Veri Seti

Projede **Animals-10 Dataset** kullanılmıştır. link:

Veri seti:
- 10 farklı hayvan sınıfı içermektedir  
- Dengeli ve etiketli görsellerden oluşmaktadır  

Sınıflar:
- Köpek
- At
- Fil
- Kelebek
- Tavuk
- Kedi
- İnek
- Koyun
- Örümcek
- Sincap

Veriler, eğitim sürecinde:
- %80 eğitim
- %20 doğrulama (validation)

olacak şekilde ayrılmıştır.

---

##  Model Eğitimi

Model eğitimi **farklı bir bilgisayarda** gerçekleştirilmiştir.  (Ekran kartı sebebiyle)
Bu GitHub reposunda:

- Modelin eğitiminde kullanılan kodlar
- Eğitilmiş model dosyaları(kısıtlı)
- Eğitim sürecine ait performans çıktıları
  yer almaktadır.
Eğitilen modele drive linki üzerinden ulaşabilirsiniz:https://drive.google.com/file/d/1nbmmtKCvDanNl6eww4WaLRfFU5i1L6ab/view?usp=drive_link
### Kullanılan Model
- `google/vit-base-patch16-224`

### Eğitim Parametreleri
- Epoch Sayısı: **7**
- Batch Size: **16**
- Learning Rate: **2e-5**
- Optimizer: **AdamW**
- Kayıp Fonksiyonu: **Cross Entropy Loss**

---

##  Eğitim Sonuçları

Eğitim sürecinde elde edilen sonuçlara göre:

- Eğitim kaybı (Training Loss) epoch’lar ilerledikçe azalmıştır  
- Doğrulama kaybı (Validation Loss) düşük ve stabil seviyede kalmıştır  
- Doğrulama doğruluğu (Validation Accuracy) yaklaşık **%98.8** seviyesine ulaşmıştır  

Eğitim sürecine ait **Loss** ve **Accuracy** grafikleri aşağıdaki dosyada yer almaktadır:

📌 `eğitim_sonuçları.jpeg`
---

##  Web Arayüzü (Streamlit)

Geliştirilen Streamlit tabanlı web arayüzü sayesinde kullanıcılar:

- Bilgisayarlarından bir hayvan görseli yükleyebilir  
- Tek tıklama ile sınıflandırma tahmini alabilir  
- Tahmin edilen sınıfı ve güven oranını görüntüleyebilir  

Arayüz, sade ve kullanıcı dostu olacak şekilde tasarlanmıştır.

---

##  Uygulamayı Çalıştırma

Uygulamayı çalıştırmadan önce gerekli Python kütüphanelerinin kurulu olması gerekmektedir.

Gereklilikler:
- Python 3.9 veya üzeri
- Proje dosyaları
- Eğitilmiş modelin bulunduğu `data/` klasörü

Gerekli ortam sağlandıktan sonra uygulama aşağıdaki komut ile çalıştırılabilir:

```bash
streamlit run app.py

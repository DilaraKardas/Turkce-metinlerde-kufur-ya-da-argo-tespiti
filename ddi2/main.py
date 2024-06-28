import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# NLTK veri setlerinin indirilmesi
nltk.download('stopwords')
nltk.download('punkt')

def metin_temizle(metin):
    metin = re.sub(r'\s+', ' ', metin)  # Fazla boşlukları kaldırma
    metin = re.sub(r'[^\w\s]', '', metin)  # Noktalama işaretlerini kaldırma
    metin = metin.strip()  # Baş ve sondaki boşlukları kaldırma
    return metin

def kucuk_harf(metin):
    return metin.lower()

def durak_kelimeleri_kaldir(metin):
    durak_kelimeler = set(stopwords.words('turkish'))
    kelimeler = word_tokenize(metin)
    filtrelenmis_kelimeler = [kelime for kelime in kelimeler if kelime not in durak_kelimeler]
    return ' '.join(filtrelenmis_kelimeler)

def metin_on_isleme(metin):
    metin = metin_temizle(metin)
    metin = kucuk_harf(metin)
    metin = durak_kelimeleri_kaldir(metin)
    return metin

def main():
    # Kaggle veri setini okuma
    df = pd.read_csv(r"C:\Users\dlrka\PycharmProjects\ddi2\data\test.csv") 


    metinler = df['text'].tolist()
    etiketler = df['label'].tolist()  # 0: normal, 1: kötü

    # Metinleri ön işleme tabi tutma
    on_islenmis_metinler = [metin_on_isleme(metin) for metin in metinler]

    # Özellik çıkarımı (TF-IDF)
    tfidf_vectorizer = TfidfVectorizer()
    x = tfidf_vectorizer.fit_transform(on_islenmis_metinler)
    y = np.array(etiketler)

    # Eğitim ve test setlerine bölme
    x_egitim, x_test, y_egitim, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Random Forest modeli
    rf_modeli = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_modeli.fit(x_egitim, y_egitim)

    def metin_tahmini(metin):
        on_islenmis_metin = metin_on_isleme(metin)
        tfidf_metin = tfidf_vectorizer.transform([on_islenmis_metin])
        tahmin = rf_modeli.predict(tfidf_metin)

        if tahmin == 0:
            sonuc = "Normal Cümle"
        else:
            sonuc = "Küfür, Argo veya Şiddet İçeren Cümle"

        return sonuc

    print("\n\n\n\n\nTürkçe Metin Analizi Arayüzüne Hoş Geldiniz!")
    print("Çıkmak için 'q' tuşuna basın.")
    while True:
        kullanici_girdisi = input("Analiz etmek istediğiniz metni girin: ")
        if kullanici_girdisi.lower() == 'q':
            break
        sonuc = metin_tahmini(kullanici_girdisi)
        print(f"Tahmin Sonucu: {sonuc}")

main()
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as f
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

# Türkçe etiket eşleştirmesi
LABEL_MAP = {
    "cane": "Köpek",
    "cavallo": "At",
    "elefante": "Fil",
    "farfalla": "Kelebek",
    "gallina": "Tavuk",
    "gatto": "Kedi",
    "mucca": "İnek",
    "pecora": "Koyun",
    "ragno": "Örümcek",
    "scoiattolo": "Sincap"
}

@st.cache_resource
def load_model(model_dir):
    processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()
    return processor, model


class AnimalClassifierApp:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = f.softmax(outputs.logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)

        label_it = self.model.config.id2label[predicted_class.item()]
        label_tr = LABEL_MAP.get(label_it, label_it)

        return label_tr, confidence.item() * 100


def main():
    st.set_page_config(page_title="Hayvan Sınıflandırma", layout="wide")

    st.markdown("""
        <style>
            .title {
                text-align: center;
                font-size: 36px;
                font-weight: 600;
            }
            .subtitle {
                text-align: center;
                font-size: 18px;
                color: #666;
                margin-bottom: 30px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">Hayvan Sınıflandırma</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Bir hayvan fotoğrafı yükleyin ve tahmin sonucunu görün.</div>', unsafe_allow_html=True)
    st.divider()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "data")

    processor, model = load_model(model_dir)
    app = AnimalClassifierApp(processor, model)

    if "image" not in st.session_state:
        st.session_state.image = None
    if "result" not in st.session_state:
        st.session_state.result = None

    col_left, col_right = st.columns([1, 1.3])
    # Sayfa tasarımını blok olarak tasarlıyoruz.

    #sol taraf
    with col_left:
        st.subheader("📤 Görsel Yükleme")

        uploaded_file = st.file_uploader(
            "Bilgisayarınızdan bir hayvan görüntüsü seçin",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            st.session_state.image = Image.open(uploaded_file).convert("RGB")

        if st.button("Tahmin Et", use_container_width=True):
            if st.session_state.image is not None:
                label, confidence = app.predict(st.session_state.image)
                st.session_state.result = (label, confidence)
            else:
                st.warning("Lütfen önce bir görsel yükleyin.")

        if st.session_state.result:
            label, confidence = st.session_state.result
            st.success("Tahmin Yapıldı.")
            st.write(f"**Sınıf:** {label}")
            st.write(f"**Doğruluk Oranı:** %{confidence:.2f}")

    # sağ taraf
    with col_right:
        st.subheader("🖼️ Yüklenen Görsel")

        if st.session_state.image is None:
            st.info("Henüz bir görsel yüklenmedi.")
        else:
            st.image(
                st.session_state.image,
                width=500
                #sabit bir boyut belirliyoruz bu sayede tüm yüklenen görseller arayüzde düzgün ve aynı duruşta gözükebilir.
            )


if __name__ == "__main__":
    main()

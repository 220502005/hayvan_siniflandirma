import torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\HP\Desktop\sektorkampuste\aybukedataset\raw-img"
MODEL_NAME = "google/vit-base-patch16-224"
OUTPUT_DIR = "./model_aybuke"
EPOCHS = 7
BATCH_SIZE = 16 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nKullanılan Cihaz: {device.upper()}")
if device == "cuda":
    print(f"Ekran Kartı: {torch.cuda.get_device_name(0)}")

print(f"\nVeri seti okunuyor: {DATASET_PATH}")
ds = load_dataset("imagefolder", data_dir=DATASET_PATH)
ds = ds['train'].train_test_split(test_size=0.2, seed=42)

labels = ds['train'].features['label'].names
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

print(f"Toplam Sınıf Sayısı: {len(labels)}")

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

def transform(example_batch):
    # Bu fonksiyon artık remove_unused_columns=False sayesinde 'image' sütununu görebilecek.
    # Yine de dinamik olarak anahtarı bulalım:
    keys = list(example_batch.keys())
    image_key = next((k for k in keys if k != 'label'), None)
    
    if image_key is None:
        # Bu hata artık çıkmamalı, ama güvenlik için kalsın
        raise ValueError(f"HATA: Resim sütunu silinmiş! Gelen anahtarlar: {keys}")

    # Resimleri işlemden geçirip (pixel_values) oluşturuyoruz
    inputs = processor([x.convert("RGB") for x in example_batch[image_key]], return_tensors='pt')
    
    # Etiketleri de ekliyoruz
    inputs['labels'] = example_batch['label']
    
    return inputs

prepared_ds = ds.with_transform(transform)

# MODELİ HAZIRLAMA
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.to(device)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# EĞİTİM AYARLARI
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=2e-5,
    weight_decay=0.01,

    remove_unused_columns=False,
    
    eval_strategy="epoch",       
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                   
    dataloader_num_workers=0,    
    logging_steps=50,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    processing_class=processor, 
    compute_metrics=compute_metrics,
    
    # Verileri birleştirirken hata olmasın diye özel collator
    data_collator=None 
)

print("\nEğitim Başlıyor...")
trainer.train()


print(f"\nModel kaydediliyor: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# Grafikler
history = trainer.state.log_history
train_loss = [x['loss'] for x in history if 'loss' in x]
eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
eval_acc = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
if len(eval_loss) > 0:
    plt.plot(np.linspace(0, len(train_loss), len(eval_loss)), eval_loss, label='Validation Loss')
plt.title('Kayıp (Loss)')
plt.legend()

plt.subplot(1, 2, 2)
if len(eval_acc) > 0:
    plt.plot(eval_acc, label='Validation Accuracy', color='green')
plt.title('Doğruluk (Accuracy)')
plt.legend()

plt.savefig('egitim_sonuclari.png')
print("Grafik kaydedildi.")
plt.show()
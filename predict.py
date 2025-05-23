# predict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib  # label encoder 로드용

# 모델, 토크나이저 로드
model_path = "./fp_model_v02"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# LabelEncoder 로드
label_encoder = joblib.load("./label_encoder.pkl")


def predict_fp_type(texts):
    model.eval()
    encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=64, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=1)

    predicted_labels = label_encoder.inverse_transform(preds.numpy())
    return predicted_labels
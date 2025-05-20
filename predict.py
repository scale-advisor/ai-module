# predict
import torch
import numpy as np

from .config import max_length
from .model import load_tokenizer, load_model

def predict_fp_type(model, tokenizer, texts, label_encoder):
    model.eval()
    encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=1)

    return label_encoder.inverse_transform(preds.numpy())

    # # json으로 출력하려면 tolist 사용
    # predicted_labels = label_encoder.inverse_transform(preds.numpy())
    # return predicted_labels.tolist()
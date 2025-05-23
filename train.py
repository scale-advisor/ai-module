# train
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset import FPDataset
from model import load_model, load_tokenizer, create_trainer
from config import max_length, random_seed

def train_model(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["FP 유형"])
    # df = df[df["FP 유형"] != "E"]

    X = df["단위프로세스"].tolist()
    y = df["FP 유형"].tolist()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_labels = len(label_encoder.classes_)

    tokenizer = load_tokenizer()
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=random_seed, stratify=y_encoded)

    train_dataset = FPDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = FPDataset(X_val, y_val, tokenizer, max_length)

    model = load_model(num_labels)
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset)

    trainer.train()

    return trainer, label_encoder
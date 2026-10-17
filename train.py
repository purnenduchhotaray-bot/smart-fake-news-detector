import os
import re
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.lower().strip()
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(cleaned)

def load_or_create_sample_data():
    os.makedirs("dataset", exist_ok=True)
    true_path = "dataset/True.csv"
    fake_path = "dataset/Fake.csv"

    if os.path.exists(true_path) and os.path.exists(fake_path):
        print("[+] Loading local datasets...")
        df_true = pd.read_csv(true_path)
        df_fake = pd.read_csv(fake_path)
        df_true["label"] = 1
        df_fake["label"] = 0
        df = pd.concat([df_true, df_fake], ignore_index=True)
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    else:
        print("[!] No CSV found in 'dataset/'. Creating demo training data...")
        sample_data = {
            "content": [
                "Government approves new solar subsidy scheme for middle class homeowners across the country.",
                "Breaking: Secret alien spacecraft discovered beneath national monument military confirms coverup.",
                "Central Bank maintains key benchmark interest rate at 6.5 percent following fiscal meeting.",
                "Drink raw onion juice twice a day to cure stage 4 cancer instantly in 48 hours.",
                "Space agency successfully places high resolution weather monitoring satellite into low Earth orbit.",
                "Leaked memo reveals government plans to replace currency notes with digital tokens tomorrow morning."
            ],
            "label": [1, 0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(sample_data)
        
    return df

def main():
    os.makedirs("models", exist_ok=True)
    df = load_or_create_sample_data()

    print("[+] Cleaning text data...")
    df["clean_content"] = df["content"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("[+] Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[+] Training model...")
    pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5)
    pac.fit(X_train_vec, y_train)

    preds = pac.predict(X_test_vec)
    print("\n================ EVALUATION ================")
    print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

    joblib.dump(pac, "models/fake_news_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("[+] Models saved inside 'models/' folder.")

if __name__ == "__main__":
    main()
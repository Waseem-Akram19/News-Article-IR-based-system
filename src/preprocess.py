
import argparse
import pandas as pd
import nltk
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.strip()

def preprocess_row(title, content):
    text = (title or "") + " " + (content or "")
    text = clean_text(text)

    tokens = nltk.word_tokenize(text)

    stop = set(stopwords.words('english'))
    lem = WordNetLemmatizer()

    tokens = [lem.lemmatize(t) for t in tokens if t not in stop and len(t) > 1]

    return tokens

def main(input_csv, out_file):
    df = pd.read_csv(input_csv, encoding="ISO-8859-1", on_bad_lines="skip")

    docs = []
    meta = []

    for i, row in df.iterrows():
        title = row.get("Heading", "") or ""
        content = row.get("Article", "") or ""

        tokens = preprocess_row(title, content)
        docs.append(tokens)

        meta.append({
            "id": int(i),
            "title": title
        })

    joblib.dump({"docs": docs, "meta": meta}, out_file)
    print(f"Saved {len(docs)} processed docs to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.input, args.out)

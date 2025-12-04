
import argparse
import joblib
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

def docs_join(docs):
    return [" ".join(tokens) for tokens in docs]

def index_bm25(docs, out_path):
    bm25 = BM25Okapi(docs)
    joblib.dump(bm25, out_path)
    print("BM25 index saved at:", out_path)

def index_tfidf(docs, out_path):
    vec = TfidfVectorizer()
    X = vec.fit_transform(docs)
    sparse.save_npz(out_path, X)
    joblib.dump({'vectorizer': vec}, out_path + ".meta")
    print("TF-IDF index saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["bm25", "tfidf"], default="bm25")
    args = parser.parse_args()

    data = joblib.load(args.input)
    docs = data['docs']

    if args.method == "bm25":
        index_bm25(docs, args.out)
    else:
        joined_docs = docs_join(docs)
        index_tfidf(joined_docs, args.out)

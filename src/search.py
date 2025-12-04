import argparse
import joblib
from rank_bm25 import BM25Okapi
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """Clean and normalize text - same as preprocess.py"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.strip()

def preprocess_query(query):
    """Preprocess query using the SAME pipeline as documents"""
    text = clean_text(query)
    tokens = nltk.word_tokenize(text)
    
    stop = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    
    tokens = [lem.lemmatize(t) for t in tokens if t not in stop and len(t) > 1]
    
    return tokens

def bm25_search(bm25, meta, query_tokens, k, verbose=False):
    """Search using BM25 with optional debug output"""
    if verbose:
        print(f"\n[DEBUG] Query tokens: {query_tokens}")
    
    scores = bm25.get_scores(query_tokens)
    
    if verbose:
        print(f"[DEBUG] Max score: {scores.max():.4f}")
        print(f"[DEBUG] Non-zero scores: {np.sum(scores > 0)}")
    
    idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in idx]

def tfidf_search(vec_meta, X, query_text, k, verbose=False):
    """Search using TF-IDF with optional debug output"""
    vec = vec_meta['vectorizer']
    
    # Preprocess query for TF-IDF as well
    query_tokens = preprocess_query(query_text)
    query_processed = " ".join(query_tokens)
    
    if verbose:
        print(f"\n[DEBUG] Processed query: {query_processed}")
    
    qv = vec.transform([query_processed])
    sims = cosine_similarity(qv, X).flatten()
    
    if verbose:
        print(f"[DEBUG] Max similarity: {sims.max():.4f}")
        print(f"[DEBUG] Non-zero similarities: {np.sum(sims > 0)}")
    
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search documents using BM25 or TF-IDF")
    parser.add_argument("--method", choices=["bm25", "tfidf"], default="bm25",
                       help="Retrieval method to use")
    parser.add_argument("--index", required=True, 
                       help="Path to index file")
    parser.add_argument("--processed", required=True,
                       help="Path to processed documents file")
    parser.add_argument("--query", required=True,
                       help="Search query")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of results to return")
    parser.add_argument("--verbose", action="store_true",
                       help="Print debug information")
    args = parser.parse_args()

    # Load processed data
    data = joblib.load(args.processed)
    meta = data["meta"]

    print(f"\nSearching for: '{args.query}'")
    print(f"Method: {args.method.upper()}")
    print(f"Top {args.k} results:\n")

    if args.method == "bm25":
        # Load BM25 index
        bm25 = joblib.load(args.index)
        
        # Preprocess query using SAME method as documents
        query_tokens = preprocess_query(args.query)
        
        res = bm25_search(bm25, meta, query_tokens, args.k, args.verbose)
    else:
        # Load TF-IDF sparse matrix and metadata
        X = sparse.load_npz(args.index + ".npz")
        vec_meta = joblib.load(args.index + ".meta")
        
        res = tfidf_search(vec_meta, X, args.query, args.k, args.verbose)

    # Display results
    for rank, (doc_id, score) in enumerate(res, 1):
        print(f"{rank}. [Doc {doc_id}] score={score:.4f}")
        print(f"   {meta[doc_id]['title'][:100]}")
        print()
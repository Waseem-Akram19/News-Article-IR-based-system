import argparse
import joblib
import time
import numpy as np
from scipy import sparse
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# Import search functions
sys.path.append(os.path.dirname(__file__))
from search import preprocess_query, bm25_search, tfidf_search

# Sample test queries with expected relevant terms
# Updated based on actual dataset content (2015 news: oil, stocks, cricket, Pakistan)
TEST_QUERIES = [
    {
        "query": "oil price market",
        "description": "Oil prices and energy markets"
    },
    {
        "query": "pakistan government economic",
        "description": "Pakistan government and economic policy"
    },
    {
        "query": "stock market trading",
        "description": "Stock market and trading news"
    },
    {
        "query": "cricket england test match",
        "description": "Cricket matches and tournaments"
    },
    {
        "query": "hong kong asia market",
        "description": "Asian financial markets"
    },
    {
        "query": "karachi sindh transport",
        "description": "Karachi and Sindh regional news"
    },
    {
        "query": "saudi arabia oil production",
        "description": "Saudi oil production and OPEC"
    },
    {
        "query": "india pakistan relations",
        "description": "India-Pakistan bilateral issues"
    },
    {
        "query": "dollar rupee currency exchange",
        "description": "Currency exchange and forex"
    },
    {
        "query": "security terrorism attack",
        "description": "Security and terrorism news"
    }
]

def calculate_precision_at_k(results, k, relevant_terms):
    """
    Calculate Precision@K based on presence of relevant terms in titles.
    This is a simple heuristic evaluation since we don't have ground truth.
    """
    if k > len(results):
        k = len(results)
    
    top_k = results[:k]
    relevant_count = 0
    
    for doc_id, score, title in top_k:
        # Check if any relevant term appears in title (case insensitive)
        title_lower = title.lower()
        if any(term in title_lower for term in relevant_terms):
            relevant_count += 1
    
    return relevant_count / k if k > 0 else 0

def calculate_mrr(results, relevant_terms):
    """
    Calculate Mean Reciprocal Rank - position of first relevant result
    """
    for rank, (doc_id, score, title) in enumerate(results, 1):
        title_lower = title.lower()
        if any(term in title_lower for term in relevant_terms):
            return 1.0 / rank
    return 0.0

def evaluate_query(method, index_data, processed_data, query_text, k=10):
    """Evaluate a single query and return metrics"""
    meta = processed_data["meta"]
    
    start_time = time.time()
    
    if method == "bm25":
        bm25 = index_data
        query_tokens = preprocess_query(query_text)
        raw_results = bm25_search(bm25, meta, query_tokens, k, verbose=False)
    else:  # tfidf
        X, vec_meta = index_data
        raw_results = tfidf_search(vec_meta, X, query_text, k, verbose=False)
    
    query_time = time.time() - start_time
    
    # Format results with titles
    results = [(doc_id, score, meta[doc_id]['title']) for doc_id, score in raw_results]
    
    return results, query_time

def run_evaluation(method, index_path, processed_path, output_file=None):
    """Run complete evaluation on test queries"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATION REPORT - {method.upper()} Retrieval")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading indexes and data...")
    processed_data = joblib.load(processed_path)
    
    if method == "bm25":
        index_data = joblib.load(index_path)
    else:
        X = sparse.load_npz(index_path + ".npz")
        vec_meta = joblib.load(index_path + ".meta")
        index_data = (X, vec_meta)
    
    print(f"Total documents: {len(processed_data['meta'])}\n")
    
    # Evaluation metrics storage
    all_precisions = []
    all_mrr = []
    all_times = []
    
    results_text = []
    
    # Evaluate each test query
    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        desc = test["description"]
        
        print(f"\nQuery {i}/{len(TEST_QUERIES)}: '{query}'")
        print(f"Description: {desc}")
        
        # Get relevant terms from query for evaluation
        relevant_terms = query.lower().split()
        
        # Run search
        results, query_time = evaluate_query(method, index_data, processed_data, query, k=10)
        
        # Calculate metrics
        p_at_5 = calculate_precision_at_k(results, 5, relevant_terms)
        p_at_10 = calculate_precision_at_k(results, 10, relevant_terms)
        mrr = calculate_mrr(results, relevant_terms)
        
        all_precisions.append(p_at_10)
        all_mrr.append(mrr)
        all_times.append(query_time)
        
        print(f"  Precision@5:  {p_at_5:.3f}")
        print(f"  Precision@10: {p_at_10:.3f}")
        print(f"  MRR:          {mrr:.3f}")
        print(f"  Query Time:   {query_time*1000:.2f} ms")
        
        # Store detailed results
        results_text.append(f"\n{'='*60}")
        results_text.append(f"Query {i}: {query}")
        results_text.append(f"{'='*60}")
        for rank, (doc_id, score, title) in enumerate(results[:5], 1):
            results_text.append(f"{rank}. [Score: {score:.4f}] {title[:80]}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Mean Precision@10: {np.mean(all_precisions):.3f} (±{np.std(all_precisions):.3f})")
    print(f"Mean MRR:          {np.mean(all_mrr):.3f} (±{np.std(all_mrr):.3f})")
    print(f"Mean Query Time:   {np.mean(all_times)*1000:.2f} ms (±{np.std(all_times)*1000:.2f} ms)")
    print(f"Total Queries:     {len(TEST_QUERIES)}")
    
    # Save detailed results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"EVALUATION RESULTS - {method.upper()}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Mean Precision@10: {np.mean(all_precisions):.3f}\n")
            f.write(f"Mean MRR: {np.mean(all_mrr):.3f}\n")
            f.write(f"Mean Query Time: {np.mean(all_times)*1000:.2f} ms\n\n")
            f.write("\n".join(results_text))
        print(f"\nDetailed results saved to: {output_file}")
    
    return {
        'mean_precision': np.mean(all_precisions),
        'mean_mrr': np.mean(all_mrr),
        'mean_time': np.mean(all_times),
        'all_precisions': all_precisions,
        'all_mrr': all_mrr,
        'all_times': all_times
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IR system")
    parser.add_argument("--method", choices=["bm25", "tfidf"], required=True)
    parser.add_argument("--index", required=True, help="Path to index file")
    parser.add_argument("--processed", required=True, help="Path to processed data")
    parser.add_argument("--output", help="Output file for detailed results")
    args = parser.parse_args()
    
    run_evaluation(args.method, args.index, args.processed, args.output)
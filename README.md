# CS516 HW3: Local Information Retrieval System

A complete, locally-hosted information retrieval system implementing BM25 and TF-IDF ranking algorithms for searching news articles. Built for CS 516 (Information Retrieval and Text Mining) at ITU.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Evaluation Results](#evaluation-results)
- [Examples](#examples)
- [License](#license)

## 🎯 Overview

This project implements a complete IR pipeline that:
1. Preprocesses raw text documents with NLP techniques
2. Builds searchable indexes using BM25 and TF-IDF algorithms
3. Executes fast keyword searches with ranked results
4. Evaluates retrieval quality using standard IR metrics

**Total Documents:** 2,692 news articles  
**Supported Methods:** BM25 (probabilistic) and TF-IDF (vector space)  
**Average Query Time:** ~10ms (after warm-up)

## ✨ Features

- **Dual Indexing:** Both BM25 and TF-IDF implementations
- **Robust Preprocessing:** Tokenization, stopword removal, lemmatization
- **Fast Retrieval:** Sub-second query response times
- **Comprehensive Evaluation:** Precision@K, MRR metrics on 10 test queries
- **Fully Local:** No cloud dependencies, runs entirely on your machine
- **Reproducible:** Complete pipeline with all dependencies specified

## 💻 System Requirements

- **Python:** 3.8 or higher
- **OS:** Windows, macOS, or Linux
- **Memory:** Minimum 4GB RAM
- **Disk Space:** ~100MB for data and indexes

## 🚀 Installation

### Step 1: Clone the Repository

```bash
https://github.com/Waseem-Akram19/News-Article-IR-based-system.git
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- rank_bm25
- scikit-learn
- numpy
- scipy
- pandas
- joblib
- nltk

### Step 4: Download NLTK Data

The system will automatically download required NLTK resources on first run, but you can manually trigger it:

```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

## 📊 Dataset

Place your dataset file `Articles.csv` in the `data/` directory. The CSV should have:
- **Heading:** Article title
- **Article:** Article content

Dataset format: ISO-8859-1 encoded CSV

## 🔧 Usage

### Complete Pipeline (First Time Setup)

Run these commands in order:

#### 1. Preprocess Documents
```bash
python src/preprocess.py --input data/Articles.csv --out data/processed.pkl
```
**Output:** `data/processed.pkl` containing tokenized documents

#### 2. Build BM25 Index
```bash
python src/indexer.py --input data/processed.pkl --out indexes/bm25.pkl --method bm25
```
**Output:** `indexes/bm25.pkl` (~15 MB)

#### 3. Build TF-IDF Index
```bash
python src/indexer.py --input data/processed.pkl --out indexes/tfidf.pkl --method tfidf
```
**Output:** `indexes/tfidf.pkl.npz` and `indexes/tfidf.pkl.meta` (~2.5 MB total)

### Search Queries

#### BM25 Search
```bash
python src/search.py --method bm25 --index indexes/bm25.pkl --processed data/processed.pkl --query "pakistan government economic" --k 10
```

#### TF-IDF Search
```bash
python src/search.py --method tfidf --index indexes/tfidf.pkl --processed data/processed.pkl --query "oil price market" --k 10
```

#### Verbose Mode (with debug information)
```bash
python src/search.py --method bm25 --index indexes/bm25.pkl --processed data/processed.pkl --query "cricket england" --k 5 --verbose
```

### Run Evaluation

#### Evaluate BM25
```bash
python src/evaluation.py --method bm25 --index indexes/bm25.pkl --processed data/processed.pkl --output outputs/evaluation/evaluation_bm25.txt
```

#### Evaluate TF-IDF
```bash
python src/evaluation.py --method tfidf --index indexes/tfidf.pkl --processed data/processed.pkl --output outputs/evaluation/evaluation_tfidf.txt
```

## 📁 Project Structure

```
cs516_hw3/
├── data/
│   ├── Articles.csv          # Raw dataset (place here)
│   └── processed.pkl         # Preprocessed documents
├── indexes/
│   ├── bm25.pkl             # BM25 index
│   ├── tfidf.pkl.npz        # TF-IDF sparse matrix
│   └── tfidf.pkl.meta       # TF-IDF vectorizer metadata
├── outputs/
│   ├── evaluation/          # Evaluation reports
│   │   ├── evaluation_bm25.txt
│   │   └── evaluation_tfidf.txt
│   └── search_results/      # Sample search outputs
│       ├── bm25_oil.txt
│       ├── tfidf_pakistan.txt
│       └── ...
├── src/
│   ├── preprocess.py        # Document preprocessing
│   ├── indexer.py           # Index building (BM25/TF-IDF)
│   ├── search.py            # Search interface
│   └── evaluation.py        # Evaluation framework
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🏗️ System Architecture

```
Raw CSV → Preprocessing → Indexing (BM25/TF-IDF) → Search → Evaluation
```

**Preprocessing Pipeline:**
1. Text cleaning (lowercase, special char removal)
2. Tokenization (NLTK word_tokenize)
3. Stopword removal (English stopwords)
4. Lemmatization (WordNet lemmatizer)

**Indexing:**
- **BM25:** Probabilistic ranking with default parameters (k1=1.5, b=0.75)
- **TF-IDF:** Vector space model with cosine similarity

**Search:**
- Queries undergo same preprocessing as documents
- Top-k results ranked by score
- Sub-linear query time with cached indexes

## 📈 Evaluation Results

### Test Queries (n=10)
1. Oil prices and energy markets
2. Pakistan government and economic policy
3. Stock market and trading news
4. Cricket matches and tournaments
5. Asian financial markets
6. Karachi and Sindh regional news
7. Saudi oil production and OPEC
8. India-Pakistan bilateral issues
9. Currency exchange and forex
10. Security and terrorism news

### Performance Metrics

| Metric | BM25 | TF-IDF |
|--------|------|--------|
| **Mean Precision@10** | 0.780 ± 0.227 | 0.780 ± 0.223 |
| **Mean MRR** | 0.875 ± 0.256 | 1.000 ± 0.000 |
| **Mean Query Time** | 308.38 ms | 314.13 ms |
| **First Query Time** | ~3000 ms | ~3000 ms |
| **Subsequent Queries** | 2-12 ms | 9-12 ms |

**Key Findings:**
- Both methods achieve ~78% precision
- TF-IDF shows perfect MRR (first result always relevant)
- First query slow due to NLTK loading; subsequent queries very fast
- Both struggle with broad, abstract queries (e.g., "security terrorism")

## 💡 Examples

### Example 1: Oil Price Query

**Query:** `"oil price market"`

**BM25 Results:**
```
1. [Doc 2156] score=7.8234
   Oil falls on profit taking after earlier rally

2. [Doc 1842] score=7.6721
   Oil prices surge after Saudi cuts production

3. [Doc 945] score=7.4532
   Global oil markets stabilize amid supply concerns
```

### Example 2: Pakistan Government Query

**Query:** `"pakistan government economic"`

**TF-IDF Results:**
```
1. [Doc 2605] score=0.8456
   PML N govt turned around Pakistans economy in 3 years PM

2. [Doc 459] score=0.7823
   WB president for Pakistan to invest in people to boost economy

3. [Doc 1375] score=0.7234
   Pak security team to submit its report on Wednesday
```

### Example 3: Verbose Debug Mode

```bash
python src/search.py --method bm25 --index indexes/bm25.pkl --processed data/processed.pkl --query "cricket match" --k 3 --verbose
```

**Output:**
```
Searching for: 'cricket match'
Method: BM25
Top 3 results:

[DEBUG] Query tokens: ['cricket', 'match']
[DEBUG] Max score: 8.9234
[DEBUG] Non-zero scores: 456

1. [Doc 1203] score=8.9234
   England win test match against Pakistan by 7 wickets

2. [Doc 867] score=8.4521
   Cricket World Cup: India defeats Australia in semifinal

3. [Doc 2301] score=8.1203
   Pakistan cricket team announces squad for upcoming series
```

## 🔬 Advanced Usage

### Custom Test Queries

Edit `TEST_QUERIES` in `src/evaluation.py` to add your own evaluation queries:

```python
TEST_QUERIES = [
    {
        "query": "your custom query",
        "description": "Description of query intent"
    },
    # ... more queries
]
```

### Adjust Result Count

Change the `--k` parameter to retrieve more/fewer results:

```bash
# Get top 20 results
python src/search.py --method bm25 --index indexes/bm25.pkl --processed data/processed.pkl --query "technology innovation" --k 20
```

### Save Search Results to File

```bash
python src/search.py --method bm25 --index indexes/bm25.pkl --processed data/processed.pkl --query "stock market" --k 10 > my_results.txt
```

## 📝 Notes

- **First run** takes longer due to NLTK data download (~50MB)
- **Indexes** must be rebuilt if preprocessing changes
- **Relevance judgments** in evaluation are based on term matching heuristics (no ground truth available)
- **Query preprocessing** must match document preprocessing for best results

## 🤝 Contributing

This is an academic assignment, but feedback is welcome:
1. Open an issue for bugs or suggestions
2. Follow the existing code style
3. Add tests for new features

## 📄 License

This project is submitted as coursework for CS 516 at ITU. Please do not copy or reuse without permission as per academic integrity policies.

## 👨‍💻 Author

**Waseem Akram**  
Roll Number: MSCS23003  
Course: CS 516 - Information Retrieval and Text Mining  
Institution: Information Technology University (ITU)  
Semester: Fall 2025

## 🙏 Acknowledgments

- Dr. Ahmad Mustafa (Course Instructor)
- rank-bm25 library by Dorian Brown
- scikit-learn and NLTK communities

---

**Last Updated:** November 30, 2025  
**Assignment:** Information Retrieval System Design

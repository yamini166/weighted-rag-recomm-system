"""
Retrieval functions for dynamic weighted RAG thesis.
Author: Yamini Uma Rani M V
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# ---------- Label relevance ----------

label_keywords = {
    'bug': ['bug', 'error', 'crash', 'fail', 'broken', 'not working',
            'incorrect', 'wrong', 'issue', 'problem'],
    'enhancement': ['feature', 'request', 'improve', 'add', 'support',
                    'allow', 'enable', 'implement'],
    'documentation': ['doc', 'documentation', 'readme', 'guide',
                      'tutorial', 'example', 'explain'],
    'good first issue': ['beginner', 'simple', 'easy', 'starter'],
    'help wanted': ['help', 'assist', 'support', 'need']
}

def get_label_relevance_score(query, label):
    if pd.isna(label):
        return 0.0
    query_lower = query.lower()
    label_lower = str(label).lower()
    if label_lower in query_lower:
        return 1.0
    for category, keywords in label_keywords.items():
        if category in label_lower:
            if any(kw in query_lower for kw in keywords):
                return 0.7
    return 0.1

# ---------- Author authority ----------

def make_authority_lookup(author_df):
    return dict(zip(author_df['author'], author_df['authority_score']))

def get_author_authority(author, authority_lookup):
    if pd.isna(author):
        return 0.0
    return authority_lookup.get(str(author), 0.05)

# ---------- Baseline retrieval ----------

def baseline_retrieve(query, model, index, df, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'rank': i + 1,
            'idx': int(idx),
            'title': df['title'].iloc[idx],
            'label': df['labels_0_name'].iloc[idx],
            'author': df['answers_0_author'].iloc[idx],
            'combined_text': df['combined_text'].iloc[idx]
        })
    return results

# ---------- Dynamic weighted retrieval ----------

def dynamic_weighted_retrieve(query, model, index, df, authority_lookup, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding.astype(np.float32), top_k * 3)
    scored_results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]
        semantic_score = 1 / (1 + dist)
        label_score = get_label_relevance_score(query, row.get('labels_0_name'))
        author_scores = [get_author_authority(row.get(f'answers_{i}_author'), authority_lookup) for i in range(6)]
        best_author_score = max(author_scores) if author_scores else 0.0
        final_score = (
            0.5 * semantic_score +
            0.3 * best_author_score +
            0.2 * label_score
        )
        scored_results.append({
            'idx': int(idx),
            'title': row['title'],
            'label': row.get('labels_0_name'),
            'best_author': max(
                [(row.get(f'answers_{i}_author'),
                  get_author_authority(row.get(f'answers_{i}_author'), authority_lookup))
                 for i in range(6)],
                key=lambda x: x[1]
            )[0],
            'semantic_score': round(semantic_score, 4),
            'label_score': round(label_score, 4),
            'author_score': round(best_author_score, 4),
            'final_score': round(final_score, 4),
            'combined_text': str(row['combined_text'])
        })
    scored_results = sorted(scored_results, key=lambda x: x['final_score'], reverse=True)
    return scored_results[:top_k]
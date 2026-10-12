import numpy as np
from sentence_transformers import SentenceTransformer

faqs = [
    "What is the company leave encashment and rollover policy?",
    "How do I submit proof for medical insurance reimbursement?",
    "Where can I find guidelines regarding the work from home rules?",
    "What is the standard notice period for voluntary resignation?",
    "I cannot log in to my company account due to invalid credentials.",
    "How to request software license access for Figma and JetBrains?",
    "My laptop keyboard and monitor need a hardware replacement.",
    "Steps to connect to company private internal network through VPN.",
    "When is monthly salary credited to employee bank accounts?",
    "How can I download Form 16 and annual tax deduction slips?",
    "Why was professional tax deducted from my final payslip?",
    "What is the deadline for submitting investment proofs for TDS?",
]


model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = model.encode(faqs)


def cosine_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def keyword_search(query):
  tokens = query.lower().split()
  results = []
  for faq in faqs:
    score = sum(1 for tok in tokens if tok in faq.lower())
    if score > 0:
      results.append((faq, score))
  results.sort(key=lambda x: x[1], reverse=True)
  return results[0][0] if results else "No keyword match found"


def semantic_search(query, top_k=1):
  q_emb = model.encode(query)
  scores = [cosine_similarity(q_emb, doc_emb) for doc_emb in faq_embeddings]
  ranked_indices = np.argsort(scores)[::-1]
  return faqs[ranked_indices[0]]



queries = [
    "I cannot log in",
    "when is salary credited",
    "work from home rules",
]

print(" Task 2 Comparison Results ")
for q in queries:
  print(f"\nQuery: '{q}'")
  print(f"Keyword Search Result  : {keyword_search(q)}")
  print(f"Semantic Search Result : {semantic_search(q)}")
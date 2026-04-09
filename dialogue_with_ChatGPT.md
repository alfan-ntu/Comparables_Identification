# AI-Assisted Transfer Pricing Comparable Identification (Technical Discussion)

---

# 1. Core Operation: Semantic Similarity Ranking

## Problem / Question
How can AI be used to automatically rank potential comparable companies based on business descriptions exported from databases such as Osiris or OneSource?

## Key Explanation
Similarity Ranking is the core operation using a pretrained sentence embedding model, e.g. 'all-mpnet-base-v2'. This model converts company descriptions into vectors. Cosine similarity between the tested party description and candidate company descriptions provides a semantic similarity score.

These scores are then used to rank potential comparables.

## Example Code

```python
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIG ----------
EXCEL_FILE = "comparables.xlsx"
TESTED_PARTY_FILE = "tested_party.txt"
TEXT_COLUMN = "description"
MODEL_NAME = "all-mpnet-base-v2"
# ----------------------------

df = pd.read_excel(EXCEL_FILE)

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Column '{TEXT_COLUMN}' not found in Excel file")

df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)

with open(TESTED_PARTY_FILE, "r", encoding="utf-8") as f:
    tested_party_text = f.read().strip()

model = SentenceTransformer(MODEL_NAME)

tested_embedding = model.encode([tested_party_text], normalize_embeddings=True)

company_embeddings = model.encode(
    df[TEXT_COLUMN].tolist(),
    normalize_embeddings=True
)

similarities = cosine_similarity(tested_embedding, company_embeddings)[0]

df["similarity_score"] = similarities

df_ranked = df.sort_values(by="similarity_score", ascending=False)

df_ranked.to_excel("ranked_comparables.xlsx", index=False)
```

### Explanation
- Business descriptions are converted into embeddings.
- Cosine similarity measures semantic closeness.
- `[0]` extracts the first row from the similarity matrix so that a 1-D array of similarity scores can be assigned to the DataFrame.

## Notes / Insights
Embedding similarity measures semantic similarity rather than exact keyword overlap.

---

# 2. Understanding the `[0]` Index in Cosine Similarity

## Problem / Question
Why is `[0]` used after calling `cosine_similarity()`?

## Key Explanation
The cosine similarity function returns a matrix.

If:
- Tested party embedding shape = `(1, 768)`
- Candidate embeddings shape = `(150, 768)`

Then the output is:

```
(1, 150)
```

A matrix containing similarity between the tested party and each candidate.

## Example Code

```python
similarities = cosine_similarity(tested_embedding, company_embeddings)[0]
```

## Explanation
`[0]` extracts the first row of the matrix so the result becomes a 1-D array with one similarity score per company.

## Notes / Insights
Without `[0]`, pandas would receive a 2-D array and fail when assigning scores to a column.

---

# 3. Tips - Part 1
## Data Cleaning Before Embedding

### Problem / Question
Why to remove missing descriptions before embedding?

### Rationale
Embedding models require text input. Missing descriptions cannot be processed.

### Example Code

```python
df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
```

### Explanation of the Example Code
- `dropna()` removes companies with missing descriptions.
- `reset_index()` ensures consistent row indexing after removal.

### Notes / Insights
Companies lacking activity descriptions should generally be excluded from comparability analysis.

---

## DataFrame Sorting and the `df_ranked` Object

### Problem / Question
What is `df_ranked` and how is it used?

### Key Explanation
`df_ranked` is a pandas DataFrame created by sorting the original dataset based on similarity scores.

### Example Code

```python
df_ranked = df.sort_values(by="similarity_score", ascending=False)
```

### Explanation of the Example Code
The rows are sorted so the most similar companies appear first.

### Notes / Insights
Sorting does not modify the original DataFrame unless explicitly assigned.

---

# 4. Explainability Layers for AI Ranking

AI ranking should include explainability. Three layers were discussed:

| Layer | Purpose |
|------|------|
| Layer 1 | Keyword / phrase overlap |
| Layer 2 | Contrastive summary |
| Layer 3 | Red-flag reasoning |

---

## 5.1 Layer 1 – Keyword and Phrase Overlap

### Goal
TF-IDF is used to identify overlapping phrases between the tested party and candidate descriptions. This provides transparent evidence why a candidate appears similar.

### Problem / Question
How to identify overlapping terms between the tested party and candidate descriptions? 

=> To extract shared key phrases using TfidfVectorizer(), Term Frequency-Inverse Document Frequency, to measure a word's relevance to a document within a collection(corpus) 

### Example Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_shared_terms(text_a, text_b, top_n=8):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=2000
    )

    tfidf = vectorizer.fit_transform([text_a, text_b])
    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf.toarray()
    shared_scores = scores[0] * scores[1]

    top_indices = shared_scores.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices if shared_scores[i] > 0]
```

### Explanation
TF-IDF identifies shared phrases between descriptions. 
TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic reflecting a word's importance to a document within a collection (corpus) 

### Notes / Insights
Shared terms provide interpretable evidence explaining similarity.

---

## 5.2 Layer 2 - Contrastive Summary
### Goal
A Contrastive Summary is a structured explanation which combines similarity scores, shared terms and red flags to produce a human-readable comparison. 

### Problem / Question
How can explanations be generated comparing a candidate company with the tested party?

### Example Code

```python
def generate_contrast_summary(row, tested_party_profile):
    summary_parts = []

    if row["similarity_score"] > 0.80:
        summary_parts.append(
            "The candidate shows strong overall similarity in business activities."
        )
    elif row["similarity_score"] > 0.70:
        summary_parts.append(
            "The candidate shows moderate similarity in core activities."
        )
    else:
        summary_parts.append(
            "The candidate shows limited similarity in overall business profile."
        )

    if row.get("shared_terms"):
        summary_parts.append(
            f"Common activity descriptors include: {row['shared_terms']}."
        )

    if "IP_HEAVY" in row["flags"]:
        summary_parts.append(
            "The candidate appears to own or license valuable intangibles, which may indicate higher functional and risk intensity."
        )

    if "MANUFACTURING" in row["flags"]:
        summary_parts.append(
            "Manufacturing activities are indicated, which differ from routine service functions."
        )

    if "REAL_ESTATE" in row["flags"]:
        summary_parts.append(
            "Real estate activities are referenced, suggesting functional non-comparability."
        )

    return " ".join(summary_parts)
```

### Explanation
Structured logic converts similarity metrics and flags into readable explanations.

## 5.3 Layer 3 – Red-Flag Reasoning

### Goal
A Red-Flag Reasoning operation is a rule-based detection which identifies potential comparability risks (e.g. IP ownership, manufacturing activities...)

### Problem / Question
How to flag potentially problematic comparables?

### Example Code

```python
def detect_red_flags(text):
    text = text.lower()
    flags = []
    if any(k in text for k in ["proprietary", "patent", "license", "royalty"]):
        flags.append("IP_HEAVY")
    if any(k in text for k in ["manufacture", "factory", "production"]):
        flags.append("MANUFACTURING")
    if any(k in text for k in ["platform", "saas", "subscription software"]):
        flags.append("SOFTWARE_PRODUCT")
    return ", ".join(flags)
```

### Explanation
Rule-based keyword checks identify functional differences such as IP ownership.

### Notes / Insights
These rules are deterministic and easily explainable during audits.

---

# Tips - Part 2
## Tagging Companies by Functional Category

### Problem / Question
How to assign tags to companies based on business descriptions?

### Example Code

```python
def tag_company(text):
    text = text.lower()
    tags = []

    if any(k in text for k in ["patent", "proprietary", "license", "royalty"]):
        tags.append("IP_HEAVY")

    if any(k in text for k in ["manufacture", "factory", "production"]):
        tags.append("MANUFACTURING")

    if any(k in text for k in ["real estate", "property management", "property investment"]):
        tags.append("REAL_ESTATE")

    return tags
```

### Explanation of the Example Code
Tags classify companies based on business characteristics.

### Notes / Insights
Returning a list of tags is preferable to returning a string.

---

## Adjusting Similarity Scores Using Penalty Rules

### Problem / Question
How to incorporate economic logic into AI similarity scoring?

### Example Code

```python
PENALTIES = {
    "IP_HEAVY": 0.10,
    "MANUFACTURING": 0.15,
    "REAL_ESTATE": 0.40
}
```

```python
def apply_penalty(row):
    score = row["similarity_score"]
    for flag in row["flags"]:
        score -= PENALTIES.get(flag, 0)
    return max(score, 0)
```

### Explanation
Penalty values reduce the similarity score based on detected flags.

---

## Sorting After Adjusted Scores
### Problem / Question
Similarity scores reevaluated and rearranged after penalty rules applied

### Example Code
```python
df_ranked["adjusted_score"] = df_ranked.apply(apply_penalty, axis=1)
df_ranked = df_ranked.sort_values(by="adjusted_score", ascending=False)
```

### Explanation of the Example Code
Adjusted scores determine the final ranking after rule-based adjustments.

---

## Debugging `KeyError: "adjusted_score"`

### Example

```python
print(df.columns)
```

### Explanation
Check whether the column exists in the DataFrame being sorted.

---

## Fixing Keyword Matching Errors

### Problem
Substring matching may produce false classifications such as `"production"` vs `"post-production"`.

### Example Code

```python
import re

MANUFACTURING_PATTERN = re.compile(
    r"\b(manufacture|manufacturing|factory|production)\b",
    re.IGNORECASE
)
```

### Explanation
`\b` enforces word boundaries to prevent substring errors.

---

# 11. Transfer Pricing Functional Taxonomy

Suggested classification categories:

- ROUTINE_SERVICE_PROVIDER  
- CONTRACT_MANUFACTURER  
- FULL_RISK_MANUFACTURER  
- LIMITED_RISK_DISTRIBUTOR  
- FULL_RISK_DISTRIBUTOR  
- IP_OWNER  
- REAL_ESTATE_ENTITY  
- FINANCIAL_INSTITUTION  
- HOLDING_COMPANY  

---

# 12. Structured Taxonomy Engine

```python
TP_TAXONOMY = {
    "CONTRACT_MANUFACTURER": re.compile(
        r"\b(contract manufacturing|production facility|assembly operations|oem production)\b",
        re.IGNORECASE
    ),
    "ROUTINE_SERVICE_PROVIDER": re.compile(
        r"\b(back-office support|bpo services|administrative services|payroll services|accounting services)\b",
        re.IGNORECASE
    ),
    "IP_OWNER": re.compile(
        r"\b(owns intellectual property|licenses technology|royalty income|develops proprietary software|research and development)\b",
        re.IGNORECASE
    ),
    "REAL_ESTATE_ENTITY": re.compile(
        r"\b(real estate development|property investment|property leasing|property management services)\b",
        re.IGNORECASE
    ),
    "FINANCIAL_INSTITUTION": re.compile(
        r"\b(banking services|insurance underwriting|asset management|securities brokerage|loan origination)\b",
        re.IGNORECASE
    ),
}
```

---

# 13. Tagging Function Using Taxonomy

```python
def tag_company(text):
    tags = []
    for category, pattern in TP_TAXONOMY.items():
        if pattern.search(text):
            tags.append(category)
    return tags
```

---

# 14. Exclusion Categories

```python
EXCLUSION_CATEGORIES = [
    "REAL_ESTATE_ENTITY",
    "FINANCIAL_INSTITUTION"
]
```

---

# Notes / Insights

- AI should assist screening, not replace professional judgment.
- Semantic similarity identifies candidate comparables efficiently.
- Rule-based tagging ensures explainability.
- Economic logic must be layered on top of AI scoring.
- Deterministic rules improve audit defensibility.

---

# Optional Improvements

Possible future enhancements:

- Use proportional penalties instead of subtraction
- Cluster companies by functional similarity
- Build a Streamlit review interface
- Add LLM-generated report explanations
- Calibrate penalty values using historical TP decisions
- Use phrase-based detection instead of single keywords
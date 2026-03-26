# Semantic Similarity

`semantic_similarity.py` provides a simple function to compare two news headlines and return a semantic similarity score.

## Function

- `semantic_similarity(headline_a, headline_b) -> float`
- Returns a score in the range `[0, 1]`
  - Closer to `1`: more similar meaning
  - Closer to `0`: less similar meaning

## Quick Usage

Run the demo in the file:

```powershell
python .\similarity\semantic_similarity.py
```

Use it in code:

```python
from similarity.semantic_similarity import semantic_similarity

score = semantic_similarity(
    "Government unveils new climate plan for coastal cities",
    "New policy aims to protect shoreline towns from climate change",
)
print(score)
```

## Notes

- First run may download the default embedding model.
- If model loading fails, the script falls back to a simple TF-IDF similarity method.
- Both headlines must be non-empty strings.


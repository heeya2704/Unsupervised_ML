# Section D — AI-Augmented Learning
**Assessment:** M7-A1 | Food Delivery K-Means + Elbow + PCA

---

## 1. Exact prompt given to the AI tool

**Tool:** Cursor (Grok) — used as the AI coding assistant.

**Prompt:**

```
Write a Python program for a food delivery platform that:
- Creates a sample customer dataset with at least 150 rows and 3 numeric
  features: monthly_orders, avg_spend, avg_rating
- Performs K-Means clustering
- Uses the Elbow Method to choose the optimal number of clusters and plots
  the inertia curve
- Prints each cluster's centroid values and the count of customers in
  that cluster
- Visualises the final clusters in 2D using PCA, each cluster a distinct
  colour, with a legend

Keep the code short and beginner-friendly.
```

---

## 2. AI original code vs corrected version

- Original (with typical AI issues left in): `ai_original.py`
- Corrected version: `ai_corrected.py`

The original script is what a short “beginner-friendly” AI draft often produces:
it clusters on **unscaled** features, starts the elbow loop at **k=1**, omits
`random_state`, and plots PCA axes without **PC1/PC2** labels.

---

## 3. What changed and why (3–4 lines)

The original code clustered on raw `monthly_orders`, `avg_spend`, and `avg_rating`.
Spend is in hundreds of rupees while rating is 1–5, so K-Means was dominated by
spend. I added `StandardScaler` before clustering and PCA. I also set
`random_state=42` (and `n_init=10`) so runs are reproducible, changed the elbow
loop from `k=1..10` to `k=2..9` (k=1 is a useless baseline and some sklearn
settings warn or behave poorly at k=1), and labelled the scatter axes as PC1
and PC2 as required.

# Section A — Concept Application
**Assessment:** M7-A1 | Data Science | Modules 5–7

---

## Question 1 — Delivery times and central tendency (Module 5)

The **mean is misleading** because it is pulled upward by a small number of extreme late deliveries (90+ minutes). Most orders finish in 25–30 minutes, so the arithmetic average is not the time a typical customer actually experiences.

**Recommended measure: the median** (the middle value when times are ordered). The median is resistant to outliers, so a handful of 90-minute trips barely moves it.

**Skewness:** This distribution is **right-skewed (positively skewed)**. A long right tail of very large times makes **mean > median**. The bulk of the mass sits on the left (around 25–30 minutes), with a thin tail stretching toward 90+ minutes. For a right-skewed variable, the median (or a trimmed mean) is the honest “typical” number to show on the app.

---

## Question 2 — Fraud flag and Bayes’ theorem (Module 5)

Define events:

- \(F\): order is fraudulent  
- \(L\): order is legitimate  
- \(+\): model flags the order as fraud  

**Prior:** \(P(F) = 0.05\), so \(P(L) = 0.95\).

**Likelihood (true positive rate):** \(P(+ \mid F) = 0.90\).

**False positive rate:** \(P(+ \mid L) = 0.08\).

**Total probability of a flag:**

\[
P(+) = P(+ \mid F)P(F) + P(+ \mid L)P(L) = (0.90)(0.05) + (0.08)(0.95) = 0.045 + 0.076 = 0.121
\]

**Posterior (Bayes):**

\[
P(F \mid +) = \frac{P(+ \mid F)P(F)}{P(+)} = \frac{0.045}{0.121} \approx 0.372
\]

A flagged order is genuinely fraudulent only about **37.2%** of the time.

**Why this surprises stakeholders:** The model “catches 90% of fraud,” which sounds highly reliable. Because fraud is rare (5%) and 8% of the much larger legitimate population is also flagged, **most flags come from legitimate orders**. High sensitivity plus a low base rate produces many false alarms; the posterior is far below 90%.

---

## Question 3 — Preprocessing and the 97% vs 61% accuracy gap (Module 6)

Two distinct preprocessing problems:

**1. Missing values in `restaurant_rating` (numeric).**  
If missing rows are dropped only on the training set, or filled with a statistic computed incorrectly (or left as NaN and handled inconsistently), the model never learns a stable representation of that feature. A worse but common issue is **imputing after the train/test split incorrectly**, or using a model that silently treats missingness in a way that does not generalise. The required fix is **mean (or median) imputation fitted on the training set only**, then applied to the test set. Unresolved missingness can make the model latch onto accidental patterns in whichever rows happen to be complete in training, which fail on new data.

**2. Categorical text features (`payment_method`, `cuisine_type`) not encoded (or encoded incorrectly).**  
Tree models and most sklearn estimators need numeric input. If categories are label-encoded as arbitrary integers, the model can treat “Wallet = 2 > Cash = 0” as a false ordinal relationship and overfit those codes. The correct technique is **one-hot encoding** (`pd.get_dummies` or `OneHotEncoder` with categories learned from train). Without proper encoding, high training accuracy can come from memorising category–label coincidences that do not hold on the test set.

Together with the 97%/61% gap, these issues sit on top of classic **overfitting**: the model has memorised training quirks (missingness patterns, category codes) instead of a general cancellation rule. Encoding and imputation, plus regularisation / simpler models / more data, close that gap.

*(Additional related cause of this exact gap: no train/test-aware preprocessing and possible leakage — but the two dataset problems named above are missing numeric values and unencoded categoricals.)*

---

## Question 4 — Imbalanced fraud classifiers (Module 6)

**Why accuracy is misleading:** With 49,500 legitimate and 500 fraudulent orders, a dummy model that always predicts “not fraud” already scores **99% accuracy**. Model A’s 99.1% can be almost entirely correct “not fraud” predictions while missing most actual fraud.

Finance cares about **minimising undetected fraud** → they want to catch as many of the 500 fraud cases as possible. That is **recall (sensitivity) for the fraud class**:

\[
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
\]

where TP = fraud correctly flagged, FN = fraud missed.

**Model B** identifies 420 of 500 fraud orders, so

\[
\text{Recall}_B = \frac{420}{500} = 0.84
\]

Model A’s recall is not given, but 99.1% overall accuracy implies it can miss a large share of the 500 fraud cases and still look excellent. **Adopt Model B.** Lower overall accuracy is acceptable because the metric that matches the business cost (missed fraud) is fraud **recall**, not accuracy. (Precision / false-alarm rate can be monitored secondarily.)

---

## Question 5 — K-Means vs DBSCAN for 80,000 customers (Module 7)

**Data characteristics:** ~3–4 compact groups of similar density, very few genuine outliers, features are order frequency, spend, and delivery window.

**K-Means** assumes roughly spherical, similar-sized clusters and a chosen \(k\). That matches “3–4 distinct groups with similar density.” It also scales well to 80,000 points.

**DBSCAN** finds arbitrary shapes and treats sparse points as noise. It is less suitable here: density is similar across groups (so one `eps`/`min_samples` pair may merge or split poorly), outliers are rare (DBSCAN’s noise handling is not the main need), and density-based tuning on 80k rows is harder than picking \(k \approx 3\)–\(4\).

**Recommendation: K-Means** (try \(k = 3\) and \(k = 4\), confirm with elbow/silhouette).

**Limitation for marketing:** K-Means **forces every customer into a cluster** and assumes **convex, similar-variance blobs**. Borderline customers and mixed behaviours get assigned anyway. Centroids are not “personas” until someone interprets them; a customer near a boundary may not fit the campaign designed for that centroid. K-Means is also sensitive to feature scaling and the arbitrary choice of \(k\).

---

## Question 6 — PCA before K-Means (Module 7)

**Purpose:** 35 correlated features make Euclidean distance noisy (curse of dimensionality), slow, and uninterpretable. PCA compresses the data into a smaller set of uncorrelated axes (principal components) that capture most of the variance, so K-Means runs on a cleaner, lower-dimensional space and 2D/3D plots become possible.

**How PCA works:** It finds orthogonal directions of **maximum variance**. The first PC is the direction of greatest spread; each later PC captures remaining variance, uncorrelated with previous PCs. Highly correlated original features collapse onto shared components, so redundant dimensions are dropped while the dominant patterns (spend vs frequency vs discount usage, etc.) are kept.

**Trade-off:** Keep more components → retain more information, but clusters stay high-dimensional and harder to plot/interpret; keep too few → you throw away structure that actually separates segments.

**Explained variance ratio:** Plot cumulative explained variance vs number of components. A common rule is to keep enough PCs to reach **~80–90%** cumulative variance (or look for an elbow). Use that \(n\) for K-Means; use the first two PCs only for visualisation, not necessarily as the full clustering space if two PCs explain too little.

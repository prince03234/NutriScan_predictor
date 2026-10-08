# NutriScan — Do Health Claims Predict Better Food?

Indian packaged food is sold on words: *Digestive*, *Multigrain*, *Baked*, *High Protein*,
*Sugar Free*. This project tests whether those words carry real nutritional information,
or whether they're decoration — the **health halo effect**.

**Short answer: the claims are mostly honest. One is not.**

---

## Key findings

| | |
| **The halo hypothesis was refuted.** Across 49 claim × category comparisons, **45** showed products carrying a health claim scoring *higher* than peers in the same category. | |
| **One claim is systematically misleading.** `gluten free` biscuits score **10.1/100** against **33.0** for other biscuits (n=40). Excluding the dominant brand, still **16.8** (n=24). | |
| **Good announces itself; bad does not.** Claim words predict *good* products (`sugar free` −2.06, `protein` −1.69). Brand names predict *bad* ones (`parle` +1.16, `britannia` +1.04). Nobody prints "high sugar" on a packet. | |
| **The project's own LLM pipeline is biased.** The claim↔quality link is ~40% stronger in rows where an LLM estimated nutrition (+15.0) than in rows read from real labels (+10.9). | |

### Why `gluten free` is the exception

It's an **allergen** claim, not a nutrition claim. It says nothing about sugar, fat or
processing — but shoppers read it as "healthy." The products confirm it:
`Gluten Free Butter Cookies`, `Kaju Butter Cookies – Gluten Free`, `Butter Sweet Cookies`.


## The dataset

4,338 packaged-food products scraped from **Flipkart** and **JioMart**, each scored 0–100
by the NutriScan engine using NOVA food-processing classification and WHO/EFSA nutrient
thresholds. After removing 3 pet-food products and 686 duplicates: **3,649 products**,
33.2% of which score below 50.


## Method

### 1. Target leakage was ruled out first

The obvious model — predict `quality_score` from the nutrition columns — is circular.
`100 − Σ(penalties) + positive_buffer` reproduces the score at **r = 0.986**. Such a model
re-derives the scoring formula and learns nothing about food.

**Only marketing-visible fields are used:** `name`, `brand`, `pack_size`. These are written
by a packaging team and are causally upstream of the nutrition label.

### 2. Every comparison is made *within* a food category

Claim words cluster by food type. `digestive` appears almost only on biscuits; `organic`
mostly on staples. Since biscuits average **30** and staples **89**, a dataset-wide
comparison measures *"is this a biscuit?"*, not *"is this claim honest?"*

The sign actually reverses:

```
"digestive", measured across the whole dataset :  54.0  vs  61.4   →  looks BAD
"digestive", measured within biscuits only     :  56.6  vs  29.5   →  is GOOD
```

That's Simpson's paradox. Controlling for category is the methodological core of this project.

### 3. The category variable had to be built

The source data has no category column. One was derived from product names by rule-based
keyword matching across 22 categories, using **whole-word** regex rather than substring
matching — plain substring search misfiled 26 *Seviyan* products as snacks via the
fragment `"sev"`, and `"nut"` would have captured 410 products including *Nutro Cookies*.

Claim words are deliberately **excluded** from category rules. Bare `protein` was removed
as a category keyword: it's the variable under study, and bucketing every protein product
together would have destroyed the comparison.

**Accuracy: 95% on 100 hand-labelled products**, which surfaced 4 classification bugs.
After fixing them, **50/50 on a fresh held-out sample** never used for debugging.
Both label sets are in the repo as evidence.


## Results

### Does the name predict quality?

| Model | PR-AUC | Note |

| Random guessing | 0.332 | base rate |
| Food category only | 0.655 | one-hot, 22 categories |
| **Product name (TF-IDF + logistic regression)** | **0.888** | 86.2% accuracy vs 66.8% majority-class |
| Category + name | 0.875 | no gain — the name already encodes category |

### Is it just memorising brands?

| Cross-validation | PR-AUC |

| Random split (brands may appear in both halves) | 0.888 (±0.014) |
| **Brand-grouped** `GroupKFold` (no brand in both) | **0.835** (±0.037) |

A drop of **0.053**. Small but real: the model leans on brand slightly, but most of its
skill survives on brands it has never seen.

### Does the signal survive within a single food type?

| Category | n | base rate | PR-AUC | lift |

| breakfast_cereal | 343 | 0.15 | 0.768 | **+0.617** |
| spices_masala | 258 | 0.28 | 0.736 | **+0.461** |
| snacks | 445 | 0.23 | 0.657 | **+0.424** |
| noodles | 256 | 0.43 | 0.842 | **+0.408** |
| condiments_sauces | 231 | 0.56 | 0.868 | **+0.309** |
| biscuits | 442 | 0.76 | 0.975 | **+0.220** |

Even when every product in the comparison is a biscuit, the words still predict quality.



## Limitations

- **2,245 of 3,649 rows** have `data_source = "gemini-estimated"` — an LLM estimated the
  nutrition rather than reading a label. All headline findings were re-checked on the
  1,404 real-label rows and survive, though weaker (+10.9 vs +15.0).
- **Ceiling effect.** Biscuits average 29 (70 points of headroom); staples average 89
  (11 points). Raw score differences are **not** comparable across categories.
- **Known categoriser bug:** the keyword `wafer` pulls ~18 banana/potato wafers into
  `biscuits`, inflating that category's mean by roughly 1 point.
- `pulses_legumes` (1% bad) and `staples_grains` (6%) have too few minority-class products
  for reliable within-category modelling and are excluded from that table.



## Repository

```
nutriscan.py             Shared logic: categoriser, data loading, halo index
proj1.ipynb              Full analysis notebook
app.py                   Streamlit dashboard
nutriscan_products.csv   4,338 scraped products
category_check.csv       100 hand-labelled products (categoriser audit, 95%)
category_check_v2.csv    50 held-out hand-labelled products (post-fix, 50/50)
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py          # dashboard
jupyter notebook proj1.ipynb  # full analysis
```


**Prince Kumar** — MCA, National Institute of Technology Karnataka, Surathkal

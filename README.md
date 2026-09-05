# NutriScan Predictor

**Can the marketing on a food packet tell you whether the food is actually bad for you?**

Indian packaged food is sold on words like *Digestive*, *Multigrain*, *Baked*, *High Protein*
and *Sugar Free*. This project tests whether those words carry real nutritional information,
or whether they are decoration — a documented consumer-psychology phenomenon known as the
**health halo effect**.

---

## The dataset

4,338 packaged food products scraped from **Flipkart** and **JioMart**, each scored 0–100 by
the [NutriScan](https://github.com/prince03234) analysis engine using NOVA food-processing
classification and WHO/EFSA nutrient thresholds.

| | |
|---|---|
| Products | 4,338 |
| Sources | Flipkart (3,408), JioMart (807), OpenFoodFacts (123) |
| Columns | 43 |
| Target | `quality_score` (0–100), `tier` (best / healthy / not_healthy / hazardous) |
| Products scoring below 50 | 34.9% |

The columns fall into three groups:

- **Marketing** — `name`, `brand`, `pack_size`, `seller_label`
- **Nutrition** — `sugars_g`, `fat_g`, `sodium_mg`, `protein_g`, `fiber_g`, …
- **Verdict** — `quality_score`, `tier`, and the penalty components behind them

---

## Method

### Predicting from marketing text only

The obvious model — predicting `quality_score` from the nutrition columns — is **target
leakage**. The score is calculated from those columns by a deterministic formula:
`100 − Σ(penalties) + positive_buffer` reproduces the score with a correlation of **0.986**.
Such a model would re-derive the scoring formula rather than learn anything about food.

This project therefore uses **only the marketing fields**, which are written by a brand's
packaging team and are causally upstream of the nutrition label. No leakage is possible.

### Why every comparison is made within a food category

Claim words are not spread evenly across food types — *Digestive* appears almost exclusively
on biscuits, *Organic* mostly on staples. Since biscuits and staples have very different
baseline scores, a naive dataset-wide comparison measures **what kind of food it is**, not
**whether the claim is honest**. Comparing across categories produces sign reversals
(Simpson's paradox).

All claim analysis is therefore done **within category only** — biscuit vs. biscuit,
snack vs. snack.

### Building the category variable

The source data has no category column, so one was derived from product names using
rule-based keyword matching across 22 food categories.

- Whole-word (regex boundary) matching, not substring matching — plain substring search
  misfiled 26 *Seviyan* products as snacks via the fragment `"sev"`, and would have
  captured 410 products including *Nutro Cookies* via `"nut"`.
- Category keywords describe **what the food is**, never how it is marketed. Claim words
  such as *protein* are deliberately excluded from category rules, since they are the
  variable under study.
- Uncategorised products: **13 of 4,338** (0.3%).

**Measured accuracy: 95% on 100 randomly sampled, hand-labelled products (≈ ±4%).**
The labelled sample is included as `category_check.csv`. Remaining errors are mostly
genuinely ambiguous products (e.g. peanut butter — a nut, or a spread?).

---

## Data quality notes

- **2,540 of 4,338 rows** have `data_source = "gemini-estimated"`, meaning nutrition values
  were estimated rather than read from a label. Findings are re-checked against the
  `gemini-extracted` and `openfoodfacts` subsets, where values come from real labels.
- **686 duplicate** `(name, brand)` rows are removed before any train/test split, so the
  same product cannot appear in both.
- **3 pet-food products** (dog treats and supplements) were found in the scrape and excluded.

---

## Status

- [x] Data exploration and leakage audit
- [x] Category variable derived and accuracy measured
- [ ] Within-category claim analysis (the Halo Index)
- [ ] Baseline classifier: TF-IDF on product name → Logistic Regression
- [ ] Brand-grouped cross-validation, to separate language signal from brand memorisation
- [ ] Findings write-up

---

## Repository

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook proj1.ipynb

Author
Prince Kumar — MCA, National Institute of Technology Karnataka, Surathkal

## Notes on what I wrote and why

**I did not put any results in it.** The "Status" checklist shows two items done and four to go. That's deliberate — a README claiming findings your notebook doesn't contain is the fastest way to lose credibility if someone opens the notebook. Add the Halo Index results once you've actually run Step 2, and I'll help you write that section properly.

**The leakage paragraph is the most valuable thing in this file.** Most beginner portfolio repos say "achieved 94% accuracy" on a leaky target. Yours explicitly explains a trap you found and avoided. Anyone technical reading it will notice.

**Two things to change before pasting:**
- The NutriScan link points at your profile — swap it for the actual NutriScan repo URL if it's public.
- If the `95%` or the counts change after you re-run with the four keyword fixes, update the numbers. Don't let the README drift from the notebook.

Then the usual three beats:

```bash
git add README.md
git commit -m "Add project README"
git push

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from nutriscan import load_data, build_halo, train_model, explain_prediction

st.set_page_config(page_title="Nutriscan - Health Halo", layout="wide")

POSITIVE, NEGATIVE = "#2a78d6", "#e34948"
GRID, AXIS, INK = "#e5e4e0", "#c9c8c3", "#52514e" 

@st.cache_data
def get_data():
    df = load_data()
    return df, build_halo(df)

df, halo = get_data()

st.title("Do health claims predict better food?")
st.caption("3,649 Products scrapped from Flipkart and Jiomart . "
           "claims compared within food category")

c1, c2, c3 = st.columns(3)
c1.metric("Products analysed", f"{len(df):,}")
c2.metric("Claim comparisons", len(halo))
c3.metric("Claims that go with better products",
          f"{(halo['difference']>0).mean():.0%}")

st.info("**Finding:** the health halo effect did not appear. "
        f"{(halo['difference']>0).sum()} of {len(halo)} comparisons show products "
        "carrying a health claim scoring **higher** than peers in the same category.")
st.subheader("The Halo Index")
chosen = st.multiselect("Filter by category", sorted(halo["category"].unique()))
view = halo[halo["category"].isin(chosen)] if chosen else halo

st.dataframe(
    view.round(1),
    use_container_width=True, hide_index = True,
    column_config = {"difference": st.column_config.NumberColumn(
            "difference", help = "Positive = claim goes with better products")},
)

st.subheader("By category")
counts = halo["category"].value_counts()
panel_cats = counts[counts >= 6].index.tolist()

fig, axes = plt.subplots(2, 2, figsize = (11,8), sharex = True)
for ax, cat in zip(axes.flat, panel_cats):
    d = halo[halo["category"] == cat].sort_values("difference")
    ax.barh(d["claim"], d["difference"],
            color = [NEGATIVE if v < 0 else POSITIVE for v in d["difference"]], height=0.62)
    ax.axvline(0, color=INK, linewidth=1.2, zorder=3)
    ax.set_title(cat.replace("_", " "), loc = "left", fontsize=11, fontweight = "bold")
    ax.grid(axis = "x", color = GRID, linewidth = 0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=9, length=0, colors=INK)
for ax in axes.flat[len(panel_cats):]:
    ax.set_visible(False)
fig.supxlabel("Quality-score difference vs. other products in the same category",
              fontsize = 9, color =INK)
fig.tight_layout()
st.pyplot(fig)

st.caption("Panels share an x-axis. Categories with low baselines scores "
           "(biscuits ≈ 29) have more room to improve than high-baseline ones"
           "(staples ≈ 89), so bar length is not comparable across panels.")

st.divider()
st.subheader("Try the model")
st.caption("Type any product name. The model has never seen it, it only reads the word")

@st.cache_resource
def get_model():
    return train_model(df)

model = get_model()

name = st.text_input("Product name", value = "Britannia Gluten Free Butter Cookies", placeholder="e.g. Organic Ragi Millet Atta 1kg")

if name.strip():
    prob = model.predict_proba([name])[0][1]

    left, right = st.columns([1,2])
    left.metric("Chance this score poorly", f"{prob:.0%}")
    rows = explain_prediction(model, name)
    if rows:
        exp = pd.DataFrame(rows, columns=["word", "push"])
        exp["direction"] = exp["push"].apply(lambda v: "→ bad" if v > 0 else "→ good")
        right.dataframe(exp.round(3), hide_index = True, use_container_width = True)
    else:
        right.info("None of these words appeared often enough in the training data.")
st.caption("Each word's push = its TF-IDF weight in this name x the coefficient the model "
           "learned for it. Positive pushes toward 'scores poorly'.")   

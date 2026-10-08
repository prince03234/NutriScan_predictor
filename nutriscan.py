import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

CATEGORY_KEYWORDS = [
    ("pet_food", ["dog","cat food","puppy","kitten","drools","firstbark"]),
    ("noodles", ["noodle","pasta","vermicelli","seviyan","semiya","sevai","macaroni","spaghetti","penne","hakka","ramen","chowmein","maggi"]),
    ("biscuits", ["biscuit","cookie","cracker","rusk","khari","marie","wafer","toastea","oreo","hide and seek","bourbon","nutrichoice","nutri choice","good day","krackjack","monaco","milk bikis","dark fantasy","diskette"]),
    ("bakery", ["bread","bun","cake","pastry","muffin","croissant","brownie","donut","doughnut","pav bun","toast","baguette","pizza base","spring roll","dough"]),
    ("breakfast_cereal", ["muesli","granola","cornflake","corn flake","cereal","oats","oat","porridge","daliya","dalia","poha","upma","froot loop","fruit loop","special k","chocos","kellogg"]),
    ("health_supplements", ["protein powder","protein bar","protein shake","protein isolate","energy bar","whey","chyawanprash","chyavanprash","chyawanplus","ashwagandha","shilajit","supplement","gainer","nutrition","health drink","spirulina","apple cider","horlicks","bournvita","bourn vita","boost","complan","protinex","pediasure","herbalife","malt","shake","ensure","isolate","psyllium","isabgol","giloy","gokhru","gokshura","restora","groviva","threptin","babyvita","womens plus","fiber complex","enteral"]),
    ("snacks", ["banana wafer","wafer masala", "banana chips", "potato chips","chips","chipps","namkeen","bhujia","bhujiya","mixture","sev","popcorn","makhana","khakhra","khakhara","fryums","papad","nachos","puff","mathri","chakli","murukku","snack","crisps","stix","kurkure","bhel","chiwda","chivda","khatta","farsan","chavanu","navratna","misal","lachha","french fries","aloo tikki","crunchem","simply salted","act ii","caramel","potato","ring","taco","bingo","sattu","kuch-kuch","tana-bana"]),
    ("chocolate_candy", ["chocolate","candy","toffee","lollipop","gummies","gummy","choco","marshmallow","eclairs","cocoa","dairy milk","five star","perk","munch","kitkat","gems","bar","truffle","compound","nutties","pepero"]),
    ("sweets_desserts", ["laddu","ladoo","laddoo","barfi","burfi","halwa","soan papdi","soanpapdi","gulab jamun","rasgulla","mysore pak","peda","kaju katli","jalebi","sweets","mithai","dessert","custard","jelly","ice cream","cornetto","kulfi","rabri","motichur","kala jamun","goli"]),
    ("condiments_sauces", ["sauce","peanut butter", "nut butter", "almond butter","ketchup","pickle","achar","chutney","mayonnaise","mayo","vinegar","paste","puree","dip","dressing","spread","jam","marmalade","syrup","seasoning","kissan","schezwan","conserve","gulkand","tahina","stock cube"]),
    ("dry_fruits_nuts", ["almond","badam","cashew","kaju","raisin","kishmish","pista","pistachio","walnut","akhrot","anjeer","fig","apricot","dry fruit","dryfruit","nuts","peanut","groundnut","khajur","prune","cranberry","hazelnut","dates","trail mix","nariyal","desiccated coconut"]),
    ("seeds", ["seed","chia","flax","alsi","sabja"]),
    ("spices_masala", ["masala","turmeric","turmaric","haldi","cumin","jeera","coriander","corainder","dhaniya","dhania","chilli","chili","chilly","mirch","pepper","clove","laung","cardamom","elaichi","cinnamon","dalchini","asafoetida","hing","fenugreek","methi","bay leaf","tej patta","spice","garam","sambar","rasam","kitchen king","curry powder","ajwain","saunf","fennel","nutmeg","star anise","kalonji","amchur","amchor","chaat","salt","mustard","sarso","rai","rosemary","herb"]),
    ("beverages", ["juice","drink","cola","soda","tea","coffee","squash","lemonade","smoothie","water","beverage","kombucha","milkshake","buttermilk","coconut water","thandai","paper boat","aamras","thums up"]),
    ("dairy", ["milk","curd","dahi","paneer","cheese","cheddar","butter","yogurt","yoghurt","lassi","cream","khoya","condensed","tofu"]),
    ("oils_fats", ["oil","ghee","vanaspati","margarine"]),
    ("sweeteners", ["sugar","jaggery","gur","honey","stevia","sweetener","misri"]),
    ("pulses_legumes", ["dal","dahl","daal","chana","channa","rajma","moong","masoor","toor","tur","urad","lobia","chickpea","kabuli","matar","pea","lentil","soybean","soya","bean","gram"]),
    ("staples_grains", ["atta","aata","flour","maida","rice","besan","sooji","suji","rava","ravva","quinoa","millet","ragi","bajra","jowar","wheat","barley","jau","corn","makai","sabudana","idli","idly","dosa","dosai","batter","india gate","rozana","grain","sushi"]),
    ("ready_to_eat", ["ready to eat","ready-to-eat","instant","mix","gravy","curry","biryani","pulao","soup","dhokla","khaman","bhaji","paratha","roti","tikka","momo","uttappam","mock meat","mock chicken","chicken"]),
    ("fresh_produce", ["onion","tomato","vegetable","fruit","banana","apple","mango","lemon","nimbu","garlic","ginger","carrot","spinach","capsicum","lauki"]),
    ("gift_combo", ["gift","combo","hamper","pack of","assorted","pcs","box"]),
]

def build(keywords):
    # words of 6+ letters: match any ending  ("biscuit" also catches "biscuits")
    # short words: must be the whole word    ("sev" must NOT catch "seviyan")
    parts = [re.escape(k) + (r"\w*" if len(k) >= 6 else r"e?s?\b") for k in keywords]
    return re.compile(r"\b(?:" + "|".join(parts) + r")", re.I)

PATTERNS = [(cat, build(kws)) for cat, kws in CATEGORY_KEYWORDS]

def find_category(name):
    n = str(name).lower()
    for cat, pattern in PATTERNS:
        if pattern.search(n):      # first match wins — order matters!
            return cat
    return "others"


CLAIM_WORDS = ["sugar free", "no added sugar", "high protein", "protein",
               "multigrain", "whole wheat", "atta", "oats", "baked", "roasted",
               "natural", "organic", "digestive", "zero", "fibre", "millet",
               "ragi", "healthy", "lite", "light", "premium", "pure", "no maida"]

def load_data(path="nutriscan_products.csv"):
    df = pd.read_csv(path)
    df["category"] = df["name"].apply(find_category)
    df = df[df["category"] != "pet_food"]
    df = df.drop_duplicates(subset=["name", "brand"])
    df["is_bad"] = (df["quality_score"] < 50).astype(int)
    return df.reset_index(drop = True)

def build_halo(data, min_products=10):
    results = []

    for category in data["category"].unique():
        subset = data[data["category"] == category]
        for claim in CLAIM_WORDS:
            has_claim = subset["name"].str.lower().str.contains(claim, regex = False)
            if has_claim.sum() < min_products:   continue
            if (~has_claim).sum() < min_products:  continue
            results.append({
                "category" : category,
                "claim" : claim,
                "n_with" : has_claim.sum(),
                "mean_with" : subset[has_claim]["quality_score"].mean(),
                "mean_without" : subset[~has_claim]["quality_score"].mean(),
            })
    out = pd.DataFrame(results)
    out["difference"] = out["mean_with"] - out["mean_without"]
    return out.sort_values("difference")

def train_model(data):
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(data["name"], data["is_bad"])
    return model

def explain_prediction(model, text, top=6):
    vec = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    x = vec.transform([text])
    words = vec.get_feature_names_out()
    rows = [(words[i], float(x[0,i] * clf.coef_[0][i])) for i in x.nonzero()[1]]
    rows.sort(key=lambda r: -abs(r[1]))
    return rows[:top]
"""
CLASSIFY: give every video one topic, in three passes.

1. Rules: title + channel keywords label the videos that say what they are.
2. Channel: a channel with 3+ rule-labeled videos passes its majority topic to the rest.
3. Model: logistic regression on title embeddings (local multilingual MiniLM, handles
   non-English titles), trained on passes 1-2, labels what's left when it's confident.
Off-limits titles get no topic, so they can never reach the site.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import config

TOPICS = {
    "Makeup & hair": r"makeup|make-up|blush|contour|concealer|bronzer|foundation|lipstick|lip ?liner|mascara|eyeliner|skincare|"
                     r"skin ?care|beauty|\bglam\b|grwm|get ready with me|hair|curls|blowout|airwrap|rollers|in the bag|tilbury|allure",
    "Femininity & dating": r"feminin\w*|\bmen\b|\bhim\b|\bguy\b|\bchase\b|attract\w*|charism\w*|\bwom[ae]n (?:who|always)|spoiled|"
                           r"high.value|it girl|psychology of (?:women|cultural)|unforgettable woman|calm women|standards|soft life|dating|obsessed with you",
    "Mindset & discipline": r"overthink\w*|discipline|habits?|level up|mindset|rebrand yourself|ceo of|focus\w*|productiv\w*|confiden\w*|"
                            r"self.concept|glow.?up|motivat\w*|success|detach|stop caring|ambitio\w*|reinvent|leila hormozi|jay shetty",
    "Calm & wellbeing": r"nervous system|anxiety|mindful\w*|binaural|frequency|meditat\w*|\bsleep\b|therapy|alcohol|regulat\w*|"
                        r"huberman|stress|screen addiction|healing|dopamine|cortisol|stretch\w*|yoga",
    "Talks & ideas": r"\bted\b|tedx|ted talk|ab ?talks|commencement|lecture|keynote",
    "Film & TV": r"trailer|movie|film|review|netflix|prime video|movieclips|bridgerton|animated summary|\bscene\b|episode|season \d|"
                 r"rotten tomatoes|youtube movies|msmojo|\bedit\b|edits|euphoria|gossip girl|summer i turned|love island|\bclip\b|"
                 r"freeform|shondaland|\bhbo\b|disney channel|\bcast\b|actor|actress",
    "Food": r"recipe|kitchen|bread|\bcook\w*|bake\w*|baking|ranveer brar|chef|dessert",
    "Sports": r"\bnfl\b|cricinfo|cricket|table tennis|pechpong|\bwtt\b|topspin|super bowl|\bipl\b",
    "Art & making": r"painting|acrylic|drawing|sketch|watercolor|crochet|\bdiy\b|artwork",
}
# Topic rules for topics the site leaves out live in a private file (gitignored) and are merged first.
_PRIVATE = config.ROOT / "pipeline/private_topics.json"
if _PRIVATE.exists():
    TOPICS = {**json.loads(_PRIVATE.read_text())["topics"], **TOPICS}
PATTERNS = {t: re.compile(p, re.I) for t, p in TOPICS.items()}
OFF_LIMITS = re.compile((config.ROOT / "pipeline/off_limits.txt").read_text().strip(), re.I)
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CONFIDENT = 0.55


def rule_label(text):
    hits = {t: len(p.findall(text)) for t, p in PATTERNS.items()}
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])
    if ranked[0][1] == 0 or ranked[0][1] == ranked[1][1]:
        return None                         # nothing, or a tie: leave it for the next pass
    return ranked[0][0]


def embed(texts):
    path = config.DATA / "title_emb.npy"
    if path.exists():
        emb = np.load(path)
        if len(emb) == len(texts):
            return emb
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(MODEL).encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    np.save(path, emb)
    return emb


def main():
    w = pd.read_parquet(config.DATA / "watches_enriched.parquet")
    v = (w.groupby("video_id", dropna=False)
          .agg(title=("title", "first"), channel=("channel", "first"), views=("ts", "size"))
          .reset_index())
    v["text"] = (v.title.fillna("") + " | " + v.channel.fillna("")).str.strip(" |")
    v["off_limits"] = v.text.str.contains(OFF_LIMITS)
    v["has_text"] = v.title.notna()

    v["topic"] = v.text.map(rule_label).where(v.has_text & ~v.off_limits)
    v["topic_source"] = np.where(v.topic.notna(), "rules", None)
    print(f"1 rules    {v.topic.notna().mean():.0%}")

    labeled = v[v.topic.notna() & v.channel.notna()]
    counts = labeled.groupby(["channel", "topic"]).size().unstack(fill_value=0)
    majority = counts.idxmax(axis=1)[(counts.sum(axis=1) >= 3) & (counts.max(axis=1) / counts.sum(axis=1) >= 0.6)]
    fill = v.topic.isna() & v.has_text & ~v.off_limits & v.channel.isin(majority.index)
    v.loc[fill, "topic"] = v.loc[fill, "channel"].map(majority)
    v.loc[fill, "topic_source"] = "channel"
    print(f"2 channel  {v.topic.notna().mean():.0%}")

    emb = embed(v.text.tolist())
    train = v.topic.notna().values
    X, y = emb[train], v.topic[train].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=3000, C=4, class_weight="balanced").fit(Xtr, ytr)
    agreement = clf.score(Xte, yte)
    print(f"  model agrees with passes 1-2 on {agreement:.0%} of held-out titles")
    clf.fit(X, y)
    todo = (v.topic.isna() & v.has_text & ~v.off_limits).values
    if todo.any():
        p = clf.predict_proba(emb[todo])
        keep = p.max(axis=1) >= CONFIDENT
        idx = np.where(todo)[0][keep]
        v.loc[idx, "topic"] = clf.classes_[p.argmax(axis=1)[keep]]
        v.loc[idx, "topic_source"] = "model"
    print(f"3 model    {v.topic.notna().mean():.0%}  (off-limits {v.off_limits.sum()}, deleted {(~v.has_text).sum()})")

    v.drop(columns="text").to_parquet(config.DATA / "videos.parquet", index=False)
    (config.DATA / "classify_log.txt").write_text(
        f"held_out_agreement={agreement:.3f}\n" + v.topic_source.value_counts(dropna=False).to_string())
    print(v.groupby("topic").views.sum().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()

from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
import numpy as np
import os

app = FastAPI(title="Semantic Search API")

# --------------------------------------------------
# LOAD KB
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
kb_path = os.path.join(BASE_DIR, "semantic_kb.pkl")

print("🔄 Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"📂 Loading KB from: {kb_path}")
with open(kb_path, "rb") as f:
    kb = pickle.load(f)

kb = kb.fillna("")

sop_df = kb[kb["Doc_Type"] == "SOP"].reset_index(drop=True)
inc_df = kb[kb["Doc_Type"] == "INCIDENT"].reset_index(drop=True)

print("\n✅ KB LOADED")
print("Total records:", len(kb))
print("SOP count:", len(sop_df))
print("INC count:", len(inc_df))

sop_embeddings = np.vstack(sop_df["Embedding"].values) if not sop_df.empty else []
inc_embeddings = np.vstack(inc_df["Embedding"].values) if not inc_df.empty else []

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def semantic_search(df, embeddings, query):
    if df.empty:
        return df

    q_emb = model.encode(query)
    scores = util.cos_sim(q_emb, embeddings)[0].tolist()

    df = df.copy()
    df["Score"] = scores
    return df.sort_values("Score", ascending=False)


def keyword_score(query, text):
    return fuzz.token_set_ratio(query.lower(), text.lower()) / 100.0


def get_confidence(score):
    if score > 0.75:
        return "HIGH"
    elif score > 0.55:
        return "MEDIUM"
    return "LOW"


def group_similar_incidents(df, threshold=0.70):
    groups = []
    used = set()
    embeddings = df["Embedding"].tolist()

    for i in range(len(df)):
        if i in used:
            continue

        group = [i]

        for j in range(i + 1, len(df)):
            if j in used:
                continue

            score = util.cos_sim(embeddings[i], embeddings[j]).item()
            if score >= threshold:
                group.append(j)
                used.add(j)

        used.add(i)
        groups.append(group)

    return groups


# --------------------------------------------------
# ✅ SEARCH API (FIXED VERSION)
# --------------------------------------------------
@app.post("/search")
def search(payload: dict):

    query = payload.get("query", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    print(f"\n🔍 Query: {query}")

    # ✅ OPTIONAL: handle vague queries
    if len(query) < 4:
        return {
            "type": "NONE",
            "message": "Please provide a more specific query"
        }

    # =====================================================
    # ✅ STEP 1: SOP (PRIORITY)
    # =====================================================
    sop_results = semantic_search(sop_df, sop_embeddings, query)

    if not sop_results.empty:

        sop_results["Keyword"] = sop_results.apply(
            lambda r: max(
                keyword_score(query, str(r.get("Title", ""))),
                keyword_score(query, str(r.get("Content", "")))
            ),
            axis=1
        )

        sop_results["FinalScore"] = (
            0.5 * sop_results["Score"] +
            0.5 * sop_results["Keyword"]
        )

        sop_results = sop_results.sort_values("FinalScore", ascending=False)

        best = sop_results.iloc[0]
        sop_score = best["FinalScore"]

        print("👉 Best SOP score:", sop_score)

        # ✅ STRICT SOP FILTER
        if sop_score >= 0.65:
            return {
                "type": "SOP",
                "results": [{
                    "Title": best.get("Title", ""),
                    "Content": best.get("Content", "")[:800],
                    "Score": float(sop_score),
                    "Confidence": get_confidence(sop_score),
                    "file_name": best.get("Title", "") + ".pdf"
                }]
            }

    # =====================================================
    # ✅ STEP 2: INCIDENT (STRICT FILTER)
    # =====================================================
    inc_results = semantic_search(inc_df, inc_embeddings, query)

    if not inc_results.empty:

        inc_results["Keyword"] = inc_results["short_description"].apply(
            lambda x: keyword_score(query, str(x))
        )

        inc_results["FinalScore"] = (
            0.6 * inc_results["Score"] +
            0.4 * inc_results["Keyword"]
        )

        inc_results = inc_results.sort_values("FinalScore", ascending=False)

        best = inc_results.iloc[0]
        print("👉 Best INC score:", best["FinalScore"])

        # ✅ STRICT threshold (FIXED)
        if best["FinalScore"] >= 0.5:

            inc_results = inc_results.head(10).reset_index(drop=True)
            grouped = group_similar_incidents(inc_results)
            best_group = grouped[0]

            related = [
                inc_results.iloc[i]["number"]
                for i in best_group[1:]
            ][:5]

            return {
                "type": "INCIDENT",
                "results": [{
                    "short_description": best.get("short_description", ""),
                    "Content": best.get("Content", "")[:800],
                    "Score": float(best["FinalScore"]),
                    "Confidence": get_confidence(best["FinalScore"]),
                    "Primary_Incident": best.get("number", ""),
                    "Related_Incidents": related,
                    "Tag": "Semantic Match ✅"
                }]
            }

    # =====================================================
    # ✅ STEP 3: STRICT FUZZY FALLBACK
    # =====================================================
    if not inc_df.empty:

        inc_df_copy = inc_df.copy()

        inc_df_copy["FuzzyScore"] = inc_df_copy["short_description"].apply(
            lambda x: keyword_score(query, str(x))
        )

        inc_df_copy = inc_df_copy.sort_values("FuzzyScore", ascending=False)

        best = inc_df_copy.iloc[0]
        print("👉 Fuzzy score:", best["FuzzyScore"])

        # ✅ STRICT threshold (FIXED)
        if best["FuzzyScore"] >= 0.4:

            return {
                "type": "INCIDENT",
                "results": [{
                    "short_description": best.get("short_description", ""),
                    "Content": best.get("Content", "")[:800],
                    "Score": float(best["FuzzyScore"]),
                    "Confidence": "LOW",
                    "Primary_Incident": best.get("number", ""),
                    "Related_Incidents": [],
                    "Tag": "Fuzzy Match ⚠️"
                }]
            }

    # =====================================================
    # ✅ FINAL FIX
    # =====================================================
    return {
        "type": "NONE",
        "message": f"No relevant results found for '{query}'. Try a more specific query."
    }
 
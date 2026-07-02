import os
import re
import pickle
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
INPUT_FILE = "incident.csv"
SOP_DIR = "sop_pdfs"
OUTPUT_FILE = "semantic_kb.pkl"

os.makedirs(SOP_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("🔄 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# NORMALIZE EMBEDDINGS
# --------------------------------------------------
def normalize(vec):
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype("float32") if norm != 0 else vec

# --------------------------------------------------
# SAFE CSV READER (ENHANCED LOGGING)
# --------------------------------------------------
def read_csv_safe(file):

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16"]
    separators = [",", ";", "\t"]

    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    file,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip"
                )
                print(f"✅ CSV loaded (encoding={enc}, sep='{sep}')")
                return df
            except Exception:
                continue

    raise ValueError("❌ CSV format or encoding not supported")

# --------------------------------------------------
# CLEAN TEXT (IMPROVED)
# --------------------------------------------------
def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\binc\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()

# --------------------------------------------------
# PROCESS INCIDENTS
# --------------------------------------------------
def process_incidents():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found")

    df = read_csv_safe(INPUT_FILE)
    df = df.fillna("")

    required_cols = ["number", "short_description", "close_notes"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ✅ Structured content
    df["Content"] = df.apply(
        lambda r: (
            f"Incident: {r['number']}\n"
            f"Issue: {clean_text(r['short_description'])}\n"
            f"Resolution: {clean_text(r['close_notes'])}"
        ),
        axis=1
    )

    df = df[df["Content"].str.strip() != ""]

    print("🔄 Generating incident embeddings...")

    embeddings = model.encode(
        df["Content"].tolist(),
        batch_size=32,
        show_progress_bar=True
    )

    df["Embedding"] = [normalize(e) for e in embeddings]
    df["Doc_Type"] = "INCIDENT"

    return df[
        [
            "Doc_Type",
            "number",
            "short_description",
            "close_notes",
            "Content",
            "Embedding"
        ]
    ]

# --------------------------------------------------
# PDF TEXT EXTRACTION (SAFER)
# --------------------------------------------------
def extract_pdf_text(path):

    try:
        reader = PdfReader(path)

        text = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text.append(content)

        return "\n".join(text)

    except Exception as e:
        print(f"❌ Error reading PDF {path}: {e}")
        return ""

# --------------------------------------------------
# PROCESS SOP PDFs
# --------------------------------------------------
def process_sops():

    rows = []

    if not os.path.exists(SOP_DIR):
        print("⚠️ SOP directory not found")
        return pd.DataFrame()

    pdf_files = [f for f in os.listdir(SOP_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No SOP PDFs found")
        return pd.DataFrame()

    for file in pdf_files:

        path = os.path.join(SOP_DIR, file)
        print(f"📘 Processing SOP: {file}")

        text = extract_pdf_text(path)

        if len(text.strip()) < 100:
            print(f"⚠️ Skipping small file: {file}")
            continue

        content = text[:5000]

        embedding = normalize(
            model.encode(content[:3000])
        )

        rows.append({
            "Doc_Type": "SOP",
            "Title": file.replace(".pdf", ""),
            "short_description": clean_text(file),
            "close_notes": "",
            "Content": content,
            "Embedding": embedding,
            "File_Name": file   # ✅ NEW (useful for linking)
        })

    return pd.DataFrame(rows)

# --------------------------------------------------
# BUILD KNOWLEDGE BASE
# --------------------------------------------------
def build_kb():

    print("\n⚙️ Processing incidents...")
    inc = process_incidents()
    print(f"✅ {len(inc)} incidents processed")

    print("\n⚙️ Processing SOPs...")
    sop = process_sops()
    print(f"✅ {len(sop)} SOPs processed")

    print("\n🔗 Combining knowledge base...")

    kb = pd.concat([sop, inc], ignore_index=True)

    if kb.empty:
        raise ValueError("❌ Knowledge base is empty!")

    # ✅ Shuffle for better retrieval
    kb = kb.sample(frac=1, random_state=42).reset_index(drop=True)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(kb, f)

    print(f"\n✅ Knowledge Base built successfully with {len(kb)} records")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    build_kb()
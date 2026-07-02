import streamlit as st
import requests
import os
import subprocess
import base64
from auth import login

API_URL ="http://127.0.0.1:8000/search"
SOP_DIR = "sop_pdfs"
INPUT_FILE = "incident.csv"

os.makedirs(SOP_DIR, exist_ok=True)

st.set_page_config(
    page_title="Semantic IT Knowledge Base",
    layout="wide"
)

st.title("🧠 Semantic IT Knowledge Base")

# -------------------------------
# LOGIN
# -------------------------------
role = login()

# -------------------------------
# Helper: Highlight text
# -------------------------------
def highlight(text, query):
    if not text:
        return ""
    for word in query.split():
        text = text.replace(word, f"**{word}**")
    return text

# -------------------------------
# Helper: Confidence display
# -------------------------------
def show_confidence(score):
    if score > 0.75:
        return "🟢 HIGH"
    elif score > 0.55:
        return "🟡 MEDIUM"
    else:
        return "🔴 LOW"

# --------------------------------------------------
# ADMIN PANEL
# --------------------------------------------------
if role == "ADMIN":

    st.sidebar.success("Logged in as Admin")
    st.subheader("🔧 Admin Panel")

    # Upload CSV
    st.markdown("### 📂 Upload Incident CSV")
    csv = st.file_uploader("Upload Incident CSV", type=["csv"])

    if csv:
        with open(INPUT_FILE, "wb") as f:
            f.write(csv.getbuffer())
        st.success("✅ Incident CSV uploaded successfully")

    # Upload PDFs
    st.markdown("### 📘 Upload SOP PDFs")
    pdfs = st.file_uploader(
        "Upload SOP PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if pdfs:
        for f in pdfs:
            with open(os.path.join(SOP_DIR, f.name), "wb") as out:
                out.write(f.read())
        st.success("✅ SOP PDFs uploaded successfully")

    # Build KB
    st.markdown("### 🔄 Build Knowledge Base")

    if st.button("Build / Refresh KB"):
        with st.spinner("Building Knowledge Base..."):
            subprocess.run(["python", "kb_builder.py"])
        st.success("✅ Knowledge Base rebuilt successfully")

# --------------------------------------------------
# SEARCH PANEL
# --------------------------------------------------
st.subheader("🔍 Search the Knowledge Base")

query = st.text_input("Enter search term")

st.caption("💡 Try: price mismatch, token issue, booking issue")

if st.button("Search"):

    if not query.strip():
        st.warning("Please enter a search term")

    else:
        with st.spinner("🔍 Searching..."):

            try:
                response = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=10
                )

                data = response.json()

                # ======================================
                # ✅ SOP RESULTS
                # ======================================
                if data.get("type") == "SOP":

                    st.subheader("📘 SOP Result")

                    for r in data.get("results", []):

                        score = round(r.get("Score", 0), 3)

                        with st.expander(f"📘 {r['Title']} (Score: {score})"):

                            st.progress(min(score, 1.0))
                            st.markdown(f"**Confidence:** {show_confidence(score)}")

                            st.markdown("### 📄 SOP Content")
                            st.markdown(highlight(r["Content"], query))

                            # ✅ Open SOP PDF
                            file_name = r.get("file_name", "")
                            file_path = os.path.join(SOP_DIR, file_name)

                            if os.path.exists(file_path):
                                with open(file_path, "rb") as f:
                                    base64_pdf = base64.b64encode(f.read()).decode("utf-8")

                                pdf_link = f'<a href="data:application/pdf;base64,{base64_pdf}" target="_blank">📄 Open SOP</a>'
                                st.markdown(pdf_link, unsafe_allow_html=True)
                            else:
                                st.warning("⚠️ SOP file not found")

                # ======================================
                # ✅ INCIDENT RESULTS
                # ======================================
                elif data.get("type") == "INCIDENT":

                    st.subheader("🧾 Incident Result")

                    for r in data.get("results", []):

                        score = round(r.get("Score", 0), 3)

                        with st.expander(f"📌 {r['short_description']} (Score: {score})"):

                            st.progress(min(score, 1.0))
                            st.markdown(f"**Confidence:** {show_confidence(score)}")

                            st.success(f"Primary Incident: {r['Primary_Incident']}")

                            st.markdown("### 📄 Details")
                            st.markdown(highlight(r["Content"], query))

                            if r.get("Related_Incidents"):
                                st.markdown("### 🔗 Related Incidents")
                                st.write(", ".join(r["Related_Incidents"]))

                # ======================================
                # ✅ NO RESULTS
                # ======================================
                else:
                    st.warning("❌ No results found")

            except Exception as e:
                st.error("❌ Unable to connect to Search API")
                st.code(str(e))
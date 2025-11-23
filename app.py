# app.py — UPDATED to use a current Groq model (MODEL_NAME from .env)
# - Robust CSV loader (skips bad lines)
# - Groq (ChatGroq) parser chain for meal parsing
# - FAISS guideline index (RAG) using HuggingFace embeddings (optional)
# - Fallback rule-based parser if LLM / chain fails
# - Interactive selection for fuzzy matching results from the CSV

import os
import json
import time
import traceback
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

# MUST be the very first Streamlit UI call
st.set_page_config(page_title="Nutrition RAG Assistant", layout="wide")

# --------------------------
# Environment & config
# --------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Read model name from environment with safe default
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

# --------------------------
# Try imports for LangChain / Groq / FAISS (optional)
# --------------------------
try:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFDirectoryLoader

    LANGCHAIN_OK = True
except Exception as e:
    LANGCHAIN_OK = False
    print("LangChain / Groq / FAISS import failed (RAG disabled):", e)
    traceback.print_exc()

# Initialize ChatGroq if possible — catch decommission/access errors
llm = None
if LANGCHAIN_OK and GROQ_API_KEY:
    try:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
        print("Initialized ChatGroq with model:", MODEL_NAME)
    except Exception as e:
        llm = None
        print("Failed to initialize ChatGroq:", e)
        # try to show helpful diagnostic if the exception contains info
        try:
            err_txt = str(e)
            if "decommission" in err_txt.lower() or "model_not_found" in err_txt.lower():
                print("Model may be decommissioned or inaccessible. Check MODEL_NAME and Groq dashboard.")
        except:
            pass
        traceback.print_exc()

# --------------------------
# CSV path (your file)
# --------------------------
DEFAULT_CSV_PATH = "data/nutrition_master_big.csv"  # <- change if your file is different

# --------------------------
# Robust CSV loader
# --------------------------
@st.cache_data
def load_nutrition_csv(path: str = DEFAULT_CSV_PATH) -> Optional[pd.DataFrame]:
    """
    Robust CSV loader:
    - uses engine='python' and on_bad_lines='skip' to avoid tokenization errors
    - normalizes column names and fills missing numeric columns with zeros
    """
    if not os.path.exists(path):
        return None

    df = None
    # try utf-8 then latin1
    for enc in ("utf-8", "latin1"):
        try:
            df = pd.read_csv(path, sep=",", engine="python", encoding=enc, on_bad_lines="skip")
            break
        except Exception as e:
            print(f"read_csv failed with encoding {enc}: {e}")
    if df is None:
        return None

    # normalize headers
    df.columns = [c.strip() for c in df.columns]

    # map common column variations to canonical names
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    for candidate in ("description", "desc", "food", "food_name"):
        if candidate in cols_lower:
            rename_map[cols_lower[candidate]] = "Description"
            break

    for candidate in ("calories", "energy", "energy_kcal", "kcal"):
        if candidate in cols_lower:
            rename_map[cols_lower[candidate]] = "Calories"
            break

    col_candidates = {
        "Protein (g)": ["protein", "protein (g)"],
        "Fat (g)": ["fat", "total lipid", "fat (g)"],
        "Carbs (g)": ["carbs", "carbohydrate", "carbohydrate (g)"],
        "Sodium (mg)": ["sodium", "sodium (mg)"]
    }
    for canonical, tries in col_candidates.items():
        for t in tries:
            if t in cols_lower:
                rename_map[cols_lower[t]] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    # ensure Description exists, if not pick first object/string column
    if "Description" not in df.columns:
        for c in df.columns:
            if df[c].dtype == object:
                df = df.rename(columns={c: "Description"})
                break

    # ensure numeric columns exist
    for c in ["Calories", "Protein (g)", "Fat (g)", "Carbs (g)", "Sodium (mg)"]:
        if c not in df.columns:
            df[c] = 0

    # clean numeric columns robustly
    def clean_numeric_series(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-eE]", "", regex=True), errors="coerce").fillna(0.0)

    df["Calories"] = clean_numeric_series(df["Calories"])
    df["Protein (g)"] = clean_numeric_series(df["Protein (g)"])
    df["Fat (g)"] = clean_numeric_series(df["Fat (g)"])
    df["Carbs (g)"] = clean_numeric_series(df["Carbs (g)"])
    df["Sodium (mg)"] = clean_numeric_series(df["Sodium (mg)"])

    df["Description"] = df["Description"].astype(str).str.strip()
    df = df[df["Description"] != ""]
    df["desc_norm"] = df["Description"].str.lower()
    return df

# Load CSV
usda_df = load_nutrition_csv()

# --------------------------
# Fuzzy match helper
# --------------------------
def fuzzy_candidates(food_name: str, limit: int = 6):
    if usda_df is None or not food_name:
        return []
    try:
        choices = usda_df["desc_norm"].tolist()
        results = process.extract(food_name.lower(), choices, scorer=fuzz.WRatio, limit=limit)
        return results  # (match_text, score, idx)
    except Exception as e:
        print("Fuzzy search failed:", e)
        return []

# --------------------------
# Fallback parser
# --------------------------
def fallback_parse_meal(text: str):
    text = text.replace(" and ", ", ")
    chunks = [c.strip() for c in text.split(",") if c.strip()]
    parsed = []
    for c in chunks:
        tokens = c.split()
        amount = None
        unit = None
        name = c
        if tokens:
            first = tokens[0]
            if any(ch.isdigit() for ch in first):
                num = "".join([ch for ch in first if (ch.isdigit() or ch == ".")])
                letters = "".join([ch for ch in first if ch.isalpha()])
                try:
                    if num:
                        amount = float(num)
                        unit = letters if letters else None
                        name = " ".join(tokens[1:]) if len(tokens) > 1 else name
                except:
                    name = c
            else:
                if len(tokens) >= 3 and tokens[0].replace(".", "", 1).isdigit():
                    try:
                        amount = float(tokens[0])
                        unit = tokens[1]
                        name = " ".join(tokens[2:])
                    except:
                        name = c
                else:
                    name = c
        parsed.append({"raw": c, "name": name.lower(), "amount": amount, "unit": unit})
    return parsed

# --------------------------
# Groq parser chain (LangChain)
# --------------------------
if LANGCHAIN_OK:
    PARSER_PROMPT = ChatPromptTemplate.from_template(
        """
Parse the following meal into a JSON array of items.
Each item must contain:
- name (string)
- amount (number or null)
- unit (string or null)
- raw (original phrase)

Return ONLY valid JSON (an array).

Meal:
{input}
"""
    )

def parse_meal_with_groq_chain(meal_text: str):
    if not (LANGCHAIN_OK and llm):
        return None
    try:
        parser_chain = create_stuff_documents_chain(llm, PARSER_PROMPT)
        resp = parser_chain.invoke({"input": meal_text})
        text = None
        if isinstance(resp, dict):
            for k in ("answer", "output_text", "text"):
                if k in resp and isinstance(resp[k], str):
                    text = resp[k]
                    break
            if text is None:
                for v in resp.values():
                    if isinstance(v, str) and "[" in v and "]" in v:
                        text = v
                        break
        else:
            text = str(resp)

        if text:
            s = text.find("[")
            e = text.rfind("]")
            if s != -1 and e != -1:
                try:
                    parsed = json.loads(text[s:e+1])
                    return parsed
                except Exception:
                    print("JSON decode failed for parser output; showing output for debugging:")
                    print(text)
                    return None
        return None
    except Exception as exc:
        print("parse_meal_with_groq_chain failed:", exc)
        traceback.print_exc()
        return None

# --------------------------
# Build guideline index (FAISS)
# --------------------------
@st.cache_resource
def build_guideline_index(folder: str = "./guidelines"):
    if not LANGCHAIN_OK:
        return None
    if not os.path.exists(folder):
        return None
    try:
        loader = PyPDFDirectoryLoader(folder)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        split_docs = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        vs = FAISS.from_documents(split_docs, embeddings)
        return vs
    except Exception as e:
        print("Failed to build guideline index:", e)
        traceback.print_exc()
        return None

# --------------------------
# Streamlit UI
# --------------------------
st.title("🍽 Nutrition RAG Assistant — Groq Llama3 (updated model)")
st.caption("Parse meal → confirm items → compute nutrition → ask health questions with RAG.")

st.sidebar.header("Status & keys")
st.sidebar.write("GROQ_API_KEY present:", bool(GROQ_API_KEY))
st.sidebar.write("LangChain OK:", LANGCHAIN_OK)
st.sidebar.write("ChatGroq initialized:", llm is not None)
st.sidebar.write("MODEL_NAME (using):", MODEL_NAME)
st.sidebar.write("CSV loaded:", usda_df is not None and len(usda_df) > 0)

# CSV status
if usda_df is None:
    st.error(f"CSV not found at '{DEFAULT_CSV_PATH}'. Please upload or place the CSV at that path.")
else:
    st.success(f"Loaded nutrition CSV with {len(usda_df)} items")
    if st.checkbox("Show sample rows (first 8)"):
        st.dataframe(usda_df.head(8))

# --- 1) Enter meal and compute nutrition ---
st.header("1) Enter the meal you had")
meal_text = st.text_area("Example: 'Lunch: 2 chapatis, 1 cup rice, 1 bowl dal, 1 banana'", height=140)

if st.button("Parse Meal & Compute Nutrition"):
    if not meal_text:
        st.warning("Type a meal description first.")
    else:
        parsed_items = parse_meal_with_groq_chain(meal_text)
        if parsed_items is None:
            st.caption("Using fallback parser (works reliably for simple meals).")
            parsed_items = fallback_parse_meal(meal_text)

        st.subheader("Parsed items")
        st.json(parsed_items)

        totals = {"Calories": 0.0, "Protein (g)": 0.0, "Carbs (g)": 0.0, "Fat (g)": 0.0, "Sodium (mg)": 0.0}
        details = []

        for i, item in enumerate(parsed_items):
            raw = item.get("raw", "")
            name = item.get("name", raw).strip().lower()
            st.markdown(f"*Item {i+1}:* {raw}")

            candidates = fuzzy_candidates(name, limit=6)
            options = []
            for (match_text, score, idx) in candidates:
                desc = usda_df.iloc[idx]["Description"]
                options.append(f"{desc}  (score {score})")
            options.append("None / use estimate")

            choice = st.selectbox(f"Select match for: '{raw}'", options, key=f"match_{i}")

            if choice == "None / use estimate":
                est_cal = 150.0
                totals["Calories"] += est_cal
                details.append({"raw": raw, "matched": None, "estimate_used": True, "estimated_calories": est_cal})
            else:
                chosen_idx = options.index(choice)
                idx = candidates[chosen_idx][2]
                row = usda_df.iloc[idx]
                try:
                    totals["Calories"] += float(row.get("Calories", 0) or 0)
                except:
                    pass
                for k in [("Protein (g)", "Protein (g)"), ("Carbs (g)", "Carbs (g)"), ("Fat (g)", "Fat (g)"), ("Sodium (mg)", "Sodium (mg)")]:
                    try:
                        totals[k[0]] += float(row.get(k[1], 0) or 0)
                    except:
                        pass
                details.append({"raw": raw, "matched": row["Description"], "score": candidates[chosen_idx][1]})

        st.subheader("Nutrition summary (estimates)")
        nut_df = pd.DataFrame({
            "nutrient": ["Calories (kcal)", "Protein (g)", "Carbs (g)", "Fat (g)", "Sodium (mg)"],
            "value": [round(totals["Calories"], 1), round(totals["Protein (g)"], 1), round(totals["Carbs (g)"], 1), round(totals["Fat (g)"], 1), round(totals["Sodium (mg)"], 1)]
        }).set_index("nutrient")
        st.dataframe(nut_df)

        st.subheader("Macro breakdown (g)")
        macro_df = pd.DataFrame({
            "Macro": ["Protein", "Carbs", "Fat"],
            "grams": [round(totals["Protein (g)"], 1), round(totals["Carbs (g)"], 1), round(totals["Fat (g)"], 1)]
        }).set_index("Macro")
        st.bar_chart(macro_df)

        try:
            c_pro = totals["Protein (g)"] * 4
            c_car = totals["Carbs (g)"] * 4
            c_fat = totals["Fat (g)"] * 9
            sizes = [c_pro, c_car, c_fat]
            if sum(sizes) > 0:
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=["Protein", "Carbs", "Fat"], autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)
        except Exception:
            pass

        with st.expander("Item details / matches"):
            for d in details:
                st.write(d)

        st.session_state["last_parsed"] = parsed_items
        st.session_state["last_totals"] = totals

# --- 2) RAG follow-ups ---
st.markdown("---")
st.header("2) Ask health questions (RAG)")

if st.button("Build Guideline Index"):
    if not LANGCHAIN_OK:
        st.error("LangChain / FAISS / embeddings not installed — RAG disabled.")
    else:
        vs = build_guideline_index("./guidelines")
        st.session_state["vs"] = vs
        if vs is not None:
            st.success("Guideline index built successfully.")
        else:
            st.error("Failed to build guideline index. Ensure ./guidelines exists and has PDFs.")

rag_q = st.text_input("Ask a question about the last meal (e.g., 'Is my meal too high in sodium?')")

if st.button("Ask Groq (RAG)"):
    if "last_parsed" not in st.session_state:
        st.warning("Parse a meal first so the RAG query has context.")
    elif not (LANGCHAIN_OK and llm):
        st.error("Groq / LangChain not available. Ensure langchain_groq and related packages are installed and GROQ_API_KEY is set.")
    elif "vs" not in st.session_state or st.session_state["vs"] is None:
        st.warning("Build the guideline index first (click 'Build Guideline Index').")
    else:
        try:
            retriever = st.session_state["vs"].as_retriever()
            rag_prompt = ChatPromptTemplate.from_template("""
Use ONLY the facts in the provided context to answer the user's question.
Be concise (2-4 sentences) and provide 1-3 actionable suggestions if relevant.
Cite sources in parentheses using the document title.

<context>
{context}
</context>

Question: {input}
""")
            document_chain = create_stuff_documents_chain(llm, rag_prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            meal_context = {"parsed_items": st.session_state.get("last_parsed"), "totals": st.session_state.get("last_totals")}
            user_input = f"User question: {rag_q}\nContext: {json.dumps(meal_context)}"

            start = time.process_time()
            response = retrieval_chain.invoke({"input": user_input})
            elapsed = round(time.process_time() - start, 2)

            answer_text = None
            if isinstance(response, dict):
                for k in ("answer", "output_text", "text"):
                    if k in response and isinstance(response[k], str):
                        answer_text = response[k]
                        break
                if answer_text is None:
                    for v in response.values():
                        if isinstance(v, str) and len(v) > 0:
                            answer_text = v
                            break
            else:
                answer_text = str(response)

            st.subheader("🧠 Groq (RAG) Answer")
            st.write(answer_text or "No answer returned.")
            st.caption(f"⏱ Response time: {elapsed} s")

            if isinstance(response, dict) and response.get("context"):
                with st.expander("Relevant document chunks"):
                    for chunk in response["context"]:
                        st.write(chunk.page_content)
                        st.write("———")

        except Exception as exc:
            # Show helpful messages if model is decommissioned or access is denied
            st.error("Groq RAG call failed. See console for details.")
            st.write("Exception type:", type(exc)._name_)
            st.write("Exception message:", str(exc))
            st.text("Traceback:")
            st.text(traceback.format_exc())
            print("Groq RAG exception:")
            traceback.print_exc()

st.markdown("---")
st.info("Notes: 1) Put your CSV at data/nutrition_master_big.csv (or change DEFAULT_CSV_PATH). 2) Set GROQ_API_KEY and optionally MODEL_NAME in .env and restart Streamlit for Groq features. 3) If you see 'model_decommissioned' or 'model_not_found', open your Groq console and pick a model shown there and set MODEL_NAME to that exact string.")
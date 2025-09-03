import os
import streamlit as st
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from io import BytesIO

# =========================
# API Call Functions
# =========================
def call_spam_api(record, spam_api, category, timeout, session):
    """Call spam API for a single record and return response dict or {}."""
    payload = {"category": category, "data": [record]}
    resp = session.post(spam_api, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) and data else {}

def call_ads_api(record, ads_api, timeout, session):
    """Call ads API for a single record and return response dict."""
    resp = session.post(ads_api, json=record, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def process_chunk_parallel(df_chunk, spam_api, ads_api, category, max_workers_spam, max_workers_ads, timeout, progress_bar, status_text, session):
    """Process a chunk of data: call spam API, then ads API for spam records."""
    records = [
        {
            "id": str(row.get("Id", "")),
            "topic": str(row.get("Topic", "")),
            "topic_id": str(row.get("TopicId", "")),
            "title": str(row.get("Title", "")),
            "content": str(row.get("Content", "")),
            "description": str(row.get("Description", "")),
            "site_name": str(row.get("SiteName", "")),
            "site_id": str(row.get("SiteId", "")),
            "type": str(row.get("Type", "")),
            "sentiment": str(row.get("Sentiment", "")),
            "label": str(row.get("label", "")),
        } for _, row in df_chunk.iterrows()
    ]

    # --- Parallel spam API calls ---
    spam_results = {}
    with ThreadPoolExecutor(max_workers=max_workers_spam) as executor:
        futures = {executor.submit(call_spam_api, rec, spam_api, category, timeout, session): rec["id"] for rec in records}
        for i, future in enumerate(as_completed(futures)):
            rec_id = futures[future]
            try:
                spam_results[rec_id] = future.result()
            except Exception as e:
                spam_results[rec_id] = {}
            progress_bar.progress((i + 1) / len(records))
            status_text.text(f"Spam API {i+1}/{len(records)}")

    # --- Parallel ads API calls for spam records ---
    ads_labels = [""] * len(records)
    spam_ids = [idx for idx, rec in enumerate(records) if spam_results.get(rec["id"], {}).get("is_spam", False)]
    with ThreadPoolExecutor(max_workers=max_workers_ads) as executor:
        futures = {executor.submit(call_ads_api, records[idx], ads_api, timeout, session): idx for idx in spam_ids}
        for i, future in enumerate(as_completed(futures)):
            idx = futures[future]
            rec_id = records[idx]["id"]
            try:
                ads_result = future.result()
                ads_labels[idx] = ads_result.get("label", "")
            except Exception as e:
                ads_labels[idx] = ""
            progress_bar.progress((i + 1) / len(spam_ids) if spam_ids else 1.0)
            status_text.text(f"Ads API {i+1}/{len(spam_ids)}")

    # --- Assign is_spam and label columns ---
    df_chunk["is_spam"] = [
        spam_results.get(rec["id"], {}).get("is_spam", False) for rec in records
    ]
    df_chunk["label"] = [
        ads_labels[idx] if spam_results.get(rec["id"], {}).get("is_spam", False) else ""
        for idx, rec in enumerate(records)
    ]
    return df_chunk

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="AI-Driven Spam/Ads Explorer", page_icon="🛡️", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://kompa.ai/assets/images/logo.svg", width=150)
    st.header("⚙️ Configuration")
    spam_api = st.text_input("Spam API URL", "http://103.232.122.6:5004/predict")
    ads_api = st.text_input("Ads API URL", "http://103.232.122.6:5005/predict")
    category = st.text_input("Category", "fmcg")
    batch_size = st.number_input("Batch size", min_value=100, value=1000, step=100)
    max_workers_spam = st.number_input("Max workers (Spam API)", min_value=1, value=10, step=1)
    max_workers_ads = st.number_input("Max workers (Ads API)", min_value=1, value=10, step=1)
    timeout = st.number_input("Request timeout (sec)", min_value=10, value=60, step=10)
    uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])

st.title("🛡️ AI-Driven Spam & Ads Explorer")

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df_full = pd.read_csv(uploaded_file)
    else:
        df_full = pd.read_excel(uploaded_file)

    st.write(f"📂 File loaded: {uploaded_file.name}, Rows: {len(df_full)}")

    if st.button("▶️ Run Spam/Ads Analysis"):
        session = requests.Session()
        start_time = time.time()

        processed_chunks = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        chunk_iter = (df_full[i:i+batch_size] for i in range(0, len(df_full), batch_size))
        total_chunks = (len(df_full) + batch_size - 1) // batch_size

        for i, chunk in enumerate(chunk_iter):
            status_text.text(f"Processing chunk {i+1}/{total_chunks}...")
            processed = process_chunk_parallel(
                chunk, spam_api, ads_api, category,
                max_workers_spam, max_workers_ads, timeout,
                progress_bar, status_text, session
            )
            processed_chunks.append(processed)

        result_df = pd.concat(processed_chunks, ignore_index=True)

        elapsed_time = time.time() - start_time
        st.success(f"✅ Processing completed in {elapsed_time:.2f} seconds")

        st.subheader("📊 Processed Data")
        st.dataframe(result_df.head(50))

        # Export Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False, sheet_name="Results")
        st.download_button(
            "💾 Download Results",
            data=output.getvalue(),
            file_name=f"output_labeled_{int(time.time())}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Please upload a CSV or XLSX file to get started.")


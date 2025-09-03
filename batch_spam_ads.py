import os
import argparse
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm   # <-- NEW
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

def process_chunk_parallel(df_chunk, spam_api, ads_api, category, max_workers_spam, max_workers_ads, timeout, session):
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
        for future in tqdm(as_completed(futures), total=len(records), desc="Spam API", leave=False):
            rec_id = futures[future]
            try:
                spam_results[rec_id] = future.result()
            except Exception:
                spam_results[rec_id] = {}

    # --- Parallel ads API calls for spam records ---
    ads_labels = [""] * len(records)
    spam_ids = [idx for idx, rec in enumerate(records) if spam_results.get(rec["id"], {}).get("is_spam", False)]
    with ThreadPoolExecutor(max_workers=max_workers_ads) as executor:
        futures = {executor.submit(call_ads_api, records[idx], ads_api, timeout, session): idx for idx in spam_ids}
        for future in tqdm(as_completed(futures), total=len(spam_ids), desc="Ads API", leave=False):
            idx = futures[future]
            try:
                ads_result = future.result()
                ads_labels[idx] = ads_result.get("label", "")
            except Exception:
                ads_labels[idx] = ""

    # --- Assign is_spam and label columns ---
    df_chunk["is_spam"] = [
        spam_results.get(rec["id"], {}).get("is_spam", False) for rec in records
    ]
    df_chunk["label"] = [
        ads_labels[idx] if spam_results.get(rec["id"], {}).get("is_spam", False) else ""
        for idx, rec in enumerate(records)
    ]
    return df_chunk

def process_file(file_path, output_dir, spam_api, ads_api, category, batch_size, max_workers_spam, max_workers_ads, timeout):
    """Process one Excel file and save result to output folder."""
    print(f"\n📂 Processing file: {file_path}")
    df_full = pd.read_excel(file_path)
    print(f"   Rows loaded: {len(df_full)}")

    session = requests.Session()
    start_time = time.time()

    processed_chunks = []
    chunk_iter = (df_full[i:i+batch_size] for i in range(0, len(df_full), batch_size))
    total_chunks = (len(df_full) + batch_size - 1) // batch_size

    for i, chunk in enumerate(chunk_iter, 1):
        print(f"   → Chunk {i}/{total_chunks}")
        processed = process_chunk_parallel(
            chunk, spam_api, ads_api, category,
            max_workers_spam, max_workers_ads, timeout, session
        )
        processed_chunks.append(processed)

    result_df = pd.concat(processed_chunks, ignore_index=True)
    elapsed_time = time.time() - start_time
    print(f"✅ Completed {file_path} in {elapsed_time:.2f} sec")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_{os.path.basename(file_path)}")
    result_df.to_excel(output_file, index=False)
    print(f"💾 Saved to {output_file}\n")

def main():
    parser = argparse.ArgumentParser(description="Batch Spam/Ads Classification")
    parser.add_argument("--input", required=True, help="Input folder with XLSX files")
    parser.add_argument("--output", required=True, help="Output folder to save results")
    parser.add_argument("--spam_api", default="http://103.232.122.6:5004/predict")
    parser.add_argument("--ads_api", default="http://103.232.122.6:5005/predict")
    parser.add_argument("--category", default="fmcg")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_workers_spam", type=int, default=10)
    parser.add_argument("--max_workers_ads", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    input_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".xlsx")]
    if not input_files:
        print("❌ No .xlsx files found in input folder.")
        return

    for file_path in input_files:
        try:
            process_file(
                file_path, args.output,
                args.spam_api, args.ads_api, args.category,
                args.batch_size, args.max_workers_spam, args.max_workers_ads, args.timeout
            )
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()

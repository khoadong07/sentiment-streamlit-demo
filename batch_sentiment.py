import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm

# =========================
# Label mapping
# =========================
label_mapping = {
    "POS": "Positive",
    "NEG": "Negative",
    "NEU": "Neutral"
}

# =========================
# Load model (cache 1 lần)
# =========================
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    if hasattr(torch, "compile"):  # torch >= 2.0
        model = torch.compile(model)
    return tokenizer, config, model, device

# =========================
# Batch inference
# =========================
def batch_inference(texts, tokenizer, config, model, device, batch_size=32):
    all_results = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device, non_blocking=True)

            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

            for score in scores:
                top_id = np.argmax(score)
                original_label = config.id2label[top_id]
                mapped_label = label_mapping.get(original_label, original_label)
                all_results.append(mapped_label)
    return all_results

# =========================
# Process file
# =========================
def process_file(input_path: str, output_path: str, tokenizer, config, model, device, batch_size=32):
    df = pd.read_excel(input_path) if input_path.endswith(".xlsx") else pd.read_csv(input_path)

    if "is_spam" not in df.columns or "label" not in df.columns:
        print(f"⚠️ File {input_path} thiếu cột is_spam hoặc label, bỏ qua.")
        return

    df_filtered = df[(df["is_spam"] == False) | (df["label"].str.strip().str.lower() == "rao vặt")]
    total_rows = len(df)
    filtered_rows = len(df_filtered)

    print(f"📂 {os.path.basename(input_path)}: {total_rows} rows -> {filtered_rows} rows to process")

    if filtered_rows == 0:
        print(f"⚠️ File {input_path} không có record hợp lệ, bỏ qua.")
        return
    
    # Chuẩn bị text để inference
    texts = []
    for _, row in df_filtered.iterrows():
        content = str(row.get("Content", "") or "")
        title = str(row.get("Title", "") or "")
        description = str(row.get("Description", "") or "")
        item_type = str(row.get("Type", "") or "")

        is_meaningless = not any(c.isalnum() for c in content)

        if is_meaningless:
            if item_type in ["fbPageTopic", "fbGroupTopic", "fbUserTopic"]:
                text = f"{title} {description} {content}"
            else:
                text = content
        else:
            text = content

        texts.append(text)

    # Batch inference
    sentiments = batch_inference(texts, tokenizer, config, model, device, batch_size)

    # Thêm cột sentiment_by_ai (giữ nguyên input)
    df.loc[df_filtered.index, "sentiment_by_ai"] = sentiments

    # Save
    if output_path.endswith(".xlsx"):
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"✅ Done. Saved {len(df)} rows to {output_path}")

# =========================
# Main
# =========================
def main(input_dir, output_dir, model_path, batch_size=32):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, config, model, device = load_model(model_path)

    files = [f for f in os.listdir(input_dir) if f.endswith((".csv", ".xlsx"))]
    if not files:
        print("⚠️ Không có file CSV/XLSX trong input/")
        return

    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"sentiment_{file}")
        process_file(input_path, output_path, tokenizer, config, model, device, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch sentiment analysis (GPU optimized)")
    parser.add_argument("--input", required=True, help="Input folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--model", default="Khoa/sentiment-analysis-all-category-122024.8", help="Model path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for GPU inference")
    args = parser.parse_args()

    main(args.input, args.output, args.model, args.batch_size)

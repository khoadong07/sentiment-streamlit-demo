import os
import argparse
import pandas as pd
import numpy as np
import torch
import asyncio
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
# Call inference cho 1 text
# =========================
async def call_inference(text: str, tokenizer, config, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    ranking = np.argsort(scores)[::-1]
    top_label_id = ranking[0]
    original_label = config.id2label[top_label_id]
    mapped_label = label_mapping.get(original_label, original_label)
    confidence = float(np.round(scores[top_label_id], 4))

    return mapped_label, confidence


# =========================
# Rule-based process 1 item
# =========================
async def process_item(item: dict, tokenizer, config, model, device):
    content = str(item.get("Content", "") or "")
    title = str(item.get("Title", "") or "")
    description = str(item.get("Description", "") or "")
    item_type = str(item.get("Type", "") or "")

    is_meaningless = not any(c.isalnum() for c in content)

    if is_meaningless:
        if item_type in ["fbPageTopic", "fbGroupTopic", "fbUserTopic"]:
            text = f"{title} {description} {content}"
            sentiment, confidence = await call_inference(text, tokenizer, config, model, device)
        else:
            sentiment = "Neutral"
            text = content
            confidence = 1.0
    else:
        text = content
        sentiment, confidence = await call_inference(content, tokenizer, config, model, device)

    item["Sentiment By AI"] = sentiment
    item["Confidence"] = confidence
    item["ProcessedText"] = text
    return item


# =========================
# Process file
# =========================
async def process_file(input_path: str, output_path: str, tokenizer, config, model, device):
    df = pd.read_excel(input_path) if input_path.endswith(".xlsx") else pd.read_csv(input_path)

    if "is_spam" not in df.columns or "label" not in df.columns:
        print(f"⚠️ File {input_path} thiếu cột is_spam hoặc Label, bỏ qua.")
        return

    df_filtered = df[(df["is_spam"] == False) | (df["label"].str.strip().str.lower() == "rao vặt")]
    total_rows = len(df)
    filtered_rows = len(df_filtered)

    print(f"📂 {os.path.basename(input_path)}: {total_rows} rows -> {filtered_rows} rows to process")

    if filtered_rows == 0:
        print(f"⚠️ File {input_path} không có record hợp lệ, bỏ qua.")
        return
    
    records = df_filtered.to_dict(orient="records")

    results = []
    for item in tqdm(records, desc=f"Processing {os.path.basename(input_path)}", unit="record"):
        res = await process_item(item, tokenizer, config, model, device)
        results.append(res)

    if not results:
        print(f"⚠️ File {input_path} không có record hợp lệ.")
        return

    df_result = pd.DataFrame(results)
    if output_path.endswith(".xlsx"):
        df_result.to_excel(output_path, index=False)
    else:
        df_result.to_csv(output_path, index=False)

    print(f"✅ Done. Saved {len(results)} rows to {output_path}")


# =========================
# Main
# =========================
async def main(input_dir, output_dir, model_path):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, config, model, device = load_model(model_path)

    files = [f for f in os.listdir(input_dir) if f.endswith((".csv", ".xlsx"))]
    if not files:
        print("⚠️ Không có file CSV/XLSX trong input/")
        return

    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"sentiment_{file}")
        await process_file(input_path, output_path, tokenizer, config, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch sentiment analysis")
    parser.add_argument("--input", required=True, help="Input folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--model", default="Khoa/sentiment-analysis-all-category-122024.8", help="Model path")
    args = parser.parse_args()

    asyncio.run(main(args.input, args.output, args.model))

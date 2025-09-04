import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm
import time

# =========================
# Set environment variable to disable tokenizer parallelism
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# Label mapping
# =========================
label_mapping = {
    "POS": "Positive",
    "NEG": "Negative",
    "NEU": "Neutral"
}

# =========================
# Save and Load model
# =========================
def save_model(model_path, local_dir="./models"):
    """Download and save model and tokenizer to local directory."""
    print(f"📁 Checking model at {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    local_model_path = os.path.join(local_dir, model_path.replace("/", "_"))
    
    if not os.path.exists(local_model_path):
        print(f"📥 Downloading model {model_path}...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)
        print(f"✅ Model saved to {local_model_path} in {time.time() - start_time:.2f} seconds")
    else:
        print(f"✅ Model already exists at {local_model_path}")
    return local_model_path

def load_model(model_path, local_dir="./models", force_device=None):
    """Load model and tokenizer from local directory or download if not present."""
    print(f"🔍 Checking for model at {local_dir}")
    local_model_path = os.path.join(local_dir, model_path.replace("/", "_"))
    
    if not os.path.exists(local_model_path):
        print(f"⚠️ Model not found at {local_model_path}, downloading...")
        local_model_path = save_model(model_path, local_dir)
    
    print(f"📂 Loading model from {local_model_path}")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    config = AutoConfig.from_pretrained(local_model_path)
    print(f"✅ Loaded tokenizer and config in {time.time() - start_time:.2f} seconds")

    if force_device:
        device = torch.device(force_device)
        print(f"🖥️ Using forced device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Automatically selected device: {device}")

    try:
        print(f"📦 Loading model to {device}...")
        start_time = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            local_model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)
        model.eval()
        if hasattr(torch, "compile"):  # torch >= 2.0
            print("🔧 Compiling model with torch.compile...")
            model = torch.compile(model)
        print(f"✅ Model loaded to {device} in {time.time() - start_time:.2f} seconds")
    except RuntimeError as e:
        print(f"⚠️ Error moving model to {device}: {e}")
        print("Falling back to CPU")
        device = torch.device("cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            local_model_path,
            torch_dtype=torch.float32
        ).to(device)
        model.eval()
        print(f"✅ Model loaded to CPU in {time.time() - start_time:.2f} seconds")

    return tokenizer, config, model, device

# =========================
# Inference cho batch text
# =========================
def batch_inference(texts, tokenizer, config, model, device, batch_size=32):
    """Tokenize and perform inference for each batch immediately."""
    print(f"🚀 Starting batch inference for {len(texts)} texts with batch_size={batch_size}")
    results = []
    
    # Use tqdm to track batch progress
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        print(f"📄 Processing batch {i//batch_size + 1} with {len(batch_texts)} texts")

        # Tokenize current batch
        start_time = time.time()
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        print(f"✅ Tokenized batch in {time.time() - start_time:.2f} seconds")

        # Run inference immediately
        start_time = time.time()
        with torch.inference_mode():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
        print(f"✅ Inferred batch in {time.time() - start_time:.2f} seconds")

        # Process scores
        for score in scores:
            top_id = int(np.argmax(score))
            original_label = config.id2label[top_id]
            mapped_label = label_mapping.get(original_label, original_label)
            confidence = float(np.round(score[top_id], 4))
            results.append((mapped_label, confidence))

        # Clear memory to prevent accumulation
        del inputs, outputs, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"✅ Completed inference for {len(texts)} texts")
    return results

# =========================
# Process file
# =========================
def process_file(input_path: str, output_path: str, tokenizer, config, model, device):
    print(f"📂 Processing file: {input_path}")
    start_time = time.time()
    df = pd.read_excel(input_path) if input_path.endswith(".xlsx") else pd.read_csv(input_path)
    print(f"✅ Loaded input file in {time.time() - start_time:.2f} seconds")

    if "is_spam" not in df.columns or "label" not in df.columns:
        print(f"⚠️ File {input_path} thiếu cột is_spam hoặc Label, bỏ qua.")
        return

    df_filtered = df[(df["is_spam"] == False) | (df["label"].str.strip().str.lower() == "rao vặt")]
    total_rows = len(df)
    filtered_rows = len(df_filtered)
    print(f"📊 {os.path.basename(input_path)}: {total_rows} rows -> {filtered_rows} rows to process")

    if filtered_rows == 0:
        print(f"⚠️ File {input_path} không có record hợp lệ, bỏ qua.")
        return

    records = df_filtered.to_dict(orient="records")

    texts = []
    processed_texts = []
    for item in records:
        content = str(item.get("Content", "") or "")
        title = str(item.get("Title", "") or "")
        description = str(item.get("Description", "") or "")
        item_type = str(item.get("Type", "") or "")

        is_meaningless = not any(c.isalnum() for c in content)
        if is_meaningless:
            if item_type in ["fbPageTopic", "fbGroupTopic", "fbUserTopic"]:
                text = f"{title} {description} {content}"
            else:
                text = content
        else:
            text = content
        texts.append(text)
        processed_texts.append(text)

    start_time = time.time()
    sentiments_confidences = batch_inference(texts, tokenizer, config, model, device, batch_size=32)
    print(f"✅ Completed sentiment analysis in {time.time() - start_time:.2f} seconds")

    for i, (sentiment, confidence) in enumerate(sentiments_confidences):
        records[i]["Sentiment By AI"] = sentiment
        records[i]["Confidence"] = confidence
        records[i]["ProcessedText"] = processed_texts[i]

    df_result = pd.DataFrame(records)
    start_time = time.time()
    if output_path.endswith(".xlsx"):
        df_result.to_excel(output_path, index=False)
    else:
        df_result.to_csv(output_path, index=False)
    print(f"✅ Saved {len(records)} rows to {output_path} in {time.time() - start_time:.2f} seconds")

# =========================
# Main
# =========================
def main(input_dir, output_dir, model_path, device=None):
    print(f"🏁 Starting sentiment analysis with input: {input_dir}, output: {output_dir}, model: {model_path}")
    os.makedirs(output_dir, exist_ok=True)

    # Load model from local or download and save
    start_time = time.time()
    tokenizer, config, model, selected_device = load_model(model_path, local_dir="./models", force_device=device)
    print(f"✅ Model setup completed in {time.time() - start_time:.2f} seconds")

    files = [f for f in os.listdir(input_dir) if f.endswith((".csv", ".xlsx"))]
    if not files:
        print("⚠️ Không có file CSV/XLSX trong input/")
        return

    for file in files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"sentiment_{file}")
        process_file(input_path, output_path, tokenizer, config, model, selected_device)

    print("🏁 Sentiment analysis completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch sentiment analysis (GPU optimized)")
    parser.add_argument("--input", required=True, help="Input folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--model", default="Khoa/sentiment-analysis-all-category-122024.8", help="Model path")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Force device (cpu or cuda), default is auto")
    args = parser.parse_args()

    try:
        main(args.input, args.output, args.model, args.device)
    except Exception as e:
        print(f"❌ Lỗi khi chạy: {e}")
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import time
from io import BytesIO

# =========================
# Function: Export DataFrame to Excel
# =========================
def export_to_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output.getvalue()


# =========================
# Label mapping
# =========================
label_mapping = {
    'POS': 'Positive',
    'NEG': 'Negative',
    'NEU': 'Neutral'
}


# =========================
# Function: Analyze with Model (optimized for GPU V100)
# =========================
@st.cache_resource  # cache model để không load lại nhiều lần
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16  # ép dùng half precision
    ).to("cuda")
    model.eval()
    # compile nếu có torch >= 2.0
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    return tokenizer, config, model


def analyze_with_model(df_filtered, model_path, num_rows, batch_size=32):
    tokenizer, config, model = load_model(model_path)

    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()

    texts = df_filtered["Content"].tolist()
    ids = df_filtered["Id"].tolist()

    all_raw_predicts = []
    all_sentiment_by_ai = []

    for i in range(0, num_rows, batch_size):
        batch_texts = texts[i: i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to("cuda")

        with torch.inference_mode():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

        for score in scores:
            ranking = np.argsort(score)[::-1]
            result = {}
            for k in range(len(score)):
                original_label = config.id2label[ranking[k]]
                mapped_label = label_mapping.get(original_label, original_label)
                result[mapped_label] = np.round(float(score[ranking[k]]), 4)
            top_label = label_mapping.get(config.id2label[ranking[0]], config.id2label[ranking[0]])
            all_raw_predicts.append(result)
            all_sentiment_by_ai.append(top_label)

        # update progress
        progress_bar.progress(min((i + batch_size) / num_rows, 1.0))
        status_text.text(f"Processing {min(i + batch_size, num_rows)}/{num_rows}")

    df_filtered["Raw Predict"] = all_raw_predicts
    df_filtered["Sentiment By AI"] = all_sentiment_by_ai

    elapsed_time = time.time() - start_time
    st.write(f"Processing Time: {elapsed_time:.2f} seconds")

    return df_filtered


# =========================
# Model list
# =========================
models = {
    "v122024.8 - latest": 'Khoa/sentiment-analysis-all-category-122024.8',
    "v122024.7": 'Khoa/sentiment-analysis-all-category-122024.7',
    "v122024.6": 'Khoa/sentiment-analysis-all-category-122024.6',
    "v122024.5": 'Khoa/sentiment-analysis-all-category-122024.5',
    "v122024.4": 'Khoa/sentiment-analysis-all-category-122024.4',
    "v122024.3": 'Khoa/sentiment-analysis-all-category-122024.3',
    "v122024.2": 'Khoa/sentiment-analysis-all-category-122024.2',
    "v122024": 'Khoa/sentiment-analysis-all-category-122024'
}


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="AI-Driven Sentiment Explorer", page_icon=":guardsman:", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://kompa.ai/assets/images/logo.svg", width=150)
    st.header("Model Configuration")
    project = st.selectbox("Select Model", list(models.keys()))
    uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])
    num_rows = st.number_input("Number of rows to process", min_value=1, value=100)
    batch_size = st.number_input("Batch size (GPU)", min_value=1, value=16, step=1)

# Custom CSS
st.markdown("""
    <style>
    .css-1d391kg {
        background-color: #ffffff;
    }
    .css-1d391kg .sidebar .sidebar-content {
        background-color: #F5F8FA;
    }
    .css-1d391kg .block-container {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("AI-Driven Sentiment Explorer")

# File upload
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    if 'Content' in df.columns:
        df = df.dropna(subset=['Content'])
        df = df.head(num_rows)

        if 'Sentiment' in df.columns:
            df.insert(df.columns.get_loc('Sentiment') + 1, 'Sentiment By AI', None)
            df_filtered = df[['Id', 'Content', 'Sentiment', 'Sentiment By AI']]

            data_json = df_filtered.to_dict(orient='records')
            with st.expander("Show JSON Data", expanded=False):
                st.json(data_json)

            if st.button('Run Sentiment Analysis'):
                selected_model_path = models[project]
                df_result = analyze_with_model(df_filtered, selected_model_path, num_rows, batch_size=batch_size)

                st.subheader("Sentiment Analysis Results")
                st.dataframe(df_result)

                xlsx_data = export_to_excel(df_result)
                st.download_button(
                    label="Download Excel file",
                    data=xlsx_data,
                    file_name=f"export-{time.time()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.error('File does not contain the "Sentiment" column.')
    else:
        st.error('File does not contain the "Content" column.')
else:
    st.info("Please upload a CSV or XLSX file to get started.")

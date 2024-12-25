import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import time
from io import BytesIO

# Function to export DataFrame to Excel
def export_to_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output.getvalue()

# Label mapping for sentiment analysis results
label_mapping = {
    'POS': 'Positive',
    'NEG': 'Negative',
    'NEU': 'Neutral'
}

# Function to load and analyze sentiment using any given model
def analyze_with_model(df_filtered, model_name, model_path, num_rows):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()

    def analyze_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            out = model(input_ids)
            scores = out.logits.softmax(dim=-1).cpu().numpy()[0]

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        result = {}
        for i in range(scores.shape[0]):
            original_label = config.id2label[ranking[i]]
            mapped_label = label_mapping.get(original_label, original_label)  # Map label
            score = scores[ranking[i]]
            result[mapped_label] = np.round(float(score), 4)
        top_label = label_mapping.get(config.id2label[ranking[0]], config.id2label[ranking[0]])
        return result, top_label

    raw_predicts = []
    sentiment_by_ai = []

    for i, (content, _id) in enumerate(zip(df_filtered['Content'], df_filtered['Id'])):
        result, top_label = analyze_sentiment(content)
        raw_predicts.append(result)
        sentiment_by_ai.append(top_label)

        df_filtered.loc[df_filtered['Id'] == _id, 'Sentiment By AI'] = top_label
        progress_bar.progress((i + 1) / num_rows)
        status_text.text(f"Processing {i + 1}/{num_rows}")

    df_filtered['Raw Predict'] = raw_predicts
    df_filtered['Sentiment By AI'] = sentiment_by_ai

    elapsed_time = time.time() - start_time
    elapsed_time_str = f"Processing Time: {elapsed_time:.2f} seconds"
    st.write(elapsed_time_str)


# List of predefined models
models = {
    "v122024.8 - latest": 'Khoa/sentiment-analysis-all-category-122024.8',
    "v122024.7": 'Khoa/sentiment-analysis-all-category-122024.7',
    "v122024.6": 'Khoa/sentiment-analysis-all-category-122024.6',
    "v122024.5": 'Khoa/sentiment-analysis-all-category-122024.5',
    "v122024.4": 'Khoa/sentiment-analysis-all-category-122024.4',
    "v122024.3": 'Khoa/sentiment-analysis-all-category-122024.3',
    "v122024.2": 'Khoa/sentiment-analysis-all-category-122024.2',
    "v122024": 'Khoa/sentiment-analysis-all-category-122024'
    # Add more models as needed
}

# Streamlit interface
st.set_page_config(page_title="AI-Driven Sentiment Explorer", page_icon=":guardsman:", layout="wide")

# Add logo at the top of the sidebar
with st.sidebar:
    st.image("https://kompa.ai/assets/images/logo.svg", width=150)  # Logo at the top of the sidebar
    st.header("Model Configuration")
    project = st.selectbox("Select Model", list(models.keys()))
    uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])
    num_rows = st.number_input("Number of rows to process", min_value=1, value=20)

# Customize background and sidebar colors using CSS
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

# Title for the page
st.title("AI-Driven Sentiment Explorer")

# Load file and process
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
                # Execute sentiment analysis for the selected model
                selected_model_path = models[project]
                analyze_with_model(df_filtered, project, selected_model_path, num_rows)

                st.subheader("Sentiment Analysis Results")
                st.dataframe(df_filtered)

                xlsx_data = export_to_excel(df_filtered)
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

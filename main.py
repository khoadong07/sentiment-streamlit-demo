import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import matplotlib.pyplot as plt
import seaborn as sns
import time
from io import BytesIO

hf_token = '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'


import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import time
from io import BytesIO

def export_to_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output.getvalue()

# Label mapping
label_mapping = {
    'POS': 'Positive',
    'NEG': 'Negative',
    'NEU': 'Neutral'
}

# Define project functions
def project_golden_gate_analysis(df_filtered, num_rows):
    model_path = 'Khoa/sentiment-analysis-tuning-golden-gate-0924-1'
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    config = AutoConfig.from_pretrained(model_path, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, token=hf_token)
    run_sentiment_analysis(df_filtered, tokenizer, model, config, label_mapping, num_rows)

def all_project(df_filtered, num_rows):
    model_path = 'Khoa/sentiment-analysis-all-category-122024.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, token=hf_token)
    run_sentiment_analysis(df_filtered, tokenizer, model, config, label_mapping, num_rows)

def project_vnmilk_analysis(df_filtered, num_rows):
    model_path = 'Khoa/sentiment-analysis-tuning-vnmilk'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    run_sentiment_analysis(df_filtered, tokenizer, model, config, label_mapping, num_rows)

# def project_4_analysis(df_filtered, num_rows):
#     model_path = 'project4/model-path'
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     config = AutoConfig.from_pretrained(model_path)
#     model = AutoModelForSequenceClassification.from_pretrained(model_path)
#     run_sentiment_analysis(df_filtered, tokenizer, model, config, label_mapping, num_rows)

def run_sentiment_analysis(df_filtered, tokenizer, model, config, label_mapping, num_rows):
    # Display progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Record the start time
    start_time = time.time()

    def analyze_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            out = model(input_ids)
            scores = out.logits.softmax(dim=-1).cpu().numpy()[0]

        # Process results
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

    # Apply sentiment analysis to "Content" column
    raw_predicts = []
    sentiment_by_ai = []

    for i, (content, _id) in enumerate(zip(df_filtered['Content'], df_filtered['Id'])):
        result, top_label = analyze_sentiment(content)
        raw_predicts.append(result)
        sentiment_by_ai.append(top_label)

        # Update the new column in df based on Id
        df_filtered.loc[df_filtered['Id'] == _id, 'Sentiment By AI'] = top_label

        # Update progress bar
        progress_bar.progress((i + 1) / num_rows)
        status_text.text(f"Processing {i + 1}/{num_rows}")

    # Add results to DataFrame
    df_filtered['Raw Predict'] = raw_predicts
    df_filtered['Sentiment By AI'] = sentiment_by_ai

    # Record the end time and calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_time_str = f"Processing Time: {elapsed_time:.2f} seconds"

    # Display elapsed time
    st.write(elapsed_time_str)

# Streamlit interface
st.title("Kompa AI - Sentiment Analysis")

# Define the available projects
projects = {
    "All": all_project,
    "Golden Gate": project_golden_gate_analysis,
    "Vinamilk": project_vnmilk_analysis,
    # "Project 3": project_3_analysis,
    # "Project 4": project_4_analysis,
}

# Sidebar for project selection
project = st.sidebar.selectbox("Select Project", list(projects.keys()))

uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])

# Option to select number of rows to process
num_rows = st.number_input("Number of rows to process", min_value=1, value=20)

if uploaded_file is not None:
    # Read file CSV or XLSX
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    # Drop rows where "Content" is NaN
    if 'Content' in df.columns:
        df = df.dropna(subset=['Content'])
        df = df.head(num_rows)  # Select specified number of rows

        # Check if "Sentiment" column exists
        if 'Sentiment' in df.columns:
            # Add new column "Sentiment By AI" with None as default values
            df.insert(df.columns.get_loc('Sentiment') + 1, 'Sentiment By AI', None)

            # Filter columns
            df_filtered = df[['Id', 'Content', 'Sentiment', 'Sentiment By AI']]

            # Convert DataFrame to list of dictionaries
            data_json = df_filtered.to_dict(orient='records')

            # Display JSON data in a collapsible section
            with st.expander("Show JSON Data", expanded=False):
                st.json(data_json)

            # Run sentiment analysis when button is clicked
            if st.button('Run Sentiment Analysis'):
                # Execute sentiment analysis based on selected project
                projects[project](df_filtered, num_rows)

                # Display sentiment analysis results
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

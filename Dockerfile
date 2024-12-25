# Use the official Python base image
FROM python:3.9.19-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

COPY .streamlit /root/.streamlit/

# Run Streamlit
ENTRYPOINT ["streamlit", "run"]

# Set the default Streamlit application to run
CMD ["main.py"]

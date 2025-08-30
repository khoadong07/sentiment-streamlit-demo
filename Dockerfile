# Sử dụng NVIDIA PyTorch image (có sẵn CUDA, cuDNN, PyTorch GPU)
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Đặt working directory
WORKDIR /app

# Copy requirements trước để cache layer
COPY requirements.txt .

# Cài các dependency
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# Mở port Streamlit
EXPOSE 8501

# Run Streamlit
ENTRYPOINT ["streamlit", "run"]

# File mặc định
CMD ["main.py"]

FROM python:3.9-slim

# Cài đặt các thư viện cần thiết
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

# Expose port và chạy Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.enableCORS=false"]
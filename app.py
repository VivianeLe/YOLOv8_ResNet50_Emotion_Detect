import os
from pyngrok import ngrok

# Khởi động ngrok tunnel
port = 8501  # Port mặc định của Streamlit
try:
    public_url = ngrok.connect(port)
    print(f"Ngrok Tunnel URL: {public_url}")
except Exception as e:
    print(f"Ngrok Error: {e}")
    print("Ensure you have closed other tunnels or upgrade your account.")

# Chạy Streamlit
os.system(f"streamlit run main.py --server.port {port}")
# ============================================================
# 1. Base Python Image
# A minimal Linux base environment + Python 3.11 and the components needed for that 
# Python runtime.
# ============================================================

FROM python:3.11-slim

# ============================================================
# 2. Set Working Directory
#From this point onward, use app as the working directory/folder inside the container."
# ============================================================

WORKDIR /app

# ============================================================
# 3. Copy Requirements

# Copy all the Dependencies in requirements.txt into the current working directory.
# ============================================================

COPY requirements.txt .

# ============================================================
# 4. Install Python Dependencies
# Install all the Dependencies 
# This is where Docker actually installs your Python dependencies inside the image.
# ============================================================

RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# 5. Copy all the Project Files
# ============================================================

COPY . .

# ============================================================
# 6. Expose Streamlit Port
#A port is a numbered communication endpoint that allows a specific application
# or service to send and receive network traffic.

#"Its like saying On this computer, connect me to the application listening on port 8501."
# ============================================================

EXPOSE 8501

# ============================================================
# 7. Start Streamlit Application
# when someone runs the image start my Streamlit application using this command."
# ============================================================

CMD ["streamlit", "run", "streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
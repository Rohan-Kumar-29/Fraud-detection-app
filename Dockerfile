# Use Python 3.10 (stable for sklearn, pandas, etc.)
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (to use Docker caching properly)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files (your app, model, etc.)
COPY . .

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]

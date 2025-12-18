# Use the official Python image as a base
FROM python:3.14-slim

# Set the working directory in the container
WORKDIR /app


# Install dependencies from the requirements.txt file
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY ./benchmark_ui.py /app/streamlit_ui.py
COPY ./tts_bench.db /app/tts_bench.db

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_ui.py"]
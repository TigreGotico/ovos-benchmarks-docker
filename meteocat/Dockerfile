# Use the official Python image as a base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app


# Install dependencies from the requirements.txt file
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY ./meteocat_ui.py /app/streamlit_ui.py
COPY ./intents_meteocat.db /app/intents_meteocat.db

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_ui.py"]
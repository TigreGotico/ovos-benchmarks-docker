import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, Column, String, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from collections import Counter
import re

Base = declarative_base()


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id = Column(Integer, primary_key=True, index=True)
    sample_id = Column(String, unique=True, index=True)
    dataset = Column(String)
    language = Column(String)
    task = Column(String)  # STT, TTS, Intent, Wakeword
    plugin = Column(String)
    model = Column(String)
    input_text = Column(String, nullable=True)
    prediction = Column(String, nullable=True)
    ground_truth = Column(String, nullable=True)


def get_engine(name: str):
    DATABASE_URL = f"sqlite:///{name}.db"  # Change to PostgreSQL if needed
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return engine


# Connect to DB
engine = get_engine("intents_meteocat")
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# recalculate_metrics(db)
st.title("Meteocat Benchmarks")

st.write(f"Meteocat dataset was tested against a live OVOS instance: https://huggingface.co/datasets/crodri/meteocat")
st.write(f"We do not have intent labels and only check if the weather skill matched, not the specific intent within the skill")


# Fetch data
query = """
SELECT plugin, input_text, prediction
FROM benchmark_results
"""
df = pd.read_sql(query, con=engine)

# Sidebar filters
st.sidebar.header("Filters")
plugin_filter = st.sidebar.selectbox("Select Match Type", ["All"] + sorted(df["plugin"].dropna().unique().tolist()))
prediction_filter = st.sidebar.selectbox("Select Prediction", ["All"] + sorted(df["prediction"].dropna().unique().tolist()))

if plugin_filter != "All":
    df = df[df["plugin"] == plugin_filter]
if prediction_filter != "All":
    df = df[df["prediction"] == prediction_filter]

# Compute Intent Match
df["matched"] = df["prediction"].astype(str).str.startswith("ovos-skill-weather")
intent_accuracy = df["matched"].mean() if not df.empty else 0

# Display filtered results
st.write(f"### Predictions ({len(df)} samples)")
st.dataframe(df)


# Summary
st.write("### Overall Metrics")
st.write(f"**Weather Skill Matches:** {intent_accuracy:.2%}")

# Compute most common words, bigrams, and trigrams
if "input_text" in df.columns and not df["input_text"].isna().all():
    text_data = " ".join(df["input_text"].dropna()).lower()  # Concatenate all text and lowercase
    words = re.findall(r"\b\w{4,}\b", text_data)  # Extract words using regex

    # Helper function to generate n-grams
    def generate_ngrams(word_list, n):
        return [" ".join(word_list[i:i+n]) for i in range(len(word_list) - n + 1)]

    # Compute frequencies
    word_counts = Counter(words).most_common(10)
    bigram_counts = Counter(generate_ngrams(words, 2)).most_common(10)
    trigram_counts = Counter(generate_ngrams(words, 3)).most_common(10)

    # Convert to DataFrame for display
    word_df = pd.DataFrame(word_counts, columns=["Word", "Frequency"])
    bigram_df = pd.DataFrame(bigram_counts, columns=["Bigram", "Frequency"])
    trigram_df = pd.DataFrame(trigram_counts, columns=["Trigram", "Frequency"])

    # Display results
    st.write("### 10 Most Common Words in Input Text")
    st.dataframe(word_df)

    st.write("### 10 Most Common Bigrams")
    st.dataframe(bigram_df)

    st.write("### 10 Most Common Trigrams")
    st.dataframe(trigram_df)

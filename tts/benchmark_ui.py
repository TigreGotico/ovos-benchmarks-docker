import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, Column, String, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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
    wer_score = Column(Float, nullable=True)
    levenshtein_score = Column(Float, nullable=True)
    rtf = Column(Float, nullable=True)
    pitch_variability = Column(Float, nullable=True)


def get_engine(name: str):
    DATABASE_URL = f"sqlite:///{name}.db"  # Change to PostgreSQL if needed
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return engine


# Connect to DB
engine = get_engine("tts_bench")
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()
# recalculate_metrics(db)
st.title("OpenVoiceOS TTS Plugin Benchmark")

# Fetch data
query = """
SELECT language, plugin, model, rtf, pitch_variability, wer_score, levenshtein_score
FROM benchmark_results
"""
df = pd.read_sql(query, con=engine)

# Sidebar filters
st.sidebar.header("Filters")

# Language selection
language_filter = st.sidebar.selectbox("Select Language", ["All"] + sorted(df["language"].unique().tolist()))

# Plugin selection
plugin_filter = st.sidebar.selectbox("Select Plugin", ["All"] + sorted(df["plugin"].unique().tolist()))

if language_filter != "All":
    df = df[df["language"] == language_filter]
if plugin_filter != "All":
    df = df[df["plugin"] == plugin_filter]

# Calculate average per plugin
avg_scores = df.groupby('plugin').agg({
    'rtf': 'mean',
    'pitch_variability': 'mean',
    'wer_score': 'mean',
    'levenshtein_score': 'mean'

}).reset_index()
st.write("### Plugin Comparison")
comparison_table = avg_scores.set_index('plugin')
st.dataframe(comparison_table)
st.write("- the lower the pitch_variability the more robotic the voice")
st.write("- lower WER / higher levenshtein_score means a more intelligible voice")

# Show filtered results
st.write(f"### Benchmark Results ({len(df)} samples)")
st.dataframe(df)

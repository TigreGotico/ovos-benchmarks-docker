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


def get_engine(name: str):
    DATABASE_URL = f"sqlite:///{name}.db"  # Change to PostgreSQL if needed
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return engine


# Connect to DB
engine = get_engine("stt_bench")
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()
# recalculate_metrics(db)
st.title("OpenVoiceOS STT Plugin Benchmark")

# Fetch data
query = """
SELECT dataset, language, plugin, model, prediction, ground_truth, 
       wer_score, levenshtein_score 
FROM benchmark_results
"""
df = pd.read_sql(query, con=engine)

# Sidebar filters
st.sidebar.header("Filters")

# Task selection (Global)
# task_filter = st.sidebar.selectbox("Select Task", ["All"] + sorted(df["task"].unique().tolist()))

# Language selection
language_filter = st.sidebar.selectbox("Select Language", ["All"] + sorted(df["language"].unique().tolist()))

# Dataset selection
dataset_filter = st.sidebar.selectbox("Select Dataset", ["All"] + sorted(df["dataset"].unique().tolist()))

# Plugin selection
plugin_filter = st.sidebar.selectbox("Select Plugin", ["All"] + sorted(df["plugin"].unique().tolist()))

# Model selection
model_filter = st.sidebar.selectbox("Select Model", ["All"] + sorted(df["model"].unique().tolist()))

# Apply filters
# if task_filter != "All":
#    df = df[df["task"] == task_filter]
if language_filter != "All":
    df = df[df["language"] == language_filter]
if dataset_filter != "All":
    df = df[df["dataset"] == dataset_filter]
if plugin_filter != "All":
    df = df[df["plugin"] == plugin_filter]
if model_filter != "All":
    df = df[df["model"] == model_filter]

# Show filtered results
st.write(f"### Filtered Benchmark Results ({len(df)} samples)")
st.dataframe(df)

# Calculate average WER and Levenshtein per plugin
avg_scores = df.groupby('plugin').agg({
    'wer_score': 'mean',
    'levenshtein_score': 'mean'
}).reset_index()

# Display the comparison table with plugins as rows and WER/Levenshtein as columns
st.write("### Plugin Comparison (Average WER and Levenshtein Score)")
comparison_table = avg_scores.set_index('plugin')

# Show the comparison table with WER and Levenshtein scores
st.dataframe(comparison_table)

# Create a new column 'plugin_model' by concatenating 'plugin' and 'model'
df['plugin_model'] = df['plugin'] + '/' + df['model']

# Calculate average WER per plugin and model
avg_wer = df.groupby(['plugin_model', 'dataset'])['wer_score'].mean().reset_index()

# Display the comparison table
st.write("### WER Comparison Between Plugins and Datasets")
comparison_table = avg_wer.pivot_table(index='plugin_model',
                                       columns=['dataset'],
                                       values='wer_score', aggfunc='mean')
# Show the comparison table with plugins and models
st.dataframe(comparison_table)

st.write("### Levenshtein Similarity Comparison Between Plugins and Datasets")
avg_levenshtein = df.groupby(['plugin_model', 'dataset'])['levenshtein_score'].mean().reset_index()

comparison_table2 = avg_levenshtein.pivot_table(index='plugin_model',
                                                columns=['dataset'],
                                                values='levenshtein_score', aggfunc='mean')
st.dataframe(comparison_table2)

# Display summary metrics
st.write("### Overall Metrics")
if not df.empty:
    if "wer_score" in df:
        st.write(f"**Average WER:** {df['wer_score'].mean():.2f}")
    if "levenshtein_score" in df:
        st.write(f"**Average Levenshtein Similarity:** {df['levenshtein_score'].mean():.2f}")
else:
    st.write("No data available for the selected filters.")

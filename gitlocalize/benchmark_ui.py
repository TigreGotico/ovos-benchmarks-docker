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


def get_engine(name: str):
    DATABASE_URL = f"sqlite:///{name}.db"  # Change to PostgreSQL if needed
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return engine


# Connect to DB
engine = get_engine("intents_bench")
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# recalculate_metrics(db)
st.title("Padatious Benchmarks")

st.write(f"A Dataset exported from gitlocalize was used to test all padatious intents against a live OVOS instance and generate these benchmarks")
st.write(f"**NOTE**: this dataset corresponds to the **training** data, high score is expected!")
st.write(f"This is meant to identify issues in the training data but **does not provide an overview of real world accuracy**")

# Fetch data
query = """
SELECT language, plugin, input_text, prediction, ground_truth
FROM benchmark_results
"""
df = pd.read_sql(query, con=engine)

# Sidebar filters
st.sidebar.header("Filters")
language_filter = st.sidebar.selectbox("Select Language", ["All"] + sorted(df["language"].dropna().unique().tolist()))
plugin_filter = st.sidebar.selectbox("Select Match Type", ["All"] + sorted(df["plugin"].dropna().unique().tolist()))
intent_filter = st.sidebar.selectbox("Select Intent", ["All"] + sorted(df["ground_truth"].dropna().unique().tolist()))
prediction_filter = st.sidebar.selectbox("Select Prediction", ["All"] + sorted(df["prediction"].dropna().unique().tolist()))

# Apply filters
if language_filter != "All":
    df = df[df["language"] == language_filter]
if plugin_filter != "All":
    df = df[df["plugin"] == plugin_filter]
if intent_filter != "All":
    df = df[df["ground_truth"] == intent_filter]
if prediction_filter != "All":
    df = df[df["prediction"] == prediction_filter]

# Compute Intent Match
df["matched"] = df["prediction"] == df["ground_truth"]
intent_accuracy = df["matched"].mean() if not df.empty else 0

# Display filtered results
st.write(f"### Predictions ({len(df)} samples)")
st.dataframe(df)

# Compute metrics per plugin
plugin_metrics = df.groupby("ground_truth")["matched"].mean().reset_index()
plugin_metrics.rename(columns={"matched": "accuracy"}, inplace=True)

st.write("### Intent Accuracy Comparison")
st.write("For each intent, how often did it match correctly?")
st.dataframe(plugin_metrics)

# Compute metrics per plugin
plugin_metrics = df.groupby("prediction")["matched"].mean().reset_index()
plugin_metrics.rename(columns={"matched": "accuracy"}, inplace=True)

st.write("### Prediction Accuracy Comparison")
st.write("For each prediction, how often was it correct?")
st.dataframe(plugin_metrics)

# Compute metrics per plugin
plugin_metrics2 = df.groupby("language")["matched"].mean().reset_index()
plugin_metrics2.rename(columns={"matched": "accuracy"}, inplace=True)

st.write("### Language Accuracy Comparison")
st.dataframe(plugin_metrics2)
# Summary
st.write("### Overall Metrics")
st.write(f"**Overall Intent Accuracy:** {intent_accuracy:.2%}")

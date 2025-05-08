import os, sys

# Force worker and driver to use this interpreter
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYSPARK_PYTHON"]        = sys.executable

import os, sys, pandas as pd
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lyrics Genre Classifier", layout="centered")
MODEL_PATH = "model_stage4_merged_Trans_way_new"

# 1️  Bring up Spark + load the fitted pipeline  (cached once per run)
@st.cache_resource  # survives Streamlit reruns
def init_spark_and_model():
    PY = sys.executable                    # active conda-env Python
    spark = (SparkSession.builder
             .appName("LyricsUI")
             .config("spark.sql.shuffle.partitions", "1")   # tiny job speed-up
             .config("spark.pyspark.python", PY)
             .config("spark.pyspark.driver.python", PY)
             .getOrCreate())

    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    model = PipelineModel.load(model_path)

    # grab label list from the StringIndexerModel stage
    label_stage = next(s for s in model.stages
                       if "StringIndexerModel" in s.__class__.__name__)
    idx2label = label_stage.labels
    return spark, model, idx2label

spark, model, idx2label = init_spark_and_model()


# 2️  Predict helper (Spark call)   +  Streamlit cache for repeats
def _predict_once(txt: str):
    pdf = pd.DataFrame({"artist_name": ["x"],
                        "track_name":  ["x"],
                        "release_date": ["2000"],
                        "genre": ["dummy"],
                        "lyrics": [txt]})

    sdf = spark.createDataFrame(pdf)
    res = model.transform(sdf).select("probability", "prediction").collect()[0]
    probs = res["probability"].toArray().tolist()
    pred  = idx2label[int(res["prediction"])]
    return probs, pred

@st.cache_data(show_spinner=False)
def predict_cached(txt: str):
    return _predict_once(txt)

# 3. Streamlit front-end
# st.set_page_config(page_title="Lyrics Genre Classifier", layout="centered")
st.title("🎶 Spark-ML Lyrics Classifier")
st.markdown("MLlib and Visualisation Homework | 200623P")
st.markdown("Paste song lyrics below and click **Predict**:")

lyrics_text = st.text_area("Lyrics", height=200)

if st.button("Predict", type="primary", use_container_width=True):
    if not lyrics_text.strip():
        st.warning("Please paste some lyrics first.")
    else:
        with st.spinner("Classifying the genre using SPARK"):
            probs, pred_g = predict_cached(lyrics_text)

        st.success(f"**Prediction → {pred_g.upper()}**") 

        # bar-chart & pie-chart side by side
        col1, col2 = st.columns(2)

        with col1: #bar chart
            chart_df = (pd.DataFrame({"Genre": idx2label,
                                    "Probability": probs})
                        .set_index("Genre"))
            st.bar_chart(chart_df, use_container_width=True)

        with col2: # pie chart
            pie_labels = [pred_g.upper(), "Other genres"]
            pie_sizes  = [probs[idx2label.index(pred_g)],
                        1 - probs[idx2label.index(pred_g)]]

            fig, ax = plt.subplots()
            ax.pie(pie_sizes,
                labels=pie_labels,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"linewidth": 1, "edgecolor": "white"})
            ax.axis("equal")
            st.pyplot(fig, use_container_width=True)

        # st.success(f"**Prediction → {pred_g.upper()}**")

st.markdown("---")
st.caption("200623P | SUBODHA KRA")

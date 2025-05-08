# # webui.py  –  Streamlit front-end for the 8-genre lyrics classifier
# # ---------------------------------------------------------------
# import sys, os
# from pyspark.sql import SparkSession
# from pyspark.ml import PipelineModel
# import pandas as pd
# import streamlit as st

# # ---- Spark session -------------------------------------------------
# PY = sys.executable                     # same interpreter for driver & workers
# spark = (SparkSession.builder
#          .appName("LyricsUI")
#          .config("spark.pyspark.python", PY)
#          .config("spark.pyspark.driver.python", PY)
#          .getOrCreate())

# # ---- Load fitted pipeline ------------------------------------------
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_stage3_merged")
# model      = PipelineModel.load(MODEL_PATH)

# # recover the string labels from the StringIndexerModel
# label_stage = [s for s in model.stages if "StringIndexerModel" in s.__class__.__name__][0]
# idx2label   = label_stage.labels

# # ---- Streamlit page -----------------------------------------------
# st.set_page_config(page_title="Lyrics Genre Classifier", layout="centered")
# st.title("🎶 Spark-ML Lyrics Classifier")
# st.markdown("Paste song lyrics, hit **Predict**, and see the genre probabilities:")

# lyrics_text = st.text_area("Lyrics", height=200)

# if st.button("Predict"):
#     # build a single-row Spark DF
#     pdf = pd.DataFrame({
#         "artist_name":   ["x"],
#         "track_name":    ["x"],
#         "release_date":  ["2000"],
#         "genre":         ["dummy"],
#         "lyrics":        [lyrics_text]
#     })
#     sdf = spark.createDataFrame(pdf)

#     result = model.transform(sdf).select("probability", "prediction").collect()[0]
#     probs  = result["probability"].toArray().tolist()
#     pred_i = int(result["prediction"])
#     pred_g = idx2label[pred_i]

#     # bar-chart
#     chart_df = pd.DataFrame({"Genre": idx2label, "Probability": probs})
#     st.bar_chart(chart_df.set_index("Genre"))

#     st.success(f"**Predicted genre → {pred_g.upper()}**")

# st.markdown("---")
# st.caption("Powered by PySpark + Streamlit")



# webui.py  ──────────────────────────────────────────────────────────────
# Streamlit UI for the 8-genre Spark-ML lyrics classifier
# • One-click prediction + bar-chart
# • Caches Spark session & model so first hit ~3 s, later hits ~0.1 s
# -----------------------------------------------------------------------

import os, sys, pandas as pd
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

st.set_page_config(page_title="Lyrics Genre Classifier", layout="centered")
MODEL_PATH = "model_stage4_merged_Trans_way"

# ════════════════════════════════════════════════════════════════════════
# 1️⃣  Bring up Spark + load the fitted pipeline  (cached once per run)
# ════════════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════════════
# 2️⃣  Predict helper (Spark call)   +  Streamlit cache for repeats
# ════════════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════════════
# 3. Streamlit front-end
# ════════════════════════════════════════════════════════════════════════
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

        # bar-chart
        chart_df = (pd.DataFrame({"Genre": idx2label,
                                  "Probability": probs})
                    .set_index("Genre"))
        st.bar_chart(chart_df)

        st.success(f"**Prediction → {pred_g.upper()}**")

st.markdown("---")
st.caption("200623P | SUBODHA KRA")

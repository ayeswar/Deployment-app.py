import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="World Development Clustering", layout="wide")

MODEL_DIR = "clustering_artifacts"

def safe_load(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(path):
        return joblib.load(path)
    return None

scaler = safe_load("scaler.joblib")
model = safe_load("chosen_model.joblib")
pca2 = safe_load("pca2.joblib")

if scaler is None or model is None:
    st.error(" Model artifacts missing. Please upload the 'clustering_artifacts' folder.")
    st.stop()


st.title(" World Development Clustering Dashboard")
st.markdown("Upload your dataset (must contain same numeric columns as training).")

uploaded_file = st.file_uploader(" Upload CSV File", type=["csv"])

if uploaded_file is not None:

   
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f" Error reading CSV: {e}")
        st.stop()

    st.subheader(" Uploaded Data Preview")
    st.dataframe(df.head())

  
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.error(" No numeric columns found.")
        st.stop()

   
    expected_cols = scaler.mean_.shape[0]
    if numeric_df.shape[1] != expected_cols:
        st.error(f"""
          Column Mismatch!
        Expected **{expected_cols}** numeric columns, got **{numeric_df.shape[1]}**.
        Make sure your CSV has the exact same numeric columns as training.
        """)
        st.stop()

    #  Scale Data
    try:
        X_scaled = scaler.transform(numeric_df)
    except Exception as e:
        st.error(f" Scaling failed: {e}")
        st.stop()

    #  Predict Clusters
    try:
        labels = model.predict(X_scaled)
    except Exception as e:
        st.error(f" Prediction failed: {e}")
        st.stop()

    df["Cluster"] = labels

    st.success(f" Clustering Successful! Found **{len(set(labels))}** clusters.")
    st.dataframe(df.head())

    #  Cluster Distribution Chart
    st.subheader(" Cluster Distribution")
    st.bar_chart(df["Cluster"].value_counts())

    #  PCA Visualization
    if pca2:
        try:
            pcs = pca2.transform(X_scaled)
            pcs_df = pd.DataFrame({"PC1": pcs[:,0], "PC2": pcs[:,1], "Cluster": labels})

            st.subheader(" PCA Visualization")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(
                data=pcs_df,
                x="PC1", y="PC2",
                hue="Cluster",
                palette="tab10",
                s=80,
                ax=ax
            )
            plt.title("PCA - Cluster Visualization")
            st.pyplot(fig)
        except Exception as e:
            st.error(f" PCA Plot Error: {e}")

    #  Download Output
    st.subheader("⬇ Download Clustered File")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "clustered_output.csv", "text/csv")

else:
    st.info(" Please upload a CSV file to begin.")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Heart Risk Dashboard", page_icon="", layout="wide")


PASTEL_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #fdf2f8 0%, #eef2ff 50%, #ecfeff 100%);
}
.title-card {
    background: #ffffffcc;
    border: 1px solid #e9d5ff;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.result-good {
    background: #dcfce7;
    border-left: 6px solid #22c55e;
    padding: 1.2rem;
    border-radius: 12px;
    font-size: 1.05rem;
}
.result-bad {
    background: #fee2e2;
    border-left: 6px solid #ef4444;
    padding: 1.2rem;
    border-radius: 12px;
    font-size: 1.05rem;
}
.nurse-box {
    background: #ffffffcc;
    border: 1px solid #c4b5fd;
    border-radius: 12px;
    padding: 1rem;
}
</style>
"""
st.markdown(PASTEL_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("heart_disease_uci.csv")
    keep_cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalch",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
        "dataset",
    ]
    df = df[keep_cols].copy()
    df = df[df["dataset"].str.lower() == "cleveland"].copy()
    df = df.drop(columns=["dataset"])
    df = df.rename(columns={"thalch": "thalach"})
    df = df.replace("?", np.nan)

    for c in ["trestbps", "chol", "thalach", "oldpeak", "ca"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    df["target"] = (df["num"] > 0).astype(int)
    return df


@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop(columns=["num", "target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cat_cols = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Keep categorical values consistently as strings for robust UI/model compatibility.
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cat_cols] = X_train[cat_cols].astype(str)
    X_test[cat_cols] = X_test[cat_cols].astype(str)

    prep = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("prep", prep),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    # prefilled example from real test patient
    prefill = X_test.iloc[0].to_dict()
    return model, X_train, prefill


def build_input_df(defaults, X_train):
    st.markdown("### Patient Input Form")
    st.caption("All 13 features are required. Hints show expected practical ranges.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (20-80)", 20, 80, int(defaults["age"]))
        sex = st.selectbox("Sex (Male/Female)", sorted(X_train["sex"].astype(str).unique()), index=sorted(X_train["sex"].astype(str).unique()).index(str(defaults["sex"])))
        cp = st.selectbox("Chest Pain Type (cp)", sorted(X_train["cp"].astype(str).unique()), index=sorted(X_train["cp"].astype(str).unique()).index(str(defaults["cp"])))
        trestbps = st.number_input("Resting BP (80-220)", 80, 220, int(round(float(defaults["trestbps"]))))
        chol = st.number_input("Cholesterol (100-600)", 100, 600, int(round(float(defaults["chol"]))))

    with c2:
        fbs = st.selectbox("Fasting Blood Sugar >120 (True/False)", sorted(X_train["fbs"].astype(str).unique()), index=sorted(X_train["fbs"].astype(str).unique()).index(str(defaults["fbs"])))
        restecg = st.selectbox("Resting ECG (restecg)", sorted(X_train["restecg"].astype(str).unique()), index=sorted(X_train["restecg"].astype(str).unique()).index(str(defaults["restecg"])))
        thalach = st.number_input("Max Heart Rate (70-210)", 70, 210, int(round(float(defaults["thalach"]))))
        exang = st.selectbox("Exercise Induced Angina (True/False)", sorted(X_train["exang"].astype(str).unique()), index=sorted(X_train["exang"].astype(str).unique()).index(str(defaults["exang"])))
        oldpeak = st.number_input("ST Depression oldpeak (0.0-7.0)", 0.0, 7.0, float(defaults["oldpeak"]), step=0.1)

    with c3:
        slope = st.selectbox("ST Slope (slope)", sorted(X_train["slope"].astype(str).unique()), index=sorted(X_train["slope"].astype(str).unique()).index(str(defaults["slope"])))
        ca = st.number_input("Major Vessels (ca: 0-4)", 0.0, 4.0, float(defaults["ca"]), step=1.0)
        thal = st.selectbox("Thal (thal)", sorted(X_train["thal"].astype(str).unique()), index=sorted(X_train["thal"].astype(str).unique()).index(str(defaults["thal"])))

    row = {
        "age": age,
        "sex": str(sex),
        "cp": str(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": str(fbs),
        "restecg": str(restecg),
        "thalach": thalach,
        "exang": str(exang),
        "oldpeak": oldpeak,
        "slope": str(slope),
        "ca": ca,
        "thal": str(thal),
    }
    return pd.DataFrame([row])


def top3_drivers(model, x_input):
    prep = model.named_steps["prep"]
    rf = model.named_steps["rf"]
    x_t = prep.transform(x_input)
    if hasattr(x_t, "toarray"):
        x_t = x_t.toarray()
    x_vec = x_t[0]

    feat_names = prep.get_feature_names_out()
    imp = rf.feature_importances_

    contrib = np.abs(x_vec * imp)
    agg = {}
    for name, val in zip(feat_names, contrib):
        raw = name.split("__", 1)[1]
        base = raw.split("_", 1)[0] if name.startswith("cat__") else raw
        agg[base] = agg.get(base, 0.0) + float(val)

    top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return pd.DataFrame(top, columns=["Feature", "Score"])


def explanation_text(pred, top_df):
    f = top_df["Feature"].tolist()
    if pred == 1:
        return (
            f"This patient is flagged as higher cardiac risk. The strongest contributing factors are "
            f"{f[0]}, {f[1]}, and {f[2]}. Please prioritize clinical review and confirm with physician-led assessment."
        )
    return (
        f"This patient is predicted as lower immediate cardiac risk. The model mainly considered "
        f"{f[0]}, {f[1]}, and {f[2]}. Continue routine monitoring and clinical judgment as normal."
    )


def main():
    st.markdown('<div class="title-card"><h2> Heart Disease Local Dashboard</h2><p>quick risk prediction and explanation</p></div>', unsafe_allow_html=True)

    model, X_train, defaults = train_model()
    x_input = build_input_df(defaults, X_train)

    if st.button("Predict", type="primary", use_container_width=True):
        prob = float(model.predict_proba(x_input)[0, 1])
        pred = 1 if prob >= 0.5 else 0
        conf = prob if pred == 1 else (1 - prob)

        top_df = top3_drivers(model, x_input)
        st.markdown("### Results Panel")
        col_graph, col_result = st.columns([1.0, 1.3], gap="large")

        with col_graph:
            st.write("#### Top 3 Features Driving Prediction")
            colors = ["#c4b5fd", "#93c5fd", "#86efac"]
            fig, ax = plt.subplots(figsize=(4.6, 2.3))
            bars = ax.barh(top_df["Feature"][::-1], top_df["Score"][::-1], color=colors)
            ax.set_xlabel("Relative Importance")
            ax.set_ylabel("Feature")
            ax.set_title("Top Feature Drivers", fontsize=10)
            legend_handles = [
                Patch(facecolor=bar.get_facecolor(), label=feature)
                for bar, feature in zip(bars[::-1], top_df["Feature"])
            ]
            ax.legend(handles=legend_handles, title="Legend", fontsize=8, title_fontsize=9, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)

        with col_result:
            if pred == 1:
                st.markdown(
                    f'<div class="result-bad"><b>Predicted Class:</b> Disease Present 🔴<br><b>Confidence:</b> {conf*100:.1f}%</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-good"><b>Predicted Class:</b> No Disease 🟢<br><b>Confidence:</b> {conf*100:.1f}%</div>',
                    unsafe_allow_html=True,
                )

            st.write("#### Nurse-Friendly Explanation")
            st.markdown(
                f'<div class="nurse-box">{explanation_text(pred, top_df)}</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

"""
LMS Student Engagement Monitor
Streamlit app that deploys the Random Forest model from main.ipynb.
Predicts whether a student is in the High or Low rating category from midterm
LMS activity logs, so instructors can spot students who may need support.
"""
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="LMS Engagement Monitor", page_icon=":material/school:", layout="wide")

# Center result pop-ups vertically (Streamlit pins them to the top by default).
# Auto margins keep tall pop-ups scrollable instead of cutting off their top.
st.markdown(
    """
    <style>
    .stDialog > div { margin-top: auto !important; margin-bottom: auto !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path("model.joblib")
DATA_PATH = Path("augmented_balanced_dataset_m.csv")

# Input columns with the descriptions from the dataset documentation
FEATURES = {
    "Submission.Actions": "Submission-related actions",
    "Quiz.Actions": "Quiz-related actions",
    "Forum.Actions": "Forum-related actions",
    "Create.Actions": "Create-related actions",
    "Read.Actions": "Read-related actions",
    "Update.Actions": "Update-related actions",
    "Delete.Actions": "Delete-related actions",
    "Days.Logged.In": "Days the student logged into the LMS",
}
INPUTS = list(FEATURES)
GROUPS = {
    "Assessment": ["Submission.Actions", "Quiz.Actions", "Forum.Actions"],
    "Content interaction": ["Create.Actions", "Read.Actions", "Update.Actions", "Delete.Actions"],
    "Attendance": ["Days.Logged.In"],
}


# ---------------------------------------------------------------- helpers
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        return bundle["model"], bundle["columns"]
    return bundle, list(bundle.feature_names_in_)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    # Must match main.ipynb exactly so the train/test split is the same
    return pd.read_csv(DATA_PATH)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as the notebook."""
    X = df.drop(columns="rating", errors="ignore").copy()
    days = X["Days.Logged.In"].clip(lower=1)
    total = X[INPUTS].drop(columns="Days.Logged.In").sum(axis=1)
    X["Total.Actions"] = total
    X["Actions.Per.Day"] = total / days
    X["Quiz.Share"] = X["Quiz.Actions"] / total.clip(lower=1)
    return X


def predict(df: pd.DataFrame):
    X = add_features(df[INPUTS])[columns]
    preds = model.predict(X)
    low_prob = model.predict_proba(X)[:, list(model.classes_).index("Low")]
    return preds, low_prob


def risk_level(p_low: float) -> str:
    if p_low >= 0.75:
        return "High risk"
    if p_low >= 0.5:
        return "Moderate risk"
    return "On track"


# ---------------------------------------------------------------- result pop-ups
@st.dialog("Prediction result", width="large")
def show_student_result(student: pd.DataFrame):
    pred, p_low = predict(student)
    pred, p_low = pred[0], float(p_low[0])
    confidence = p_low if pred == "Low" else 1 - p_low

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted rating", pred)
    r2.metric("Model confidence", f"{confidence:.0%}")
    r3.metric("Support status", risk_level(p_low))

    if pred == "Low":
        st.warning(
            "**This student may need additional academic support.** Consider reaching out, "
            "checking on missing submissions and quizzes, and encouraging regular LMS logins."
        )
    else:
        st.success("**This student shows engagement patterns typical of High-rated students.**")


@st.dialog("Class prediction results", width="large")
def show_class_result(df: pd.DataFrame):
    preds, p_low = predict(df)
    out = df.drop(columns="rating", errors="ignore").copy()
    out["Predicted.Rating"] = preds
    out["Low.Probability"] = p_low.round(3)
    out["Support.Status"] = [risk_level(p) for p in p_low]

    n_low = int((preds == "Low").sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Students", len(out))
    m2.metric("Predicted High", len(out) - n_low)
    m3.metric("Predicted Low (need support)", n_low)

    # If the file has true labels (e.g. the original dataset), show accuracy too
    if "rating" in df.columns:
        st.info(f"File contains actual ratings. Accuracy on this file: **{accuracy_score(df['rating'], preds):.1%}**")

    flagged_tab, all_tab = st.tabs([f"Need support ({n_low})", f"All students ({len(out)})"])
    with flagged_tab:
        flagged = out[out["Predicted.Rating"] == "Low"].sort_values("Low.Probability", ascending=False)
        if flagged.empty:
            st.success("No students flagged.")
        else:
            st.dataframe(flagged, hide_index=True, width="stretch")
    with all_tab:
        st.dataframe(out, hide_index=True, width="stretch")

    st.download_button(
        "Download predictions",
        out.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv",
        type="primary",
        on_click="ignore",  # keep the pop-up open after downloading
    )


# ---------------------------------------------------------------- app
if not MODEL_PATH.exists():
    st.error(f"`{MODEL_PATH}` not found. Run all cells in `main.ipynb` to train and save the model first.")
    st.stop()

model, columns = load_model()
data = load_dataset()

st.title("LMS Student Engagement Monitor")
st.write(
    "Predict a student's rating category from their LMS activity and identify "
    "students who may need additional academic support."
)

tab_predict, tab_perf = st.tabs(["Predict", "Model performance"])

# ---------------------------------------------------------------- tab 1: predict
with tab_predict:
    mode = st.segmented_control(
        "Input method",
        ["One student", "Upload class CSV"],
        default="One student",
        required=True,
    )

    if mode == "One student":
        with st.form("single"):
            st.subheader("Student activity logs (midterm)")
            values = {}
            for col, (group, names) in zip(st.columns(3), GROUPS.items()):
                with col:
                    st.markdown(f"**{group}**")
                    for name in names:
                        values[name] = st.number_input(
                            FEATURES[name], min_value=0, value=0, step=1, help=f"Column: {name}"
                        )
            if st.form_submit_button("Predict rating", type="primary", width="stretch"):
                show_student_result(pd.DataFrame([values]))

    else:
        st.subheader("Upload a class list")
        st.caption("Required columns: " + ", ".join(INPUTS) + ". Extra columns like student ID or name are kept.")
        file = st.file_uploader("Class CSV file", type="csv")
        if file is not None:
            df = pd.read_csv(file)
            missing = [c for c in INPUTS if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                st.write(f"Loaded **{len(df)}** students.")
                if st.button("Predict ratings", type="primary", width="stretch"):
                    show_class_result(df)

# ---------------------------------------------------------------- tab 2: performance
with tab_perf:
    if data is None:
        st.info(f"Put `{DATA_PATH}` next to app.py to show test-set performance.")
    else:
        # Re-create the exact test split used in main.ipynb
        X_all = add_features(data)
        _, X_test, _, y_test = train_test_split(
            X_all[columns], data["rating"], test_size=0.2, stratify=data["rating"], random_state=42
        )
        y_pred = model.predict(X_test)

        st.subheader("Evaluation on the held-out test set (20%)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
        k2.metric("Precision (Low)", f"{precision_score(y_test, y_pred, pos_label='Low'):.1%}")
        k3.metric("Recall (Low)", f"{recall_score(y_test, y_pred, pos_label='Low'):.1%}")
        k4.metric("F1 (macro)", f"{f1_score(y_test, y_pred, average='macro'):.1%}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            labels = list(model.classes_)
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            cm_df = pd.DataFrame(
                [(a, p, int(cm[i, j])) for i, a in enumerate(labels) for j, p in enumerate(labels)],
                columns=["Actual", "Predicted", "Count"],
            )
            base = alt.Chart(cm_df).encode(
                x=alt.X("Predicted:N", axis=alt.Axis(labelAngle=0)), y="Actual:N"
            )
            heat = base.mark_rect().encode(
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None)
            )
            text = base.mark_text(fontSize=22, fontWeight="bold").encode(
                text="Count:Q",
                color=alt.condition(alt.datum.Count > cm.max() / 2, alt.value("white"), alt.value("black")),
            )
            st.altair_chart((heat + text).properties(height=320), width="stretch")

        with right:
            st.markdown("**Feature importance**")
            imp = pd.DataFrame({"Feature": columns, "Importance": model.feature_importances_})
            st.altair_chart(
                alt.Chart(imp)
                .mark_bar()
                .encode(x="Importance:Q", y=alt.Y("Feature:N", sort="-x", title=None))
                .properties(height=320),
                width="stretch",
            )
        st.caption(
            f"Model: Random Forest, trained on {len(data) - len(X_test)} records, "
            f"tested on {len(X_test)}."
        )

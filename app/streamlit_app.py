import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_pipeline.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.json"
EDA_DIR = BASE_DIR / "outputs" / "eda"
EDA_FIGURES_DIR = EDA_DIR / "figures"
SUPERVISED_DIR = BASE_DIR / "outputs" / "supervised"
UNSUPERVISED_DIR = BASE_DIR / "outputs" / "unsupervised"

RAW_USABLE_ROWS = 181
FINAL_ML_ROWS = 118
TRIAL_ROWS = 92
NO_TRIAL_ROWS = 26
DEFAULT_LIKERT = 3

AGE_LABELS = [
    "< 18 ปี",
    "18-22 ปี",
    "23-29 ปี",
    "30-39 ปี",
    "40-49 ปี",
    "50-59 ปี",
    "60 ปีขึ้นไป",
]

INCOME_LABELS = [
    "< 10,000 บาท",
    "10,000-14,999 บาท",
    "15,000-19,999 บาท",
    "20,000-29,999 บาท",
    "30,000-39,999 บาท",
    "40,000-49,999 บาท",
    "50,000-59,999 บาท",
    "60,000 บาทขึ้นไป",
]

COFFEE_INPUTS = [
    {
        "key": "aroma_score",
        "label": "กลิ่นหอมกาแฟ",
        "keywords": ["กลิ่น"],
    },
    {
        "key": "value_score",
        "label": "ความคุ้มค่ากว่ากาแฟสด",
        "keywords": ["ประหยัด", "คุ้ม"],
    },
    {
        "key": "convenience_score",
        "label": "ความสะดวก / พกพาง่าย",
        "keywords": ["สะดวก", "พกพา"],
    },
    {
        "key": "caffeine_score",
        "label": "คาเฟอีน / ความตื่นตัว",
        "keywords": ["คาเฟอีน", "ตื่นตัว", "ดีด"],
    },
    {
        "key": "brand_score",
        "label": "แบรนด์น่าเชื่อถือ",
        "keywords": ["แบรนด์"],
    },
    {
        "key": "packaging_score",
        "label": "แพ็กเกจ / ขวดสวย",
        "keywords": ["แพ็คเกจ", "แพ็กเกจ", "ขวด"],
    },
]

SOCIAL_INPUTS = [
    {
        "key": "family_score",
        "label": "คนใกล้ตัว",
        "keywords": ["เพื่อน", "ครอบครัว", "คนใกล้ตัว"],
    },
    {
        "key": "online_review_score",
        "label": "รีวิวออนไลน์",
        "keywords": ["บล็อกเกอร์สายอาหารและเครื่องดื่ม"],
    },
]

CLUSTER_NAMES = {
    0: "กลุ่มเปิดรับกาแฟ RTD ปานกลาง",
    1: "กลุ่มต้องกระตุ้นด้วยข้อเสนอ",
    2: "กลุ่มชอบประสบการณ์กาแฟและภาพลักษณ์สินค้า",
    3: "กลุ่มเปิดรับต่อเนื่อง",
}

CLUSTER_NOTES = {
    0: "มีแนวโน้มเปิดรับ แต่ยังไม่เด่นเท่า Cluster 2",
    1: "ต้องใช้ข้อเสนอหรือประสบการณ์สินค้าเพื่อกระตุ้นเพิ่ม",
    2: "ให้ความสำคัญกับกลิ่น คนใกล้ตัว และแพ็กเกจ",
    3: "เปิดรับค่อนข้างสูง เหมาะกับการสื่อสารต่อเนื่อง",
}

CLUSTER_COUNTS_FALLBACK = {
    0: 54,
    1: 22,
    2: 12,
    3: 30,
}

CLUSTER_RATES_FALLBACK = {
    0: 83.3,
    1: 59.1,
    2: 91.7,
    3: 76.7,
}

PROFILE_FEATURES = [
    ("กลิ่นหอมกาแฟ", "likert_กลิ่นหอมกาแฟ"),
    ("ความคุ้มค่ากว่ากาแฟสด", "likert_ประหยัดกว่ากินกาแฟสดตามร้าน"),
    ("ความสะดวก / พกพาง่าย", "likert_ความสะดวก_พกพาง่าย_ไม่เลอะเทอะ"),
    ("คาเฟอีน / ความตื่นตัว", "likert_ต้องการความดีด_ตื่นตัวจากคาเฟอีน"),
    ("แบรนด์น่าเชื่อถือ", "likert_แบรนด์ดัง_น่าเชื่อถือ"),
    ("แพ็กเกจ / ขวดสวย", "likert_แพ็คเกจ_ขวดสวย"),
    ("คนใกล้ตัว", "likert_เพื่อน_ครอบครัว_และคนใกล้ตัว"),
]


st.set_page_config(
    page_title="Coffee RTD Consumer Insight Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    .metric-card, .insight-card, .workflow-box, .note-box {
        border: 1px solid #30363d;
        border-radius: 8px;
        background: #161b22;
        color: #f0f6fc;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        min-height: 110px;
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.92rem;
        margin-bottom: 0.7rem;
    }
    .metric-value {
        color: #f0f6fc;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .insight-card {
        padding: 1.1rem;
        min-height: 145px;
    }
    .insight-title {
        color: #f0f6fc;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .insight-body {
        color: #c9d1d9;
        font-size: 0.98rem;
        line-height: 1.6;
    }
    .workflow-box {
        padding: 1rem 1.1rem;
        margin-top: 0.5rem;
        font-weight: 650;
        color: #f0f6fc;
    }
    .note-box {
        padding: 0.85rem 1rem;
        color: #c9d1d9;
        border-left: 4px solid #58a6ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="กำลังโหลดข้อมูล...")
def load_assets():
    if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        return None, None, "ยังไม่พบไฟล์โมเดล กรุณารันสคริปต์ Supervised อีกครั้ง"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(MODEL_PATH, "rb") as model_file:
                model = pickle.load(model_file)

        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as feature_file:
            feature_columns = json.load(feature_file)

        return model, feature_columns, None
    except Exception:
        return None, None, "โหลดโมเดลไม่สำเร็จ กรุณาตรวจสอบไฟล์โมเดลอีกครั้ง"


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title, body):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note_box(text):
    st.markdown(
        f"""<div class="note-box">{text}</div>""",
        unsafe_allow_html=True,
    )


def workflow_box(text):
    st.markdown(
        f"""<div class="workflow-box">{text}</div>""",
        unsafe_allow_html=True,
    )


def show_image_if_exists(path, caption):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
        return True
    return False


def load_rank_table(path, item_label, top_n=5):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    first_col = df.columns[0]
    display_df = df.head(top_n).copy()
    display_df = display_df.rename(
        columns={
            first_col: item_label,
            "Trial_Group": "กลุ่ม Trial",
            "No_Trial_Group": "กลุ่ม No Trial",
        }
    )
    return display_df


def load_model_comparison():
    path = SUPERVISED_DIR / "model_comparison.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    wanted_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    existing_cols = [col for col in wanted_cols if col in df.columns]
    if not existing_cols:
        return None

    display_df = df[existing_cols].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda value: f"{value:.3f}")
    return display_df


def feature_categories(feature_columns, prefix):
    return [col.replace(prefix, "", 1) for col in feature_columns if col.startswith(prefix)]


def default_index(options, preferred):
    return options.index(preferred) if preferred in options else 0


def find_likert_features(feature_columns, keywords):
    matches = []
    for col in feature_columns:
        if col.startswith("likert_") and any(keyword in col for keyword in keywords):
            matches.append(col)
    return matches


def build_feature_mappings(feature_columns):
    mappings = {}
    for item in COFFEE_INPUTS + SOCIAL_INPUTS:
        mappings[item["key"]] = find_likert_features(feature_columns, item["keywords"])
    return mappings


def set_one_hot(input_df, feature_columns, prefix, selected_value):
    col = f"{prefix}{selected_value}"
    if col in feature_columns:
        input_df.at[0, col] = 1


def build_persona_input(feature_columns, values, mappings):
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    if "age_ordinal" in feature_columns:
        input_df.at[0, "age_ordinal"] = values["age"]
    if "income_ordinal" in feature_columns:
        input_df.at[0, "income_ordinal"] = values["income"]
    if "is_drinker_binary" in feature_columns:
        input_df.at[0, "is_drinker_binary"] = 1

    set_one_hot(input_df, feature_columns, "gender_", values["gender"])
    set_one_hot(input_df, feature_columns, "job_", values["job"])
    set_one_hot(input_df, feature_columns, "edu_", values["education"])

    for col in feature_columns:
        if col.startswith("likert_"):
            input_df.at[0, col] = DEFAULT_LIKERT

    for item in COFFEE_INPUTS + SOCIAL_INPUTS:
        for feature in mappings.get(item["key"], []):
            input_df.at[0, feature] = values[item["key"]]

    return input_df


def trial_probability(model, input_df):
    probabilities = model.predict_proba(input_df)[0]
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return float(probabilities[classes.index(1)])
    return float(probabilities[-1])


def tendency_level(probability):
    if probability < 0.5:
        return "ต่ำ"
    if probability <= 0.7:
        return "ปานกลาง"
    return "สูง"


def tendency_text(probability):
    if probability < 0.5:
        return "Persona นี้ยังไม่ใช่กลุ่มที่เด่นที่สุดสำหรับการทดลองสินค้า"
    if probability <= 0.7:
        return "Persona นี้มีแนวโน้มระดับปานกลาง ควรใช้ร่วมกับข้อมูล segment เพื่อประกอบการตัดสินใจ"
    return "Persona นี้มีแนวโน้มเปิดรับสินค้า เหมาะสำหรับใช้เป็นกลุ่มทดลองแคมเปญเบื้องต้น"


def persona_preset_defaults(preset):
    presets = {
        "กลุ่มทั่วไป": {
            "age": 2,
            "income": 3,
            "coffee": 3,
            "social": 3,
        },
        "กลุ่มสนใจสินค้า RTD สูง": {
            "age": 2,
            "income": 3,
            "coffee": 4,
            "social": 4,
        },
        "กลุ่มต้องกระตุ้นเพิ่ม": {
            "age": 2,
            "income": 3,
            "coffee": 2,
            "social": 2,
        },
    }
    return presets.get(preset, presets["กลุ่มทั่วไป"])


def validate_persona(values):
    issues = []
    age = int(values["age"])
    income = int(values["income"])
    job = values["job"]
    education = values["education"]

    if age == 0 and job in {"แพทย์", "พนักงานบริษัทเอกชน", "ธุรกิจส่วนตัว"}:
        issues.append("อายุต่ำกว่า 18 ปีไม่สอดคล้องกับอาชีพที่เลือก")

    if age == 0 and income == 7:
        issues.append("อายุต่ำกว่า 18 ปีไม่ควรเลือกรายได้ 60,000 บาทขึ้นไป")

    if job == "แพทย์" and education not in {"ปริญญาตรี", "ปริญญาโท"}:
        issues.append("อาชีพแพทย์ควรมีระดับการศึกษาปริญญาตรีขึ้นไป")

    if job in {"นักเรียน", "นักศึกษา"} and income >= 6:
        issues.append("นักเรียนหรือนักศึกษาไม่ควรเลือกรายได้ระดับสูงมาก")

    return issues


def load_cluster_profile_table():
    for profile_path in UNSUPERVISED_DIR.glob("cluster_profile_*.csv"):
        profile_df = pd.read_csv(profile_path)
        if "cluster" in profile_df.columns:
            return profile_df
    return None


def load_cluster_summary_table():
    summary_path = UNSUPERVISED_DIR / "cluster_summary.csv"
    if not summary_path.exists():
        return None

    summary_df = pd.read_csv(summary_path)
    required_cols = {"cluster", "count", "try_rate"}
    if not required_cols.issubset(set(summary_df.columns)):
        return None
    return summary_df


def load_cluster_assignment_table():
    assignment_path = UNSUPERVISED_DIR / "cluster_assignments.csv"
    if not assignment_path.exists():
        return None

    assignment_df = pd.read_csv(assignment_path)
    if "cluster" not in assignment_df.columns:
        return None
    return assignment_df


def load_cluster_rates():
    candidate_files = [
        UNSUPERVISED_DIR / "cluster_trial_rate.csv",
        UNSUPERVISED_DIR / "cluster_try_rate.csv",
    ]
    summary_df = load_cluster_summary_table()
    if summary_df is not None:
        return summary_df[["cluster", "try_rate"]].copy()

    candidate_files.extend(UNSUPERVISED_DIR.glob("cluster_profile_*.csv"))

    for path in candidate_files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "cluster" in df.columns and "target_trial" in df.columns:
            rate_df = df[["cluster", "target_trial"]].copy()
            rate_df["try_rate"] = rate_df["target_trial"] * 100
            return rate_df[["cluster", "try_rate"]]

        lower_cols = {col.lower(): col for col in df.columns}
        cluster_col = lower_cols.get("cluster")
        rate_col = lower_cols.get("try rate") or lower_cols.get("try_rate") or lower_cols.get("trial_rate")
        if cluster_col and rate_col:
            rate_df = df[[cluster_col, rate_col]].copy()
            rate_df.columns = ["cluster", "try_rate"]
            if rate_df["try_rate"].max() <= 1:
                rate_df["try_rate"] = rate_df["try_rate"] * 100
            return rate_df

    return None


def cluster_count_lookup():
    count_lookup = CLUSTER_COUNTS_FALLBACK.copy()
    summary_df = load_cluster_summary_table()
    if summary_df is not None:
        for _, row in summary_df.iterrows():
            count_lookup[int(row["cluster"])] = int(row["count"])
    return count_lookup


def rate_lookup_from_df(rate_df):
    rate_lookup = CLUSTER_RATES_FALLBACK.copy()
    if rate_df is not None:
        for _, row in rate_df.iterrows():
            rate_lookup[int(row["cluster"])] = float(row["try_rate"])
    return rate_lookup


def cluster_profile_row(cluster_id, profile_df=None):
    if profile_df is None:
        profile_df = load_cluster_profile_table()
    if profile_df is None:
        return None

    cluster_rows = profile_df[profile_df["cluster"].astype(int) == int(cluster_id)]
    if cluster_rows.empty:
        return None
    return cluster_rows.iloc[0]


def cluster_feature_profile(cluster_id, profile_df=None):
    row = cluster_profile_row(cluster_id, profile_df)
    if row is None:
        return pd.DataFrame(columns=["ปัจจัย", "ค่าเฉลี่ยของ Cluster"])

    rows = []
    for label, feature in PROFILE_FEATURES:
        if feature in row.index:
            rows.append(
                {
                    "ปัจจัย": label,
                    "ค่าเฉลี่ยของ Cluster": float(row[feature]),
                }
            )

    return pd.DataFrame(rows).sort_values("ค่าเฉลี่ยของ Cluster", ascending=False)


def top_cluster_features(cluster_id, profile_df=None, top_n=3):
    profile = cluster_feature_profile(cluster_id, profile_df)
    if profile.empty:
        if int(cluster_id) == 2:
            return [
                ("กลิ่นหอมกาแฟ", 4.33),
                ("คนใกล้ตัว", 4.25),
                ("แพ็กเกจ / ขวดสวย", 4.17),
            ]
        return []
    return [
        (row["ปัจจัย"], float(row["ค่าเฉลี่ยของ Cluster"]))
        for _, row in profile.head(top_n).iterrows()
    ]


def build_cluster_model_input(feature_columns, cluster_id, profile_df=None):
    row = cluster_profile_row(cluster_id, profile_df)
    if row is None:
        return None

    input_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    for col in feature_columns:
        if col in row.index:
            input_df.at[0, col] = float(row[col])
    return input_df


def cluster_model_tendency(model, feature_columns, cluster_id, profile_df=None):
    assignment_df = load_cluster_assignment_table()
    if assignment_df is not None:
        cluster_rows = assignment_df[assignment_df["cluster"].astype(int) == int(cluster_id)]
        if not cluster_rows.empty:
            input_df = pd.DataFrame(0.0, index=range(len(cluster_rows)), columns=feature_columns)
            for col in feature_columns:
                if col in cluster_rows.columns:
                    input_df[col] = cluster_rows[col].astype(float).values
            probabilities = model.predict_proba(input_df)
            classes = list(getattr(model, "classes_", []))
            class_index = classes.index(1) if 1 in classes else -1
            return float(probabilities[:, class_index].mean()), "ค่าเฉลี่ยผลโมเดลของสมาชิกใน Cluster นี้"

    input_df = build_cluster_model_input(feature_columns, cluster_id, profile_df)
    if input_df is None:
        return None, None
    return trial_probability(model, input_df), "ผลโมเดลจากโปรไฟล์เฉลี่ยของ Cluster นี้"


def format_cluster_feature_name(feature_name):
    clean_name = feature_name.replace("likert_", "")
    clean_name = clean_name.replace("_", " ")
    replacements = {
        "กลิ่นหอมกาแฟ": "กลิ่นหอมกาแฟ",
        "เพื่อน ครอบครัว และคนใกล้ตัว": "คนใกล้ตัว",
        "แพ็คเกจ ขวดสวย": "แพ็กเกจ / ขวดสวย",
    }
    return replacements.get(clean_name, clean_name)


def load_cluster_2_profile():
    profile_df = None
    for profile_path in UNSUPERVISED_DIR.glob("cluster_profile_*.csv"):
        candidate_df = pd.read_csv(profile_path)
        if "cluster" in candidate_df.columns:
            profile_df = candidate_df
            break

    if profile_df is None:
        return [
            ("กลิ่นหอมกาแฟ", 4.33),
            ("คนใกล้ตัว", 4.25),
            ("แพ็กเกจ / ขวดสวย", 4.17),
        ]

    cluster_rows = profile_df[profile_df["cluster"].astype(int) == 2]
    if cluster_rows.empty:
        return [
            ("กลิ่นหอมกาแฟ", 4.33),
            ("คนใกล้ตัว", 4.25),
            ("แพ็กเกจ / ขวดสวย", 4.17),
        ]

    cluster_2 = cluster_rows.iloc[0]
    preferred_features = [
        "likert_กลิ่นหอมกาแฟ",
        "likert_เพื่อน_ครอบครัว_และคนใกล้ตัว",
        "likert_แพ็คเกจ_ขวดสวย",
    ]
    profile = []
    for feature in preferred_features:
        if feature in cluster_2.index:
            profile.append((format_cluster_feature_name(feature), float(cluster_2[feature])))

    return profile or [
        ("กลิ่นหอมกาแฟ", 4.33),
        ("คนใกล้ตัว", 4.25),
        ("แพ็กเกจ / ขวดสวย", 4.17),
    ]


def cluster_2_rate(rate_df):
    if rate_df is None:
        return 91.7
    clusters = rate_df["cluster"].astype(int)
    if 2 not in clusters.tolist():
        return 91.7
    return float(rate_df.loc[clusters == 2, "try_rate"].iloc[0])


def build_cluster_summary(rate_df):
    rate_lookup = rate_lookup_from_df(rate_df)
    count_lookup = cluster_count_lookup()

    rows = []
    for cluster_id in sorted(rate_lookup):
        rows.append(
            {
                "Cluster": cluster_id,
                "จำนวนคน": count_lookup.get(cluster_id, "-"),
                "Try Rate": f"{rate_lookup[cluster_id]:.1f}%",
                "Segment Interpretation": CLUSTER_NAMES.get(cluster_id, "กลุ่มลูกค้า"),
                "Note": CLUSTER_NOTES.get(cluster_id, ""),
            }
        )
    return pd.DataFrame(rows)


def build_cluster_chart_data(rate_df):
    summary_df = build_cluster_summary(rate_df)
    chart_df = summary_df[["Cluster", "Try Rate"]].copy()
    chart_df["Cluster"] = chart_df["Cluster"].map(lambda value: f"Cluster {value}")
    chart_df["Try Rate"] = chart_df["Try Rate"].str.replace("%", "", regex=False).astype(float)
    return chart_df.set_index("Cluster")


model, feature_columns, load_error = load_assets()

st.title("Coffee RTD Consumer Insight Dashboard")
st.write(
    "แดชบอร์ดนี้สรุปผลการวิเคราะห์แบบสอบถามผู้บริโภคกาแฟ RTD "
    "เพื่อดูพฤติกรรมการลองสินค้า กลุ่มลูกค้าที่น่าสนใจ และแนวทางสื่อสารการตลาดเบื้องต้น"
)

tab_overview, tab_persona, tab_segment, tab_recommendation = st.tabs(
    [
        "Overview",
        "Segment Profile Explorer",
        "Segment Match",
        "Business Recommendation",
    ]
)

with tab_overview:
    st.header("Overview")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Survey Data", f"{RAW_USABLE_ROWS} rows")
    with metric_cols[1]:
        metric_card("ML Dataset", f"{FINAL_ML_ROWS} rows")
    with metric_cols[2]:
        metric_card("Trial", f"{TRIAL_ROWS} คน")
    with metric_cols[3]:
        metric_card("No Trial", f"{NO_TRIAL_ROWS} คน")

    st.subheader("Data Flow Summary")
    workflow_box(
        f"Survey Data {RAW_USABLE_ROWS} rows → ML Dataset {FINAL_ML_ROWS} rows → "
        f"Trial {TRIAL_ROWS} คน / No Trial {NO_TRIAL_ROWS} คน"
    )

    st.subheader("Trial / No Trial Distribution")
    distribution_df = pd.DataFrame(
        {
            "กลุ่ม": ["Trial", "No Trial"],
            "จำนวน": [TRIAL_ROWS, NO_TRIAL_ROWS],
        }
    )
    st.bar_chart(distribution_df.set_index("กลุ่ม"))
    st.caption("จำนวนผู้ตอบที่ใช้ในชุดข้อมูลสำหรับโมเดล")

    st.subheader("Consumer Insight จากข้อมูลแบบสอบถาม")
    insight_cols = st.columns(2)
    with insight_cols[0]:
        shown = show_image_if_exists(
            EDA_FIGURES_DIR / "income_vs_trial.png",
            "Try Rate ตามช่วงรายได้",
        )
        if not shown:
            insight_card("รายได้ที่น่าสนใจ", "กลุ่มรายได้ 20,000-29,999 บาทมี Try Rate เด่นในข้อมูลชุดนี้")
    with insight_cols[1]:
        shown = show_image_if_exists(
            EDA_FIGURES_DIR / "reason_to_try.png",
            "เหตุผลที่กระตุ้นให้ทดลองสินค้า",
        )
        if not shown:
            insight_card("เหตุผลการทดลองสินค้า", "การแจกชิมเป็นเหตุผลสำคัญในการกระตุ้นให้ลองสินค้า")

    shown = show_image_if_exists(
        EDA_FIGURES_DIR / "macro_drivers_comparison.png",
        "ปัจจัยที่ต่างกันระหว่าง Trial และ No Trial",
    )
    if not shown:
        insight_card("ปัจจัยสินค้า", "ความคุ้มค่ากว่ากาแฟสดเป็นจุดต่างที่ควรใช้สื่อสาร")

    st.subheader("คำตอบเบื้องต้นจากข้อมูล")
    answer_df = pd.DataFrame(
        [
            {
                "คำถาม": "กลุ่มไหนน่าสนใจ?",
                "คำตอบเบื้องต้น": "ผู้ดื่มกาแฟ / รายได้ 20,000-29,999 / Cluster 2",
            },
            {
                "คำถาม": "อะไรกระตุ้นให้ลอง?",
                "คำตอบเบื้องต้น": "การแจกชิม + ความคุ้มค่ากว่ากาแฟสด",
            },
            {
                "คำถาม": "ควรสื่อสารอย่างไร?",
                "คำตอบเบื้องต้น": "สื่อสังคมออนไลน์ + ร้านสะดวกซื้อเป็นจุดขาย",
            },
        ]
    )
    st.dataframe(answer_df, hide_index=True, use_container_width=True)

    channel_col, buy_col = st.columns(2)
    with channel_col:
        st.markdown("#### ช่องทางรับรู้สินค้า")
        insight_card(
            "สื่อออนไลน์หลายช่องทาง",
            "กลุ่ม Trial พบการเปิดรับสื่อออนไลน์หลายช่องทาง โดยสื่อสังคมออนไลน์เหมาะกับการสร้างการรับรู้ก่อนซื้อ",
        )
    with buy_col:
        st.markdown("#### ช่องทางซื้อสินค้า")
        buy_df = load_rank_table(EDA_DIR / "buy_channel_by_try_not_try.csv", "ช่องทางซื้อ")
        if buy_df is not None:
            st.dataframe(buy_df, hide_index=True, use_container_width=True)
        else:
            insight_card("ช่องทางซื้อ", "ร้านสะดวกซื้อเหมาะเป็นจุดกระตุ้นการซื้อ")

    st.subheader("Project Workflow")
    workflow_box(
        "Survey Data → Data Cleaning → Supervised Learning → Customer Segmentation → Business Insight"
    )

    st.caption("หมายเหตุ: ผลลัพธ์นี้อ้างอิงจากข้อมูลแบบสอบถามในโปรเจค")

with tab_persona:
    st.header("Segment Profile Explorer")
    st.write(
        "เลือกกลุ่มลูกค้าจากผลการแบ่งกลุ่ม เพื่อดูโปรไฟล์จริงของแต่ละกลุ่มและแนวโน้มการเปิดรับกาแฟ RTD"
    )

    rate_df = load_cluster_rates()
    profile_df = load_cluster_profile_table()
    rate_lookup = rate_lookup_from_df(rate_df)
    count_lookup = cluster_count_lookup()

    if profile_df is None and rate_df is None:
        st.warning("ยังไม่พบผลลัพธ์บางส่วนของการแบ่งกลุ่ม กรุณารันสคริปต์ Unsupervised อีกครั้ง")
    else:
        selected_cluster = st.selectbox(
            "เลือก Cluster",
            options=sorted(rate_lookup),
            format_func=lambda cluster_id: f"Cluster {cluster_id} - {CLUSTER_NAMES.get(cluster_id, 'กลุ่มลูกค้า')}",
            index=sorted(rate_lookup).index(2) if 2 in rate_lookup else 0,
        )
        selected_rate = rate_lookup.get(selected_cluster, CLUSTER_RATES_FALLBACK.get(selected_cluster, 0))
        selected_count = count_lookup.get(selected_cluster, "-")

        st.subheader(f"Cluster {selected_cluster}: {CLUSTER_NAMES.get(selected_cluster, 'กลุ่มลูกค้า')}")
        cluster_cols = st.columns(3)
        with cluster_cols[0]:
            metric_card("จำนวนคนในกลุ่ม", f"{selected_count} คน")
        with cluster_cols[1]:
            metric_card("Try Rate", f"{selected_rate:.1f}%")
        with cluster_cols[2]:
            metric_card("ลักษณะกลุ่ม", CLUSTER_NAMES.get(selected_cluster, "กลุ่มลูกค้า"))

        note_box(CLUSTER_NOTES.get(selected_cluster, "สรุปจากโปรไฟล์ของกลุ่มที่เลือก"))

        st.subheader("ปัจจัยเด่นของกลุ่มที่เลือก")
        top_features = top_cluster_features(selected_cluster, profile_df)
        if top_features:
            feature_cols = st.columns(len(top_features))
            for col, (label, value) in zip(feature_cols, top_features):
                with col:
                    metric_card(label, f"{value:.2f} / 5")
        else:
            st.info("ยังไม่มีข้อมูลค่าเฉลี่ยของปัจจัยสำหรับกลุ่มนี้")

        st.subheader("ตารางโปรไฟล์จากข้อมูลจริงของกลุ่ม")
        cluster_profile_df = cluster_feature_profile(selected_cluster, profile_df)
        if not cluster_profile_df.empty:
            display_profile = cluster_profile_df.copy()
            display_profile["ค่าเฉลี่ยของ Cluster"] = display_profile["ค่าเฉลี่ยของ Cluster"].map(
                lambda value: f"{value:.2f} / 5"
            )
            st.dataframe(display_profile, hide_index=True, use_container_width=True)
        else:
            st.info("ยังไม่มีตารางโปรไฟล์ของกลุ่มนี้")

        if load_error:
            st.warning(load_error)
        elif model is not None and feature_columns is not None:
            probability, tendency_source = cluster_model_tendency(
                model,
                feature_columns,
                selected_cluster,
                profile_df,
            )
            if probability is not None:
                st.subheader("Model Trial Tendency ของ Cluster นี้")
                model_cols = st.columns(3)
                with model_cols[0]:
                    metric_card("Trial Tendency", f"{probability * 100:.1f}%")
                with model_cols[1]:
                    metric_card("No Trial Tendency", f"{(1 - probability) * 100:.1f}%")
                with model_cols[2]:
                    metric_card("ระดับแนวโน้ม", tendency_level(probability))
                st.caption(f"{tendency_source} ไม่ใช่การทำนายลูกค้ารายบุคคล")

        st.subheader("Supervised Model Result")
        st.write(
            "โมเดลส่วนนี้ใช้ประเมินแนวโน้ม Trial / No Trial จากข้อมูลแบบสอบถาม "
            "โดยเลือกโมเดลจากผลเปรียบเทียบ F1-score"
        )

        model_metrics = load_model_comparison()
        if model_metrics is not None:
            st.dataframe(model_metrics, hide_index=True, use_container_width=True)

        supervised_cols = st.columns(2)
        with supervised_cols[0]:
            show_image_if_exists(
                SUPERVISED_DIR / "model_comparison_chart.png",
                "ผลเปรียบเทียบโมเดล Supervised",
            )
        with supervised_cols[1]:
            show_image_if_exists(
                SUPERVISED_DIR / "confusion_matrix_heatmap.png",
                "Confusion matrix ของโมเดลที่เลือก",
            )
            st.caption(
                "โมเดลจับกลุ่ม Trial ได้ดี แต่ยังมีข้อจำกัดในการแยกกลุ่ม No Trial เนื่องจากข้อมูลไม่สมดุล"
            )

        with st.expander("ทดลองปรับค่าเพิ่มเติม"):
            st.caption("ส่วนนี้ใช้ทดลองเปรียบเทียบเท่านั้น ไม่ใช่ข้อมูลจริงจากแบบสอบถาม")
            if load_error:
                st.warning(load_error)
            elif model is None or feature_columns is None:
                st.warning("ยังไม่พบข้อมูลสำหรับการวิเคราะห์ กรุณารันสคริปต์ที่เกี่ยวข้องอีกครั้ง")
            else:
                mappings = build_feature_mappings(feature_columns)
                gender_options = feature_categories(feature_columns, "gender_")
                job_options = feature_categories(feature_columns, "job_")
                education_options = feature_categories(feature_columns, "edu_")

                with st.form("manual_persona_form"):
                    demo_col, coffee_col, social_col = st.columns([1, 1.2, 1])

                    with demo_col:
                        st.subheader("ข้อมูล Persona")
                        age = st.selectbox(
                            "อายุ",
                            options=list(range(len(AGE_LABELS))),
                            format_func=lambda idx: AGE_LABELS[idx],
                            index=2,
                        )
                        income = st.selectbox(
                            "รายได้",
                            options=list(range(len(INCOME_LABELS))),
                            format_func=lambda idx: INCOME_LABELS[idx],
                            index=3,
                        )
                        gender = st.selectbox(
                            "เพศ",
                            options=gender_options,
                            index=default_index(gender_options, "ไม่ระบุ"),
                        )
                        job = st.selectbox(
                            "อาชีพ",
                            options=job_options,
                            index=default_index(job_options, "พนักงานบริษัทเอกชน"),
                        )
                        education = st.selectbox(
                            "ระดับการศึกษา",
                            options=education_options,
                            index=default_index(education_options, "ปริญญาตรี"),
                        )

                    with coffee_col:
                        st.subheader("ปัจจัยเกี่ยวกับกาแฟ RTD")
                        coffee_values = {}
                        for item in COFFEE_INPUTS:
                            coffee_values[item["key"]] = st.slider(
                                item["label"],
                                min_value=1,
                                max_value=5,
                                value=3,
                                step=1,
                            )

                    with social_col:
                        st.subheader("อิทธิพลต่อการตัดสินใจ")
                        social_values = {}
                        for item in SOCIAL_INPUTS:
                            social_values[item["key"]] = st.slider(
                                item["label"],
                                min_value=1,
                                max_value=5,
                                value=3,
                                step=1,
                            )

                    submitted = st.form_submit_button("ทดลองคำนวณแนวโน้ม")

                if submitted:
                    persona_values = {
                        "age": age,
                        "income": income,
                        "gender": gender,
                        "job": job,
                        "education": education,
                        **coffee_values,
                        **social_values,
                    }
                    issues = validate_persona(persona_values)
                    if issues:
                        st.warning(
                            "ข้อมูล persona ชุดนี้ไม่สมเหตุสมผลพอสำหรับการจำลองแนวโน้ม "
                            "กรุณาปรับข้อมูลให้ใกล้เคียงกลุ่มผู้ตอบแบบสอบถาม"
                        )
                        for issue in issues:
                            st.caption(f"- {issue}")
                    else:
                        input_df = build_persona_input(feature_columns, persona_values, mappings)
                        probability = trial_probability(model, input_df)
                        result_cols = st.columns(3)
                        with result_cols[0]:
                            metric_card("Trial Tendency", f"{probability * 100:.1f}%")
                        with result_cols[1]:
                            metric_card("No Trial Tendency", f"{(1 - probability) * 100:.1f}%")
                        with result_cols[2]:
                            metric_card("ระดับแนวโน้ม", tendency_level(probability))
                        st.caption(
                            "ผลนี้เป็นการทดลองเปรียบเทียบจากชุดค่าที่ปรับเอง ไม่ใช่หลักฐานหลักของข้อเสนอแนะทางธุรกิจ"
                        )

        with st.expander("ตัวอย่างการแยกกลุ่มด้วย Decision Tree"):
            shown = show_image_if_exists(
                SUPERVISED_DIR / "decision_tree_explainable.png",
                "ใช้ Decision Tree เป็นภาพประกอบการตีความ ไม่ใช่โมเดลหลักที่ใช้ทำนาย",
            )
            if not shown:
                st.write("ยังไม่พบภาพประกอบส่วนนี้")

with tab_segment:
    st.header("Customer Segment")
    st.write("สรุปผลการแบ่งกลุ่มลูกค้าจากพฤติกรรมและทัศนคติต่อกาแฟ RTD")

    pca_path = UNSUPERVISED_DIR / "pca_clusters.png"
    heatmap_path = UNSUPERVISED_DIR / "cluster_profile_heatmap.png"
    rate_df = load_cluster_rates()
    has_segment_output = pca_path.exists() or heatmap_path.exists() or rate_df is not None

    if not has_segment_output:
        st.warning("ยังไม่พบผลลัพธ์บางส่วนของการแบ่งกลุ่ม กรุณารันสคริปต์ Unsupervised อีกครั้ง")
    else:
        segment_cols = st.columns(3)
        with segment_cols[0]:
            metric_card("จำนวนกลุ่ม", "4 กลุ่ม")
        with segment_cols[1]:
            metric_card("กลุ่มที่น่าสนใจที่สุด", "Cluster 2")
        with segment_cols[2]:
            metric_card("Try Rate ของ Cluster 2", f"{cluster_2_rate(rate_df):.1f}%")

        st.subheader("Try Rate by Cluster")
        cluster_chart_data = build_cluster_chart_data(rate_df)
        st.bar_chart(cluster_chart_data)
        st.caption("เปรียบเทียบอัตรา Trial ของแต่ละกลุ่ม")

        st.subheader("Cluster Summary")
        cluster_summary_df = build_cluster_summary(rate_df)
        st.dataframe(cluster_summary_df, hide_index=True, use_container_width=True)
        st.caption("ชื่อ segment เป็นชื่อเชิงตีความจากผลวิเคราะห์ ไม่ใช่ชื่อที่โมเดลสร้างเอง")

        st.subheader("ลักษณะเด่นของ Cluster 2")
        cluster_profile = load_cluster_2_profile()
        profile_cols = st.columns(len(cluster_profile))
        for col, (label, value) in zip(profile_cols, cluster_profile):
            with col:
                metric_card(label, f"{value:.2f} / 5")

        note_box(
            "Cluster 2 เป็นกลุ่มที่ให้ความสำคัญกับประสบการณ์ของสินค้า เช่น กลิ่นหอม "
            "ภาพลักษณ์ และอิทธิพลจากคนใกล้ตัว จึงเหมาะกับการทดลองแคมเปญที่เน้นให้ลองสินค้า "
            "และสร้างความมั่นใจผ่านประสบการณ์จริง"
        )
        note_box(
            "เมื่อเทียบกับ cluster อื่น Cluster 2 มี Try Rate สูงที่สุด "
            "จึงถูกเลือกเป็นกลุ่มเป้าหมายเบื้องต้นสำหรับแคมเปญทดลองสินค้า"
        )

        image_cols = st.columns(2)
        if pca_path.exists():
            image_cols[0].image(
                str(pca_path),
                caption="ภาพรวมการกระจายของกลุ่มลูกค้า",
                use_container_width=True,
            )
        else:
            image_cols[0].warning(
                "ยังไม่พบผลลัพธ์บางส่วนของการแบ่งกลุ่ม กรุณารันสคริปต์ Unsupervised อีกครั้ง"
            )

        if heatmap_path.exists():
            image_cols[1].image(
                str(heatmap_path),
                caption="คะแนนเฉลี่ยของปัจจัยสำคัญในแต่ละกลุ่ม",
                use_container_width=True,
            )
        else:
            image_cols[1].warning(
                "ยังไม่พบผลลัพธ์บางส่วนของการแบ่งกลุ่ม กรุณารันสคริปต์ Unsupervised อีกครั้ง"
            )

        with st.expander("รายละเอียดวิธีวิเคราะห์"):
            st.write(
                "การแบ่งกลุ่มใช้ K-Means จากตัวแปรด้านพฤติกรรมและทัศนคติ "
                "จากนั้นนำผล Trial / No Trial กลับมาประกอบการแปลผลของแต่ละกลุ่ม"
            )
            st.write(
                "เลือก K=4 เพื่อให้ตีความเชิงธุรกิจได้ละเอียดขึ้น แม้ค่าทางสถิติอาจแนะนำจำนวนกลุ่มอื่น"
            )
            method_cols = st.columns(2)
            with method_cols[0]:
                show_image_if_exists(
                    UNSUPERVISED_DIR / "elbow_method.png",
                    "การพิจารณาจำนวนกลุ่มด้วย Elbow Method",
                )
            with method_cols[1]:
                show_image_if_exists(
                    UNSUPERVISED_DIR / "silhouette_score.png",
                    "การตรวจสอบคุณภาพการแบ่งกลุ่มด้วย Silhouette Score",
                )

with tab_recommendation:
    st.header("Business Recommendation")
    st.write(
        "สรุปคำตอบจากข้อมูลว่า กลุ่มใดน่าสนใจ อะไรช่วยกระตุ้นการทดลองสินค้า และควรสื่อสารอย่างไร"
    )

    st.subheader("Evidence ที่ใช้ประกอบคำแนะนำ")
    evidence_df = pd.DataFrame(
        [
            {
                "หลักฐาน": "Cluster 2 Try Rate 91.7%",
                "ความหมาย": "กลุ่มนี้เปิดรับสินค้าใหม่สูง",
                "ข้อเสนอแนะ": "ใช้เป็นกลุ่มเป้าหมายเบื้องต้น",
            },
            {
                "หลักฐาน": "กลิ่นหอมกาแฟ 4.33 / 5",
                "ความหมาย": "ให้ความสำคัญกับประสบการณ์กาแฟ",
                "ข้อเสนอแนะ": "สื่อสารเรื่องกลิ่นและรสชาติใกล้กาแฟสด",
            },
            {
                "หลักฐาน": "คนใกล้ตัว 4.25 / 5",
                "ความหมาย": "การบอกต่อมีผลต่อการเปิดรับ",
                "ข้อเสนอแนะ": "ใช้รีวิวผู้ใช้จริงเพื่อเพิ่มความมั่นใจ",
            },
            {
                "หลักฐาน": "แพ็กเกจ / ขวดสวย 4.17 / 5",
                "ความหมาย": "ภาพลักษณ์สินค้ามีผล",
                "ข้อเสนอแนะ": "เน้นแพ็กเกจและหน้าชั้นวาง",
            },
            {
                "หลักฐาน": "การแจกชิม / ทดลองสินค้า",
                "ความหมาย": "ช่วยกระตุ้นการทดลองสินค้า",
                "ข้อเสนอแนะ": "ทำแคมเปญแจกชิมในจุดที่เข้าถึงกลุ่มเป้าหมาย",
            },
        ]
    )
    st.dataframe(evidence_df, hide_index=True, use_container_width=True)

    rec_cols = st.columns(3)
    with rec_cols[0]:
        insight_card(
            "กลุ่มเป้าหมาย",
            "เริ่มจากกลุ่มผู้ดื่มกาแฟที่มีแนวโน้มเปิดรับสินค้าใหม่ "
            "โดยเฉพาะกลุ่มรายได้ 20,000-29,999 บาท และ Cluster 2",
        )
    with rec_cols[1]:
        insight_card(
            "ปัจจัยกระตุ้นการลองสินค้า",
            "ควรเน้นการให้ทดลองสินค้า และสื่อสารเรื่องความคุ้มค่ากว่ากาแฟสด",
        )
    with rec_cols[2]:
        insight_card(
            "ช่องทางและข้อความสื่อสาร",
            "ใช้สื่อสังคมออนไลน์เพื่อสร้างการรับรู้ก่อนซื้อ และใช้ร้านสะดวกซื้อเป็นจุดกระตุ้นการซื้อ",
        )

    st.subheader("Message หลัก")
    note_box("กาแฟพร้อมดื่มที่ให้ความรู้สึกใกล้เคียงกาแฟสด พกพาง่าย และคุ้มค่ากว่า")

    st.subheader("แผนดำเนินการเบื้องต้น")
    action_df = pd.DataFrame(
        [
            {"ลำดับ": 1, "แผนดำเนินการ": "เริ่มจากกลุ่ม Cluster 2 และผู้ดื่มกาแฟรายได้ 20,000-29,999 บาท"},
            {"ลำดับ": 2, "แผนดำเนินการ": "ใช้ข้อความสื่อสารเรื่องกาแฟสดพกพาง่ายและคุ้มค่า"},
            {"ลำดับ": 3, "แผนดำเนินการ": "ใช้สื่อสังคมออนไลน์เพื่อสร้างการรับรู้ก่อนซื้อ และใช้รีวิวผู้ใช้จริงช่วยเพิ่มความมั่นใจ"},
            {"ลำดับ": 4, "แผนดำเนินการ": "ใช้ร้านสะดวกซื้อเป็นจุดกระตุ้นการซื้อ"},
            {"ลำดับ": 5, "แผนดำเนินการ": "ทดลองแคมเปญแจกชิมในจุดที่เข้าถึงกลุ่มเป้าหมาย"},
        ]
    )
    st.dataframe(action_df, hide_index=True, use_container_width=True)

    st.caption(
        "ข้อเสนอแนะนี้ใช้สำหรับการวิเคราะห์เบื้องต้นจากข้อมูลแบบสอบถามในโปรเจค หากนำไปใช้จริงควรเก็บข้อมูลเพิ่มและทดลองแคมเปญจริงก่อนตัดสินใจ"
    )

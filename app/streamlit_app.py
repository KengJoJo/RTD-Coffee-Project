import streamlit as st
import pandas as pd
import pickle
import json
from pathlib import Path

# ตั้งค่า Path
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_pipeline.pkl"
FEAT_COLS_PATH = BASE_DIR / "models" / "feature_columns.json"

st.set_page_config(page_title="RTD Coffee Predictor", page_icon="☕", layout="wide")

st.title("☕ Coffee RTD Trial Predictor (Academic Prototype)")
st.info("⚠️ **หมายเหตุ:** แอปพลิเคชันนี้เป็น Prototype สำหรับการนำเสนอวิชา AIE322-325 เท่านั้น ข้อมูลที่กรอกจะถูกนำไปใช้ทำนายผ่านโมเดล Machine Learning ที่ฝึกสอนจากข้อมูลแบบสอบถามจริง")

def load_assets():
    if not MODEL_PATH.exists() or not FEAT_COLS_PATH.exists(): return None, None
    with open(MODEL_PATH, "rb") as f: pipeline = pickle.load(f)
    with open(FEAT_COLS_PATH, "r") as f: feat_cols = json.load(f)
    return pipeline, feat_cols

pipeline, feat_cols = load_assets()

AGE_LABELS = ["<18", "18-22", "23-29", "30-39", "40-49", "50-59", "60+"]
INCOME_LABELS = ["<10k", "10-15k", "15-20k", "20-30k", "30-40k", "40-50k", "50-60k", "60k+"]
SCALE_OPTIONS = {
    "1 - ไม่สำคัญเลย": 1,
    "2 - ไม่ค่อยสำคัญ": 2,
    "3 - ปานกลาง": 3,
    "4 - สำคัญ": 4,
    "5 - สำคัญมาก": 5,
}

def feature_categories(prefix):
    return [c.replace(prefix, "", 1) for c in feat_cols if c.startswith(prefix)]

def default_index(options, preferred):
    return options.index(preferred) if preferred in options else 0

def set_one_hot(input_df, prefix, selected_value):
    col = f"{prefix}{selected_value}"
    if col in input_df.columns:
        input_df[col] = 1

def get_knn_vote_summary(model_pipeline, input_df):
    if not hasattr(model_pipeline, "named_steps"):
        return None
    scaler = model_pipeline.named_steps.get("scaler")
    model = model_pipeline.named_steps.get("model")
    if model is None or model.__class__.__name__ != "KNeighborsClassifier":
        return None

    x_model = scaler.transform(input_df) if scaler is not None else input_df
    distances, indices = model.kneighbors(x_model)
    neighbor_labels = model._y[indices[0]]
    trial_votes = int((neighbor_labels == 1).sum())
    total_neighbors = len(neighbor_labels)
    return {
        "model_name": "K-Nearest Neighbors",
        "k": total_neighbors,
        "trial_votes": trial_votes,
        "no_trial_votes": total_neighbors - trial_votes,
        "avg_distance": float(distances[0].mean()),
    }

if pipeline is None:
    st.error("❌ ไม่พบไฟล์โมเดล! โปรดรันสคริปต์ 01-04 ในเครื่องก่อน")
else:
    gender_options = feature_categories("gender_")
    job_options = feature_categories("job_")
    edu_options = feature_categories("edu_")

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("👤 Demographic")
            age = st.selectbox("ช่วงอายุ", options=list(range(len(AGE_LABELS))), format_func=lambda x: AGE_LABELS[x])
            income = st.selectbox("รายได้", options=list(range(len(INCOME_LABELS))), format_func=lambda x: INCOME_LABELS[x])
            gender = st.selectbox("เพศ", options=gender_options, index=default_index(gender_options, "ไม่ระบุ"))
            job = st.selectbox("อาชีพ", options=job_options, index=default_index(job_options, "พนักงานบริษัทเอกชน"))
            edu = st.selectbox("ระดับการศึกษา", options=edu_options, index=default_index(edu_options, "ปริญญาตรี"))
        with col2:
            st.subheader("🌟 Product Factors (1-5)")
            attr_price_label = st.select_slider(
                "ความสำคัญด้านราคาประหยัด",
                options=list(SCALE_OPTIONS.keys()),
                value="3 - ปานกลาง",
            )
            attr_aroma_label = st.select_slider(
                "ความสำคัญด้านกลิ่นหอมกาแฟ",
                options=list(SCALE_OPTIONS.keys()),
                value="3 - ปานกลาง",
            )
        with col3:
            st.subheader("📢 Social Influence (1-5)")
            inf_blogger_label = st.select_slider(
                "อิทธิพลของ Blogger/Reviewer",
                options=list(SCALE_OPTIONS.keys()),
                value="3 - ปานกลาง",
            )
            
        submit = st.form_submit_button("วิเคราะห์แนวโน้มการลองสินค้า")
        
        if submit:
            attr_price = SCALE_OPTIONS[attr_price_label]
            attr_aroma = SCALE_OPTIONS[attr_aroma_label]
            inf_blogger = SCALE_OPTIONS[inf_blogger_label]

            # สร้าง Input Dataframe ตาม Feature List
            input_df = pd.DataFrame(0, index=[0], columns=feat_cols)
            input_df["age_ordinal"] = age
            input_df["income_ordinal"] = income
            input_df["is_drinker_binary"] = 1
            set_one_hot(input_df, "gender_", gender)
            set_one_hot(input_df, "job_", job)
            set_one_hot(input_df, "edu_", edu)

            for c in feat_cols:
                if c.startswith("likert_"):
                    input_df[c] = 3
            
            # เติมค่า Likert (พยายามจับคู่ชื่อฟีเจอร์ที่ใกล้เคียงที่สุด)
            for c in feat_cols:
                if "ประหยัด" in c: input_df[c] = attr_price
                if "กลิ่นหอม" in c: input_df[c] = attr_aroma
                if "บล็อกเกอร์" in c: input_df[c] = inf_blogger

            # Prediction
            prob = pipeline.predict_proba(input_df)[0][1]
            pred = pipeline.predict(input_df)[0]
            knn_summary = get_knn_vote_summary(pipeline, input_df)
            
            st.divider()
            if pred == 1:
                st.success(f"### ผลการทำนาย: **มีแนวโน้มจะลอง (Trial)**")
            else:
                st.error(f"### ผลการทำนาย: **ไม่สนใจลอง (No Trial)**")

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("โอกาสที่จะลองสินค้า (Trial Probability)", f"{prob:.2%}")
            metric_col2.metric("โอกาสที่จะไม่ลอง (No Trial Probability)", f"{(1 - prob):.2%}")
            st.caption("การจัดกลุ่มใช้เกณฑ์จากโมเดล: ถ้า Trial Probability สูงพอ โมเดลจะจัดเป็น Trial")

            st.markdown("#### โมเดลทำนายยังไง")
            if knn_summary:
                st.write(
                    f"โมเดลที่ใช้คือ **{knn_summary['model_name']}** โดยเทียบผู้ตอบรายนี้กับผู้ตอบแบบสอบถามที่ใกล้เคียงที่สุด "
                    f"**{knn_summary['k']} คน** ในชุด train"
                )
                vote_col1, vote_col2, vote_col3 = st.columns(3)
                vote_col1.metric("เพื่อนบ้านที่เป็น Trial", f"{knn_summary['trial_votes']} / {knn_summary['k']}")
                vote_col2.metric("เพื่อนบ้านที่เป็น No Trial", f"{knn_summary['no_trial_votes']} / {knn_summary['k']}")
                vote_col3.metric("ค่าเฉลี่ยระยะห่าง", f"{knn_summary['avg_distance']:.2f}")
                st.caption(
                    "เพราะ KNN ใช้ k=5 ค่า probability จึงขยับเป็นขั้นละ 20% เช่น 2/5 = 40% "
                    "ถ้าปรับ input แล้วกลุ่มเพื่อนบ้าน 5 คนแรกยังเหมือนเดิม ค่า probability จะยังซ้ำได้"
                )
            else:
                st.write("โมเดลคำนวณจากฟีเจอร์ที่ผ่าน preprocessing และส่งเข้า pipeline ที่บันทึกไว้")

            st.markdown("#### ข้อมูลที่ส่งเข้าโมเดลจากฟอร์ม")
            input_summary = pd.DataFrame(
                [
                    ("ช่วงอายุ", AGE_LABELS[age]),
                    ("รายได้", INCOME_LABELS[income]),
                    ("เพศ", gender),
                    ("อาชีพ", job),
                    ("ระดับการศึกษา", edu),
                    ("ราคาประหยัด", attr_price_label),
                    ("กลิ่นหอมกาแฟ", attr_aroma_label),
                    ("Blogger/Reviewer", inf_blogger_label),
                ],
                columns=["ข้อมูล", "ค่าที่เลือก"],
            )
            st.dataframe(input_summary, hide_index=True, use_container_width=True)
            st.caption("ฟีเจอร์ Likert อื่นที่ไม่ได้เปิดให้กรอกใน prototype ถูกตั้งเป็นค่ากลาง 3 - ปานกลาง เพื่อป้องกัน shape mismatch")

            if pred == 1:
                st.write("**Interpretation:** โมเดลจัดผู้ตอบรายนี้อยู่ในกลุ่มที่มีแนวโน้มทดลองสินค้า โดยปัจจัยที่กรอกมีทิศทางสนับสนุนการตัดสินใจลอง")
            else:
                st.write("**Interpretation:** โมเดลจัดผู้ตอบรายนี้อยู่ในกลุ่มที่ยังมีแนวโน้มทดลองต่ำ จึงอาจต้องใช้แรงจูงใจเพิ่มเติม เช่น การแจกชิม โปรโมชัน หรือข้อความด้านความคุ้มค่า")

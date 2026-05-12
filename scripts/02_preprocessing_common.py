import pandas as pd
import numpy as np
import re
from pathlib import Path

# ตั้งค่า Path
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "BU_Data_from_Survey.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def normalize_str(s):
    if pd.isna(s): return ""
    s = str(s).strip()
    s = re.sub(r'\s+', '', s)
    return s

def clean_feature_name(v):
    return re.sub(r"[^0-9A-Za-zก-๙_]+", "_", str(v)).strip("_")

def extract_digit(s):
    if pd.isna(s): return np.nan
    m = re.search(r'(\d)', str(s))
    return int(m.group(1)) if m else np.nan

def find_col(columns, keywords):
    for col in columns:
        if any(kw in col for kw in keywords):
            return col
    return None

def run_preprocessing():
    print("--- เริ่มการทำ Preprocessing (Final Coffee-Only Mode) ---")
    raw_df = pd.read_csv(RAW_DATA_PATH, header=None, encoding="utf-8-sig")
    header_idx = raw_df[raw_df[0].astype(str).str.contains("Timestamp|ประทับเวลา", na=False)].index[0]
    
    df = pd.read_csv(RAW_DATA_PATH, skiprows=header_idx, encoding="utf-8-sig")
    df = df.dropna(how="all").copy()
    
    # ระบุคอลัมน์สำคัญ
    COL_AGE = find_col(df.columns, ["อายุ"])
    COL_INCOME = find_col(df.columns, ["รายได้"])
    COL_DRINKER = find_col(df.columns, ["คุณดื่มกาแฟหรือไม่"])
    COL_TARGET = find_col(df.columns, ["คุณจะลองหรือไม่"])

    # 1. กรองเฉพาะคนดื่มกาแฟ
    df['is_drinker_binary'] = df[COL_DRINKER].apply(lambda x: 0 if "ไม่ดื่ม" in str(x) else (1 if "ดื่ม" in str(x) else np.nan))
    df = df[df['is_drinker_binary'] == 1].copy()
    
    # 2. จัดการ Demographic
    age_map = {"ต่ำกว่า18":0, "18-22":1, "23-29":2, "30-39":3, "40-49":4, "50-59":5, "60":6, "50ปีขึ้นไป":5}
    df["age_ordinal"] = df[COL_AGE].apply(normalize_str).apply(lambda x: next((v for k, v in age_map.items() if k in x), 2))

    income_map = {"ต่ำกว่า10,000":0, "10,001-14,999":1, "15,000-19,999":2, "20,000-29,999":3, "30,000-39,999":4, "40,000-49,999":5, "50,000-59,999":6, "60,000":7}
    df["income_ordinal"] = df[COL_INCOME].apply(normalize_str).apply(lambda x: next((v for k, v in income_map.items() if k in x), 3))

    # 3. จัดการ Target
    df["target_binary_trial"] = df[COL_TARGET].apply(lambda x: 0 if "ไม่ลอง" in str(x) else (1 if "ลอง" in str(x) else np.nan))
    df["target_trial_reason"] = df[COL_TARGET].fillna("ไม่ระบุ")
    df = df.dropna(subset=["target_binary_trial"]).copy()
    df["target_binary_trial"] = df["target_binary_trial"].astype(int)

    # 4. Extract Likert Features (ตัดคอลัมน์ "ชา" ทิ้ง)
    # เราจะกรองไม่เอาคอลัมน์ที่มีคำว่า "ชา" หรือ "tea"
    likert_cols = [c for c in df.columns if ("(5 =" in c or "เรียงตามลำดับ" in c) and ("ชา" not in c and "tea" not in c.lower())]
    likert_df = pd.DataFrame(index=df.index)
    for col in likert_cols:
        clean_name = "likert_" + clean_feature_name(col.split("[")[-1].replace("]", "") if "[" in col else col[:15])
        likert_df[clean_name] = df[col].apply(extract_digit).fillna(3).astype(int)

    # 5. One-Hot (ตัดเรื่องชา)
    cat_cols = [find_col(df.columns, ["เพศ"]), find_col(df.columns, ["อาชีพ"]), find_col(df.columns, ["ระดับการศึกษา"])]
    df_onehot = pd.get_dummies(df[cat_cols].fillna("ไม่ระบุ"), prefix=["gender", "job", "edu"], dtype=int)

    # 6. Concat Final
    final_supervised = pd.concat([df[["target_binary_trial", "age_ordinal", "income_ordinal", "is_drinker_binary"]], df_onehot, likert_df], axis=1)
    final_unsupervised = final_supervised.drop(columns=["target_binary_trial"])

    # Export
    df_readable = df.rename(columns={COL_AGE: "age", COL_INCOME: "income"})
    df_readable.to_csv(PROCESSED_DIR / "cleaned_readable_survey.csv", index=False, encoding="utf-8-sig")
    final_supervised.to_csv(PROCESSED_DIR / "final_supervised_dataset.csv", index=False)
    final_unsupervised.to_csv(PROCESSED_DIR / "final_unsupervised_dataset.csv", index=False)
    
    print(f"--- Preprocessing สำเร็จ! เหลือข้อมูล {len(df)} แถว (ตัดฟีเจอร์ชาออกแล้ว) ---")

if __name__ == "__main__":
    run_preprocessing()

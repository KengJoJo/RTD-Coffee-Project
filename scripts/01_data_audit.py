import pandas as pd
import numpy as np
import os
from pathlib import Path

# ตั้งค่า Path
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "BU_Data_from_Survey.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_audit():
    print("--- เริ่มการตรวจสอบข้อมูล (Data Audit - Refactored) ---")
    if not RAW_DATA_PATH.exists():
        print(f"Error: ไม่พบไฟล์ที่ {RAW_DATA_PATH}")
        return

    # อ่านไฟล์ (header=1)
    df_raw = pd.read_csv(RAW_DATA_PATH, header=1, encoding="utf-8-sig")
    total_rows_before = len(df_raw)
    
    # ลบแถวว่างทั้งหมด
    df_clean = df_raw.dropna(how="all").copy()
    total_rows_after = len(df_clean)
    
    # บันทึก Row Summary
    pd.DataFrame({"Stage": ["Before Cleaning", "After Cleaning"], "Row Count": [total_rows_before, total_rows_after]}).to_csv(OUTPUT_DIR / "row_summary.csv", index=False)
    
    # ตรวจสอบ Target (Logic ใหม่: ป้องกัน "ไม่ลอง" ติดไปกับ "ลอง")
    target_col = "ถ้ามีแบรนด์กาแฟพร้อมดื่ม (Ready to drink) ออกใหม่ คุณจะลองหรือไม่"
    if target_col in df_clean.columns:
        s = df_clean[target_col].astype(str)
        mask_no = s.str.contains("ไม่ลอง", na=False)
        mask_yes = s.str.contains("ลอง", na=False) & (~mask_no)
        
        counts = {
            "ลอง (Trial)": mask_yes.sum(),
            "ไม่ลอง (No Trial)": mask_no.sum(),
            "อื่นๆ/ไม่ระบุ": total_rows_after - (mask_yes.sum() + mask_no.sum())
        }
        pd.DataFrame(list(counts.items()), columns=["Category", "Count"]).to_csv(OUTPUT_DIR / "target_raw_answer_counts.csv", index=False)

    # ระบุ Leakage Risk (ละเอียดขึ้น)
    leakage_risks = [
        {"Column": target_col, "Reason": "Direct Target Column"},
        {"Column": "trial_text", "Reason": "Contains reason for trial (Direct leakage)"},
        {"Column": "review_interest", "Reason": "Feature derived from target question text"},
        {"Column": "sampling_interest", "Reason": "Feature derived from target question text"},
        {"Column": "promotion_interest", "Reason": "Feature derived from target question text"},
        {"Column": "advertising_interest", "Reason": "Feature derived from target question text"}
    ]
    pd.DataFrame(leakage_risks).to_csv(OUTPUT_DIR / "leakage_risk_columns.csv", index=False)
    
    # Missing Values
    df_clean.isnull().sum().reset_index(name="Missing Count").to_csv(OUTPUT_DIR / "missing_values.csv", index=False)
    
    print(f"--- Audit เสร็จสิ้น (Rows: {total_rows_after}) ---")

if __name__ == "__main__":
    run_audit()

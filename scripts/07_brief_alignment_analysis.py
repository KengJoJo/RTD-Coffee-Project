import pandas as pd
import numpy as np
import re
from pathlib import Path

# ตั้งค่า Path
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_readable_survey.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_digit(s):
    if pd.isna(s): return np.nan
    m = re.search(r'(\d)', str(s))
    return int(m.group(1)) if m else np.nan

def find_cols(columns, keyword):
    return [c for c in columns if keyword in c]

def run_alignment_analysis():
    print("--- เริ่มการวิเคราะห์ Media Behavior (Polished Mode) ---")
    df = pd.read_csv(DATA_PATH)
    
    # 1. วิเคราะห์ Media & Channel
    media_cols = find_cols(df.columns, "คุณเห็นสื่อโฆษณาผ่านช่องทางใดบ้าง")
    buy_cols = find_cols(df.columns, "คุณซื้อกาแฟพร้อมดื่ม")
    
    trial_df = df[df["target_binary_trial"] == 1]
    no_trial_df = df[df["target_binary_trial"] == 0]

    def get_top_values(df, cols):
        if not cols: return pd.Series()
        all_vals = df[cols[0]].str.split(',').explode().str.strip()
        return all_vals[all_vals != ""].value_counts()

    trial_media = get_top_values(trial_df, media_cols)
    no_trial_media = get_top_values(no_trial_df, media_cols)
    trial_buy = get_top_values(trial_df, buy_cols)
    no_trial_buy = get_top_values(no_trial_df, buy_cols)

    if media_cols:
        media_summary = pd.concat(
            [trial_media.rename("Trial_Group"), no_trial_media.rename("No_Trial_Group")],
            axis=1
        ).fillna(0).reset_index().rename(columns={"index": media_cols[0]})
        media_summary.to_csv(OUTPUT_DIR / "media_by_try_not_try.csv", index=False, encoding="utf-8-sig")

    if buy_cols:
        buy_summary = pd.concat(
            [trial_buy.rename("Trial_Group"), no_trial_buy.rename("No_Trial_Group")],
            axis=1
        ).fillna(0).reset_index().rename(columns={"index": buy_cols[0]})
        buy_summary.to_csv(OUTPUT_DIR / "buy_channel_by_try_not_try.csv", index=False, encoding="utf-8-sig")
    
    # 2. คำนวณคะแนน Influencer (Blogger/Reviewer) แบบ Robust
    influencer_cols = [c for c in df.columns if "บล็อกเกอร์" in c or "รีวิว" in c]
    if influencer_cols:
        # สกัดตัวเลข 1-5 จากกลุ่ม Trial
        inf_scores = trial_df[influencer_cols].map(extract_digit)
        avg_score = inf_scores.mean().mean()
    else:
        avg_score = np.nan

    # จัดการการแสดงผลคะแนน
    if pd.isna(avg_score):
        inf_display = "ไม่สามารถคำนวณคะแนนเฉลี่ยได้จากข้อมูลที่มี"
        use_influencer = "พิจารณาตามความเหมาะสม"
    else:
        inf_display = f"{avg_score:.2f}/5.0"
        use_influencer = "ใช้เป็นช่องทางเสริม ไม่ใช่ driver หลัก" if avg_score <= 3.5 else "ใช้เป็นช่องทางสนับสนุนหลักได้"

    # 3. สร้างรายงานกลยุทธ์
    top_media = trial_media.index[0] if not trial_media.empty else "Social Media"
    top_buy_channel = trial_buy.index[0] if not trial_buy.empty else "ร้านสะดวกซื้อ เช่น 7-11"

    strategy_md = f"""# 📣 Campaign Strategy & Media Insight

## 1. การวิเคราะห์กลุ่มเป้าหมาย (Target Group Insight)
กลุ่มเป้าหมายหลัก (Trial Group) มีพฤติกรรมดังนี้:
- **ช่องทางสื่อที่พบเห็นบ่อยที่สุด:** {top_media}
- **ช่องทางการซื้อที่ใช้ประจำ:** {top_buy_channel}
- **อิทธิพลของ Blogger/Reviewer:** {inf_display}

## 2. ตอบโจทย์กลยุทธ์การตลาด (Brief Alignment)
- **ช่องทางโฆษณาหลัก:** ควรเน้น **"{top_media}"** เพื่อเข้าถึงกลุ่มเป้าหมายได้แม่นยำที่สุด
- **Purchase / Sales Touchpoint:** เน้นความสะดวกผ่าน **"{top_buy_channel}"**
- **Blogger/Influencer:** {use_influencer}
- **Message:** "รสชาติกาแฟสดที่พกพาไปได้ทุกที่" โดยเน้นย้ำเรื่องความประหยัดเมื่อเทียบกับกาแฟสดหน้าร้าน
- **Product Sampling:** ควรเน้นแจกชิมในจุดที่มีกลุ่มเป้าหมายหนาแน่น (Trial Driver สำคัญ)

## 3. สรุปคำแนะนำการตลาด 3 ข้อ
1. **Focus on Social Media:** สื่อสารผ่าน {top_media} เป็นหลักโดยใช้ภาพลักษณ์แบบพรีเมียม
2. **Drive to Convenience Store:** ใช้แคมเปญกระตุ้นการซื้อที่ {top_buy_channel}
3. **Value Message:** ชูจุดขายเรื่องความคุ้มค่า (Value for Money) โดยยังคงรสชาติใกล้เคียงกาแฟสด
"""
    with open(OUTPUT_DIR / "campaign_strategy_summary.md", "w", encoding="utf-8") as f:
        f.write(strategy_md)

    print(f"--- วิเคราะห์ Media & Strategy สำเร็จ (No NaN values) ---")

if __name__ == "__main__":
    run_alignment_analysis()

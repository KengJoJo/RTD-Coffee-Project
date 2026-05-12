import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# ตั้งค่า Path
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_readable_survey.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Tahoma"

def run_analysis():
    print("--- เริ่มการวิเคราะห์เชิงลึก (Scientific Insight Mode) ---")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Trial Reason (เฉพาะกลุ่มที่จะลองจริง Trial == 1)
    trial_df = df[df["target_binary_trial"] == 1].copy()
    # แก้ไขปัญหา Multiple Choice (ผู้ตอบสามารถเลือกได้หลายเหตุผลคั่นด้วยลูกน้ำ)
    trial_df["target_trial_reason_list"] = trial_df["target_trial_reason"].str.split(',')
    exploded_trial_df = trial_df.explode("target_trial_reason_list")
    exploded_trial_df["target_trial_reason_list"] = exploded_trial_df["target_trial_reason_list"].str.strip()
    
    reason_counts = exploded_trial_df["target_trial_reason_list"].value_counts()
    # กรองเอาเหตุผลที่สื่อถึงความอยากลองจริงๆ (ตัด "ไม่ลอง" ออก)
    top_reasons = reason_counts[~reason_counts.index.str.contains("ไม่ลอง", na=False)]
    top_reason = top_reasons.index[0] if len(top_reasons) > 0 else "ไม่ระบุ"
    
    # 2. Try Rate by Income (เฉพาะกลุ่มที่มี Sample Size >= 5)
    income_stats = df.groupby("income")["target_binary_trial"].agg(['count', 'mean']).reset_index()
    income_stats['try_rate_pct'] = income_stats['mean'] * 100
    
    # กรองเอาเฉพาะกลุ่มที่มีคนตอบอย่างน้อย 5 คน เพื่อความน่าเชื่อถือทางสถิติ
    valid_income = income_stats[income_stats['count'] >= 5]
    if len(valid_income) > 0:
        best_income = valid_income.sort_values(by='try_rate_pct', ascending=False).iloc[0]
        best_income_group = best_income['income']
        best_income_rate = best_income['try_rate_pct']
    else:
        best_income_group = "N/A (Sample Size Too Small)"
        best_income_rate = 0

    # 3. Feature Comparison
    proc_df = pd.read_csv(BASE_DIR / "data" / "processed" / "final_supervised_dataset.csv")
    likert_cols = [c for c in proc_df.columns if c.startswith("likert_")]
    mean_diff = proc_df.groupby("target_binary_trial")[likert_cols].mean().T
    mean_diff.columns = ["No Trial", "Trial"]
    mean_diff["Diff"] = mean_diff["Trial"] - mean_diff["No Trial"]
    top_feature = mean_diff["Diff"].idxmax()
    diff_val = mean_diff.loc[top_feature, "Diff"]

    # 4. Write Real Insight Markdown
    insight_text = f"""# 💡 Business Insights (Validated Analysis)

## 1. เหตุผลหลักในการตัดสินใจลอง (Trial Driver)
กลุ่มเป้าหมายส่วนใหญ่สนใจลองสินค้าใหม่เพราะ: **"{top_reason}"**
*(วิเคราะห์เฉพาะผู้ที่ตอบ "ลอง" จำนวน {len(trial_df)} คน)*

## 2. กลุ่มเป้าหมายตามรายได้ที่มีศักยภาพสูงสุด (Priority Segment)
- **กลุ่มที่มี Try Rate สูงสุด:** คือกลุ่ม **"{best_income_group}"**
- **อัตราการลอง (Try Rate):** **{best_income_rate:.1f}%**
*(วิเคราะห์เฉพาะกลุ่มที่มีกลุ่มตัวอย่างอย่างน้อย 5 คน เพื่อความแม่นยำ)*

## 3. ปัจจัยสินค้าที่ดึงดูดใจได้ดีที่สุด (Key Differentiator)
กลุ่มที่ตัดสินใจลอง ให้ความสำคัญกับปัจจัย **"{top_feature.replace('likert_', '')}"** สูงกว่ากลุ่มที่ไม่ลองอย่างมีนัยสำคัญ (คะแนนต่างกัน {diff_val:.2f} คะแนน)

## 4. ข้อเสนอแนะ (Recommendation)
แบรนด์ควรทำแคมเปญมุ่งเน้นไปที่กลุ่มรายได้ระดับ {best_income_group} โดยชูจุดเด่นเรื่อง {top_feature.replace('likert_', '')} เป็นแกนหลักในการสื่อสาร
"""
    with open(OUTPUT_DIR / "business_insight_try_not_try.md", "w", encoding="utf-8") as f:
        f.write(insight_text)

    # -----------------------------------------------------
    # 5. Generate EDA Plots (สำหรับนำไปใส่สไลด์นำเสนอ)
    # -----------------------------------------------------
    FIG_DIR = OUTPUT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 5.1 Target Distribution (Trial vs No Trial)
    plt.figure(figsize=(6, 6))
    target_counts = df["target_binary_trial"].value_counts()
    labels = ["Trial (สนใจลอง)", "No Trial (ไม่สนใจ)"] if len(target_counts) == 2 else ["Trial"]
    plt.pie(target_counts, labels=labels, autopct='%1.1f%%', colors=['#4C72B0', '#DD8452'], startangle=90)
    plt.title("สัดส่วนความสนใจทดลองสินค้าใหม่ (Target Distribution)")
    plt.savefig(FIG_DIR / "target_distribution.png", bbox_inches="tight")
    plt.close()

    # 5.2 Age Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(y="age", data=df, order=df["age"].value_counts().index, palette="Blues_d")
    plt.title("การกระจายตัวของช่วงอายุ (Age Distribution)")
    plt.xlabel("จำนวนคน")
    plt.ylabel("ช่วงอายุ")
    plt.savefig(FIG_DIR / "age_distribution.png", bbox_inches="tight")
    plt.close()

    # 5.3 Income Distribution by Trial
    plt.figure(figsize=(10, 6))
    sns.countplot(y="income", hue="target_binary_trial", data=df, 
                  order=df["income"].value_counts().index, palette="Set2")
    plt.title("การกระจายตัวของรายได้และพฤติกรรมการลอง (Income vs Trial)")
    plt.xlabel("จำนวนคน")
    plt.ylabel("ช่วงรายได้")
    plt.legend(title="Target", labels=["No Trial", "Trial"])
    plt.savefig(FIG_DIR / "income_vs_trial.png", bbox_inches="tight")
    plt.close()

    import textwrap
    def wrap_labels(labels, width=40):
        return [textwrap.fill(str(label), width=width) for label in labels]

    # 5.4 Reason to Try / Not Try
    # เหตุผลที่อยากลอง
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_reasons.values, y=wrap_labels(top_reasons.index), palette="Greens_d")
    plt.title("เหตุผลหลักที่กลุ่มเป้าหมายอยากลองสินค้าใหม่ (Reasons to Try)")
    plt.xlabel("จำนวนคนที่เลือกเหตุผลนี้")
    plt.savefig(FIG_DIR / "reason_to_try.png", bbox_inches="tight")
    plt.close()

    # เหตุผลที่ไม่อยากลอง
    no_trial_df = df[df["target_binary_trial"] == 0].copy()
    no_trial_df["target_trial_reason_list"] = no_trial_df["target_trial_reason"].str.split(',')
    exploded_no_trial_df = no_trial_df.explode("target_trial_reason_list")
    exploded_no_trial_df["target_trial_reason_list"] = exploded_no_trial_df["target_trial_reason_list"].str.strip()
    
    no_trial_reasons = exploded_no_trial_df["target_trial_reason_list"].value_counts()
    if len(no_trial_reasons) > 0:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=no_trial_reasons.values, y=wrap_labels(no_trial_reasons.index), palette="Reds_d")
        plt.title("เหตุผลหลักที่กลุ่มเป้าหมายไม่อยากลองสินค้าใหม่ (Reasons NOT to Try)")
        plt.xlabel("จำนวนคนที่เลือกเหตุผลนี้")
        plt.savefig(FIG_DIR / "reason_not_to_try.png", bbox_inches="tight")
        plt.close()

    # 5.5 Correlation Heatmap (ดูความสัมพันธ์ของฟีเจอร์ก่อนเข้าโมเดล)
    plt.figure(figsize=(14, 12))
    df_corr = proc_df.copy()
    
    # ลดความยาวของชื่อคอลัมน์เพื่อให้กราฟไม่อ่านยาก
    rename_dict = {}
    for c in df_corr.columns:
        new_name = c.replace("likert_", "").replace("target_binary_", "").replace("job_", "").replace("edu_", "")
        rename_dict[c] = new_name[:30] # ตัดคำที่ยาวเกิน 30 ตัวอักษร
    df_corr = df_corr.rename(columns=rename_dict)
    
    corr_matrix = df_corr.corr()
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False, xticklabels=True, yticklabels=True)
    plt.title("Correlation Heatmap ของฟีเจอร์ทั้งหมด")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig(FIG_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # 5.6 Cross-Analysis: Reason to Try by Income
    if len(exploded_trial_df) > 0 and "income" in exploded_trial_df.columns:
        plt.figure(figsize=(12, 8))
        top_5_reasons = exploded_trial_df["target_trial_reason_list"].value_counts().nlargest(5).index
        plot_df = exploded_trial_df[exploded_trial_df["target_trial_reason_list"].isin(top_5_reasons)].copy()
        plot_df["target_trial_reason_wrap"] = wrap_labels(plot_df["target_trial_reason_list"])
        sns.countplot(y="target_trial_reason_wrap", hue="income", data=plot_df, palette="viridis")
        plt.title("เหตุผลที่อยากลองสินค้า แยกตามกลุ่มรายได้")
        plt.xlabel("จำนวนคนที่เลือกเหตุผลนี้")
        plt.ylabel("เหตุผล")
        plt.legend(title="รายได้ (Income)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(FIG_DIR / "reason_to_try_by_income.png", bbox_inches="tight")
        plt.close()

    # 5.7 Cross-Analysis: Reason NOT to Try by Income
    if len(exploded_no_trial_df) > 0 and "income" in exploded_no_trial_df.columns:
        plt.figure(figsize=(12, 8))
        top_5_no_reasons = exploded_no_trial_df["target_trial_reason_list"].value_counts().nlargest(5).index
        plot_no_df = exploded_no_trial_df[exploded_no_trial_df["target_trial_reason_list"].isin(top_5_no_reasons)].copy()
        plot_no_df["target_trial_reason_wrap"] = wrap_labels(plot_no_df["target_trial_reason_list"])
        sns.countplot(y="target_trial_reason_wrap", hue="income", data=plot_no_df, palette="magma")
        plt.title("เหตุผลที่ไม่อยากลองสินค้า แยกตามกลุ่มรายได้")
        plt.xlabel("จำนวนคนที่เลือกเหตุผลนี้")
        plt.ylabel("เหตุผล")
        plt.legend(title="รายได้ (Income)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(FIG_DIR / "reason_not_to_try_by_income.png", bbox_inches="tight")
        plt.close()

    # 5.8 Macro Marketing Drivers (Feature Engineering for Insight)
    df_macro = proc_df.copy()
    
    # 1. Product Quality (คุณภาพสินค้า)
    prod_cols = [c for c in df_macro.columns if "กลิ่น" in c or "ดีด" in c or "แหล่งที่มา" in c or "โภชนาการ" in c]
    df_macro["Driver: Product Quality"] = df_macro[prod_cols].mean(axis=1)
    
    # 2. Value & Convenience (ความคุ้มค่าและสะดวก)
    func_cols = [c for c in df_macro.columns if "สะดวก" in c or "ประหยัด" in c]
    df_macro["Driver: Value & Convenience"] = df_macro[func_cols].mean(axis=1)
    
    # 3. Brand & Image (ภาพลักษณ์และแพ็คเกจ)
    emo_cols = [c for c in df_macro.columns if "แบรนด์" in c or "พรีเมียม" in c or "แพ็คเกจ" in c]
    df_macro["Driver: Brand & Image"] = df_macro[emo_cols].mean(axis=1)
    
    # 4. Social Influence (อิทธิพลสื่อและคนรอบข้าง)
    soc_cols = [c for c in df_macro.columns if "รีวิว" in c or "เพื่อน" in c]
    df_macro["Driver: Social Influence"] = df_macro[soc_cols].mean(axis=1)
    
    macro_cols = ["Driver: Product Quality", "Driver: Value & Convenience", "Driver: Brand & Image", "Driver: Social Influence"]
    
    macro_mean = df_macro.groupby("target_binary_trial")[macro_cols].mean().reset_index()
    macro_mean["target"] = macro_mean["target_binary_trial"].map({0: "No Trial", 1: "Trial"})
    macro_melt = macro_mean.melt(id_vars="target", value_vars=macro_cols, var_name="Marketing Driver", value_name="Average Score")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Average Score", y="Marketing Driver", hue="target", data=macro_melt, palette="Set1")
    plt.title("เปรียบเทียบ 4 ปัจจัยขับเคลื่อนหลัก (Macro Drivers) ระหว่างคนที่อยากลองและไม่อยากลอง")
    plt.xlabel("คะแนนเฉลี่ย (ความสำคัญ)")
    plt.ylabel("เสาหลักการตลาด (Marketing Pillars)")
    plt.legend(title="Target", loc='lower right')
    plt.xlim(0, 5) # คะแนนเต็ม 5
    plt.savefig(FIG_DIR / "macro_drivers_comparison.png", bbox_inches="tight")
    plt.close()

    print(f"--- วิเคราะห์และสร้างกราฟ EDA เสร็จสิ้น บันทึกผลลัพธ์ใน {OUTPUT_DIR} ---")

if __name__ == "__main__":
    run_analysis()

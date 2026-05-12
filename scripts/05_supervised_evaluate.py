import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ตั้งค่า Path
BASE_DIR = Path(__file__).parent.parent
INTERIM_DIR = BASE_DIR / "data" / "interim"
MODEL_PATH = BASE_DIR / "models" / "best_pipeline.pkl"
OUTPUT_DIR = BASE_DIR / "outputs" / "supervised"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Tahoma"

def run_evaluation():
    print("--- เริ่มการประเมินโมเดล (Academic Reporting Mode) ---")
    
    # 1. โหลดข้อมูล
    X_train = pd.read_csv(INTERIM_DIR / "X_train.csv")
    y_train = pd.read_csv(INTERIM_DIR / "y_train.csv").values.ravel()
    X_test = pd.read_csv(INTERIM_DIR / "X_test.csv")
    y_test = pd.read_csv(INTERIM_DIR / "y_test.csv").values.ravel()
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
        
    y_pred = pipeline.predict(X_test)
    
    # 2. Confusion Matrix & Breakdown
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,cm[0,0])
    
    actual_no_trial = (y_test == 0).sum()
    actual_trial = (y_test == 1).sum()
    no_trial_recall = tn / actual_no_trial if actual_no_trial else 0
    trial_recall = tp / actual_trial if actual_trial else 0

    # 3. Save Detailed Model Report
    report_text = f"""# 📊 Model Evaluation Report

## 1. ผลการทดสอบโมเดล (Test Set Breakdown)
- **จำนวนกลุ่มตัวอย่างที่ใช้ทดสอบ (Test Set Size):** {len(y_test)} คน
- **กลุ่มที่ไม่ลองจริง (Actual No Trial):** {actual_no_trial} คน | **โมเดลทายถูก:** {tn} คน | **Recall:** {no_trial_recall:.2f}
- **กลุ่มที่จะลองจริง (Actual Trial):** {actual_trial} คน | **โมเดลทายถูก:** {tp} คน | **Recall:** {trial_recall:.2f}

## 2. หมายเหตุประกอบการวิเคราะห์ (Academic Notes)
- **ประชากรเป้าหมาย:** โมเดลวิเคราะห์เฉพาะผู้ที่ดื่มกาแฟเท่านั้น (is_drinker_binary == 1)
- **ประสิทธิภาพ:** โมเดลเหมาะกับการทำ "Positive Screening" เพื่อคัดกรองกลุ่มที่มีแนวโน้มจะลองสินค้า เพราะ Recall ของกลุ่ม Trial สูง
- **ข้อจำกัด:** การจำแนกกลุ่ม No Trial ยังมีข้อจำกัดจาก **Class Imbalance** (สัดส่วนผู้ที่อยากลองมีมากกว่าผู้ที่ไม่ลองอย่างชัดเจน) จึงไม่ควรสรุปว่าโมเดลแม่นยำเท่ากันทุกกลุ่ม
- **การใช้งาน:** ผลลัพธ์นี้เป็น prototype เพื่อการเรียนและการนำเสนอ ไม่ใช่ production model สำหรับตัดสินใจทางธุรกิจโดยตรง

## 3. Decision Tree Analysis (Explainable AI)
(แผนภาพด้านล่างสร้างจากข้อมูลการเรียนรู้ (Training Set) เพื่อแสดงตรรกะการตัดสินใจของโมเดล)
"""
    with open(OUTPUT_DIR / "model_evaluation_summary.md", "w", encoding="utf-8") as f:
        f.write(report_text)
        
    # 3.1 Save Full Classification Report
    class_report = classification_report(y_test, y_pred, target_names=["No Trial", "Trial"])
    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("=== Classification Report ===\n")
        f.write(class_report)

    # 3.2 Save Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Trial", "Trial"], yticklabels=["No Trial", "Trial"])
    plt.title("Confusion Matrix (ผลการทำนายเทียบกับของจริง)")
    plt.ylabel("ของจริง (Actual)")
    plt.xlabel("สิ่งที่โมเดลทำนาย (Predicted)")
    plt.savefig(OUTPUT_DIR / "confusion_matrix_heatmap.png", bbox_inches="tight")
    plt.close()

    # 4. Decision Tree Visualization (ใช้ Train Set ตามหลักที่ถูกต้อง)
    dt_explainer = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_explainer.fit(X_train, y_train)
    plt.figure(figsize=(20, 10))
    plot_tree(dt_explainer, feature_names=X_train.columns, class_names=["No Trial", "Trial"], 
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree Logic (Trained on X_train)")
    plt.savefig(OUTPUT_DIR / "decision_tree_explainable.png")
    plt.close()

    print(f"--- ประเมินเสร็จสิ้น บันทึกรายงานสรุปวิชาการและรูปภาพใน {OUTPUT_DIR} ---")

if __name__ == "__main__":
    run_evaluation()

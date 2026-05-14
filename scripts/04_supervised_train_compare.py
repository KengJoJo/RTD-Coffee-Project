import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ตั้งค่า Path
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_supervised_dataset.csv"
INTERIM_DIR = BASE_DIR / "data" / "interim"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "supervised"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_training():
    print("--- เริ่มการฝึกสอน Supervised Learning (Pipeline Mode) ---")
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop(columns=["target_binary_trial"])
    y = df["target_binary_trial"]
    
    # บันทึก Feature Columns สำหรับ Web App
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(X.columns.tolist(), f, ensure_ascii=False, indent=2)
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # บันทึก Dataset สำหรับสคริปต์ 05 (ทั้ง Train และ Test)
    X_train.to_csv(INTERIM_DIR / "X_train.csv", index=False)
    y_train.to_csv(INTERIM_DIR / "y_train.csv", index=False)
    X_test.to_csv(INTERIM_DIR / "X_test.csv", index=False)
    y_test.to_csv(INTERIM_DIR / "y_test.csv", index=False)
    
    # นิยามโมเดล (ใช้ Pipeline สำหรับโมเดลที่ต้องการ Scaling)
    # GridSearch เทียบทั้ง uniform/distance แต่เลือก distance candidate ที่ดีที่สุด
    # เพื่อลดปัญหา probability เป็นขั้น 20/40/60/80 จาก uniform voting
    knn_grid = GridSearchCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
        param_grid={
            "model__n_neighbors": [5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        scoring="f1",
        cv=5,
        refit=True,
    )

    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
        "KNN (distance-weighted tuned)": knn_grid,
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    best_f1 = 0
    best_pipeline = None
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        best_params = model.best_params_ if hasattr(model, "best_params_") else {}

        if name == "KNN (distance-weighted tuned)":
            cv_results = pd.DataFrame(model.cv_results_)
            distance_rows = cv_results[cv_results["param_model__weights"] == "distance"]
            best_distance_idx = distance_rows["mean_test_score"].idxmax()
            best_params = model.cv_results_["params"][best_distance_idx]
            fitted_model = clone(model.estimator).set_params(**best_params)
            fitted_model.fit(X_train, y_train)

        y_pred = fitted_model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1,
            "Best Params": json.dumps(best_params, ensure_ascii=False) if best_params else ""
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = fitted_model
            best_model_name = name

    # บันทึกผลเปรียบเทียบ
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # แสดงตารางผลลัพธ์ใน Terminal เพื่อใช้ประกอบการนำเสนอ
    display_df = results_df.sort_values(by="F1-Score", ascending=False).copy()
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    display_df[metric_cols] = display_df[metric_cols].round(3)
    print("\n=== Model Performance Comparison on Test Set ===")
    print(display_df.to_string(index=False))
    print(f"\nSelected Best Model: {best_model_name} (highest F1-Score = {best_f1:.3f})")
    print("Reason: F1-Score balances Precision and Recall, which is useful when Trial / No Trial classes are imbalanced.\n")
    
    # สร้างรูปกราฟเปรียบเทียบทุก metric หลัก
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams["font.family"] = "Tahoma"
    
    plt.figure(figsize=(12, 6))
    melted_df = results_df.melt(id_vars="Model", value_vars=metric_cols, 
                                var_name="Metric", value_name="Score")
    sns.barplot(x="Model", y="Score", hue="Metric", data=melted_df, palette="viridis")
    plt.title("การเปรียบเทียบประสิทธิภาพโมเดล: Accuracy, Precision, Recall และ F1-Score")
    plt.ylim(0, 1.1)
    plt.ylabel("Score (0.0 - 1.0)")
    plt.xlabel("Model")
    plt.legend(loc='lower right')
    plt.savefig(OUTPUT_DIR / "model_comparison_chart.png", bbox_inches="tight")
    plt.close()
    
    # บันทึก Best Pipeline
    with open(MODEL_DIR / "best_pipeline.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)
        
    print(f"--- ฝึกสอนเสร็จสิ้น โมเดลที่ดีที่สุดคือ {best_model_name} (F1: {best_f1:.2f}) ---")

if __name__ == "__main__":
    run_training()

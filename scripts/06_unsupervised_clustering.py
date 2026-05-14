import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ตั้งค่า Path
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_unsupervised_dataset.csv"
TARGET_PATH = BASE_DIR / "data" / "processed" / "final_supervised_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "unsupervised"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_clustering():
    print("--- เริ่มการทำ Clustering (Coffee Segment Profiler Mode) ---")
    df = pd.read_csv(DATA_PATH)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # 1. K-Means Evaluation (Elbow & Silhouette)
    inertias = []
    sil_scores = []
    k_range = range(2, 8)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
        
    plt.rcParams["font.family"] = "Tahoma"

    # กราฟ Elbow
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, marker='o', linestyle='--')
    plt.title("The Elbow Method (หาจำนวนกลุ่มที่เหมาะสมที่สุด)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.savefig(OUTPUT_DIR / "elbow_method.png", bbox_inches="tight")
    plt.close()

    # กราฟ Silhouette
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, sil_scores, marker='s', color='orange', linestyle='-')
    plt.title("Silhouette Score (ตรวจสอบคุณภาพการแบ่งกลุ่ม)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.savefig(OUTPUT_DIR / "silhouette_score.png", bbox_inches="tight")
    plt.close()

    # 2. Final K-Means (K=4)
    selected_k = 4 # ใช้ K=4 ตามที่วิเคราะห์ไว้ก่อนหน้านี้ว่าแบ่งกลุ่มได้ชัดเจนทางธุรกิจ
    kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 3. PCA & Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="viridis", s=100)
    plt.title(f"Customer Segments Plot (PCA) K={selected_k}")
    plt.savefig(OUTPUT_DIR / "pca_clusters.png", bbox_inches="tight")
    plt.close()

    # 4. Cluster Profile Calculation & Heatmap
    target_df = pd.read_csv(TARGET_PATH)
    profile_df = df.copy()
    profile_df['cluster'] = clusters
    profile_df['target_trial'] = target_df['target_binary_trial']
    
    cluster_profile = profile_df.groupby('cluster').mean()
    cluster_counts = profile_df['cluster'].value_counts().sort_index()
    cluster_summary = (
        pd.DataFrame(
            {
                "cluster": cluster_counts.index,
                "count": cluster_counts.values,
                "try_rate": cluster_profile.loc[cluster_counts.index, "target_trial"].values * 100,
            }
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    # Heatmap สรุปบุคลิกแต่ละ Cluster
    plt.figure(figsize=(12, 6))
    # เลือกมาเฉพาะ Likert Score เพื่อดูว่าแต่ละกลุ่มให้ความสำคัญกับอะไร
    likert_cols = [c for c in cluster_profile.columns if c.startswith('likert_')]
    if likert_cols:
        sns.heatmap(cluster_profile[likert_cols].T, cmap="YlGnBu", annot=True, fmt=".1f")
        plt.title("Cluster Profile Heatmap (ทัศนคติของลูกค้าในแต่ละกลุ่ม)")
        plt.xlabel("Cluster ID")
        plt.ylabel("ปัจจัยความสนใจ (Features)")
        plt.savefig(OUTPUT_DIR / "cluster_profile_heatmap.png", bbox_inches="tight")
    plt.close()
    
    # 4. Generate Summary
    best_cluster_idx = cluster_profile['target_trial'].idxmax()
    best_rate = cluster_profile.loc[best_cluster_idx, 'target_trial'] * 100
    best_profile = cluster_profile.loc[best_cluster_idx]
    best_likert_cols = [c for c in likert_cols if c in best_profile.index]
    if best_likert_cols:
        top_likert = best_profile[best_likert_cols].sort_values(ascending=False).head(3)
        top_likert_text = "\n".join(
            [f"- **{idx.replace('likert_', '')}:** {val:.2f}/5" for idx, val in top_likert.items()]
        )
    else:
        top_likert_text = "- ไม่มี Likert feature สำหรับสรุป profile"
    
    summary_text = f"""# 🧩 Customer Segmentation Summary

## 1. ผลการแบ่งกลุ่ม (Segmentation Results)
จากการใช้ K-Means Clustering แบ่งกลุ่มลูกค้าออกเป็น **{selected_k} กลุ่ม** พบว่า:
- **กลุ่มที่มีศักยภาพสูงสุด (Best Segment):** คือ **Cluster {best_cluster_idx}**
- **อัตราการอยากลองสินค้าใหม่ (Try Rate):** **{best_rate:.1f}%**

## 2. บุคลิกของกลุ่มที่มีศักยภาพสูงสุด (Cluster {best_cluster_idx} Profile)
Profile นี้อ้างอิงเฉพาะฟีเจอร์ที่ใช้ทำ Clustering เท่านั้น ไม่รวมคำตอบ target, เหตุผลการลอง, media channel หรือ buy channel
{top_likert_text}

## 3. หมายเหตุประกอบการวิเคราะห์ (Academic Notes)
- **การเลือกจำนวนกลุ่ม (K):** เราเลือกใช้ **K=4** แม้ค่า Silhouette Score จะแนะนำค่าอื่น ทั้งนี้เพื่อให้สามารถ **ตีความผลลัพธ์ในเชิงธุรกิจได้ดีที่สุด (Business Interpretability)** และแบ่งกลุ่มลูกค้าได้ชัดเจนเพียงพอต่อการทำแผนการตลาด
- **ฟีเจอร์ที่ใช้:** วิเคราะห์เฉพาะปัจจัยที่เกี่ยวข้องกับพฤติกรรมและทัศนคติต่อกาแฟเท่านั้น (Coffee-Centric Features)
- **การใช้ Target:** target_binary_trial ถูก merge กลับมาหลังทำ Clustering แล้วเท่านั้น เพื่อดู Try Rate ของแต่ละกลุ่ม ไม่ได้ใช้ในการ fit K-Means
"""
    
    with open(OUTPUT_DIR / "cluster_profile_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Export CSV สำหรับตรวจสอบตัวเลขละเอียด
    cluster_profile.to_csv(OUTPUT_DIR / "cluster_profile_details.csv")
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    profile_df.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)
    
    print(f"--- Clustering สำเร็จ บันทึก Profile สรุปกลุ่มลูกค้าใน {OUTPUT_DIR} ---")

if __name__ == "__main__":
    run_clustering()

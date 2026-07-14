import os
import sys
import json
import time
import subprocess

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc as sk_auc

from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# Dam bao driver va worker Spark dung DUNG CUNG 1 ban Python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CURRENT_DIR, "charts")
os.makedirs(OUT_DIR, exist_ok=True)


def show_and_save(fig_path, title_for_console):
    """Luu bieu do vao charts/ để xem qua dashboard web Streamlit."""
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"  -> Da luu: {fig_path}")
    plt.close()


# =====================================================================
# BUOC 1: KHOI TAO SPARK SESSION
# =====================================================================
print("=" * 70)
print("BUOC 1: KHOI TAO SPARK SESSION")
print("=" * 70)

spark = SparkSession.builder \
    .appName("CustomerChurnPrediction") \
    .master("spark://127.0.0.1:7077") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# =====================================================================
# BUOC 2: DOC DU LIEU TU HDFS
# =====================================================================
print("\n" + "=" * 70)
print("BUOC 2: DOC DU LIEU TU HDFS")
print("=" * 70)

file_name = sys.argv[1] if len(sys.argv) > 1 else "telco_churn.csv"
csv_path = f"hdfs://localhost:9000/input/{file_name}"
print(f"Dang doc du lieu tu: {csv_path}")

df = spark.read.csv(csv_path, header=True, inferSchema=True)
df = df.dropna()

print("TONG SO DONG:", df.count())
print("SCHEMA:")
df.printSchema()

# Chuyen sang Pandas de dung cho cac bieu do phan tich du lieu
df_plots = df.toPandas()

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
print("TRAIN COUNT:", train_data.count(), " | TEST COUNT:", test_data.count())

# =====================================================================
# BUOC 3: XAY DUNG PIPELINE TIEN XU LY DU LIEU
# =====================================================================
print("\n" + "=" * 70)
print("BUOC 3: XAY DUNG PIPELINE TIEN XU LY DU LIEU")
print("=" * 70)

NUMERIC_COLS_ALL = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "SatisfactionScore", "SupportCallsLast6Months",
    "Data_Usage_Drop_Rate", "Call_Drop_Rate", "Slow_Speed_Days", "Unresolved_Complaints"
]

CATEGORICAL_COLS_ALL = [
    "Contract", "InternetService", "OnlineSecurity", "TechSupport",
    "PaperlessBilling", "PaymentMethod", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "OnlineBackup", "DeviceProtection",
    "StreamingTV", "StreamingMovies"
]

NUMERIC_COLS = [c for c in NUMERIC_COLS_ALL if c in df.columns]
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS_ALL if c in df.columns]
INDEXED_COLS = [c + "Index" for c in CATEGORICAL_COLS]
VEC_COLS = [c + "Vec" for c in CATEGORICAL_COLS]

print("Cac cot dang so su dung:", NUMERIC_COLS)
print("Cac cot dang chu su dung:", CATEGORICAL_COLS)

indexer = StringIndexer(inputCols=CATEGORICAL_COLS, outputCols=INDEXED_COLS, handleInvalid="keep")
encoder = OneHotEncoder(inputCols=INDEXED_COLS, outputCols=VEC_COLS)
feature_cols = NUMERIC_COLS + VEC_COLS
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)

models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="Churn"),
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="Churn", seed=42),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="Churn", seed=42),
}

# =====================================================================
# BUOC 4: HUAN LUYEN + DANH GIA 3 MO HINH
# =====================================================================
print("\n" + "=" * 70)
print("BUOC 4: HUAN LUYEN VA DANH GIA CAC MO HINH")
print("=" * 70)

results = {}
rf_feature_importance = None
rf_feature_names = []
probs_by_model = {}

for model_name, model_algorithm in models.items():
    print(f"\n--- Dang huan luyen: {model_name} ---")
    pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, model_algorithm])
    model_fit = pipeline.fit(train_data)
    predictions = model_fit.transform(test_data)

    print(f"--- BANG DU DOAN MAU CUA MO HINH: {model_name} ---")
    cols_to_show = [c for c in ["CustomerID", "Contract", "Churn", "prediction"] if c in predictions.columns]
    predictions.select(*cols_to_show).show(5)

    acc_evaluator = MulticlassClassificationEvaluator(labelCol="Churn", predictionCol="prediction", metricName="accuracy")
    auc_evaluator = BinaryClassificationEvaluator(labelCol="Churn", rawPredictionCol="prediction", metricName="areaUnderROC")
    
    acc = acc_evaluator.evaluate(predictions)
    auc_val = auc_evaluator.evaluate(predictions)

    cm = predictions.groupBy("Churn", "prediction").count().collect()
    cm_dict = {(int(r["Churn"]), int(r["prediction"])): r["count"] for r in cm}
    tn = cm_dict.get((0, 0), 0)
    fp = cm_dict.get((0, 1), 0)
    fn = cm_dict.get((1, 0), 0)
    tp = cm_dict.get((1, 1), 0)

    prec = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    results[model_name] = {
        "accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "auc": auc_val,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

    if model_name == "Random Forest":
        rf_model = model_fit.stages[-1]
        rf_feature_importance = rf_model.featureImportances.toArray().tolist()

        rf_feature_names = list(NUMERIC_COLS)
        if CATEGORICAL_COLS:
            fitted_indexer = model_fit.stages[0]
            labels_array = fitted_indexer.labelsArray
            for col, labels in zip(CATEGORICAL_COLS, labels_array):
                for lbl in labels[:-1]:
                    rf_feature_names.append(f"{col}={lbl}")

    prob_col = "probability" if "probability" in predictions.columns else None
    if prob_col:
        pdf_probs = predictions.select(
            "Churn",
            vector_to_array(predictions[prob_col])[1].alias("prob1")
        ).toPandas()
    else:
        pdf_probs = predictions.select("Churn", predictions["prediction"].alias("prob1")).toPandas()
    probs_by_model[model_name] = pdf_probs

    print(f"[{model_name}] Tinh rieng Churn=1: acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} auc={auc_val:.4f}")
    print(f"  Confusion Matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

with open(os.path.join(CURRENT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump({"results": results, "rf_importance": rf_feature_importance,
               "rf_feature_names": rf_feature_names}, f, indent=2, ensure_ascii=False)
print("\nDa luu kết quả tinh toan moi vao results.json")

# =====================================================================
# BUOC 5: TRUC QUAN HOA - HIEN THI LAN LUOT TUNG BIEU DO
# =====================================================================
print("\n" + "=" * 70)
print("BUOC 5: TRUC QUAN HOA DU LIEU VA KET QUA MO HINH")
print("=" * 70)

# --- 5.1 Ty le Churn Rate tong the ---
print("\n[Bieu do 1/10] Ty le Churn Rate tong the")
plt.figure(figsize=(6, 5))
df_plots['Churn'].value_counts().sort_index().plot(
    kind='pie', autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'],
    startangle=90, labels=['O lai (0)', 'Roi di (1)']
)
plt.title('Ty le Churn Rate tong the')
plt.ylabel('')
show_and_save(os.path.join(OUT_DIR, "churn_rate.png"), "Ty le Churn Rate")

# --- 5.2 Ty le Churn theo nhom muc cuoc hang thang (MonthlyCharges) ---
print("\n[Bieu do 2/10] Ty le Churn theo muc cuoc hang thang (MonthlyCharges)")
plt.figure(figsize=(7, 5))
charges_bins = [0, 40, 60, 80, 100, df_plots['MonthlyCharges'].max() + 1]
charges_labels = ['<40$', '40-60$', '60-80$', '80-100$', '>100$']
df_plots['MonthlyChargesGroup'] = pd.cut(df_plots['MonthlyCharges'], bins=charges_bins, labels=charges_labels)
charges_rate = df_plots.groupby('MonthlyChargesGroup', observed=True)['Churn'].mean().reindex(charges_labels) * 100
ax1 = charges_rate.plot(kind='bar', color='#4c72b0')
for i, v in enumerate(charges_rate):
    ax1.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=9)
plt.title('Ty le Churn theo muc cuoc hang thang (MonthlyCharges)')
plt.ylabel('Ty le Churn (%)')
plt.xlabel('Muc cuoc hang thang (USD)')
plt.xticks(rotation=0)
plt.ylim(0, max(float(charges_rate.max()) * 1.2, 10))
show_and_save(os.path.join(OUT_DIR, "churn_by_monthlycharges.png"), "Churn theo muc cuoc")

# --- 5.3 Ty le Churn theo nhom thoi gian gan bo (tenure) ---
print("\n[Bieu do 3/10] Ty le Churn theo thoi gian gan bo (tenure)")
plt.figure(figsize=(7, 5))
tenure_bins = [0, 6, 12, 24, 48, df_plots['tenure'].max() + 1]
tenure_labels = ['0-6 thang', '6-12 thang', '1-2 nam', '2-4 nam', 'Tren 4 nam']
df_plots['TenureGroup'] = pd.cut(df_plots['tenure'], bins=tenure_bins, labels=tenure_labels)
tenure_rate = df_plots.groupby('TenureGroup', observed=True)['Churn'].mean().reindex(tenure_labels) * 100
ax2 = tenure_rate.plot(kind='bar', color='#55a868')
for i, v in enumerate(tenure_rate):
    ax2.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=9)
plt.title('Ty le Churn theo thoi gian gan bo (tenure)')
plt.ylabel('Ty le Churn (%)')
plt.xlabel('Thoi gian gan bo voi dich vu')
plt.xticks(rotation=0)
plt.ylim(0, max(float(tenure_rate.max()) * 1.2, 10))
show_and_save(os.path.join(OUT_DIR, "churn_by_tenure.png"), "Churn theo tenure")

# --- 5.4 Churn rate theo loai hop dong ---
print("\n[Bieu do 4/10] Ty le Churn theo loai hop dong")
plt.figure(figsize=(7, 5))
rate = df_plots.groupby('Contract')['Churn'].mean().sort_values(ascending=False) * 100
rate.plot(kind='bar', color='#ff7f50')
plt.title('Ty le Churn theo loai hop dong')
plt.ylabel('Ty le Churn (%)')
plt.xlabel('Loai hop dong')
plt.xticks(rotation=0)
show_and_save(os.path.join(OUT_DIR, "churn_by_contract.png"), "Churn theo hop dong")

# --- THÊM MỚI 5.5: Churn rate theo Diem hai long (SatisfactionScore) ---
print("\n[Bieu do 5/10] Ty le Churn theo Diem hai long (SatisfactionScore)")
plt.figure(figsize=(7, 5))
sat_rate = df_plots.groupby('SatisfactionScore')['Churn'].mean() * 100
sat_rate.plot(kind='bar', color='#4ca3a3')
plt.title('Ty le Churn theo Diem hai long (SatisfactionScore)')
plt.ylabel('Ty le Churn (%)')
plt.xlabel('Diem hai long (1-5)')
plt.xticks(rotation=0)
show_and_save(os.path.join(OUT_DIR, "churn_by_satisfaction.png"), "Churn theo Diem hai long")

# --- THÊM MỚI 5.6: Churn rate theo Khieu nai chua xu ly (Unresolved_Complaints) ---
print("\n[Bieu do 6/10] Ty le Churn theo Khieu nai chua xu ly (Unresolved_Complaints)")
plt.figure(figsize=(7, 5))
comp_rate = df_plots.groupby('Unresolved_Complaints')['Churn'].mean() * 100
comp_rate.plot(kind='bar', color='#b84a4a')
plt.title('Ty le Churn theo Khieu nai chua xu ly (Unresolved_Complaints)')
plt.ylabel('Ty le Churn (%)')
plt.xlabel('So khieu nai chua xu ly')
plt.xticks(rotation=0)
show_and_save(os.path.join(OUT_DIR, "churn_by_complaints.png"), "Churn theo Khieu nai")

# --- 5.7 Feature Importance (Random Forest) ---
print("\n[Bieu do 7/10] Muc do quan trong dac trung - Random Forest")
if rf_feature_importance and len(rf_feature_names) > 0:
    feat_names = rf_feature_names[:len(rf_feature_importance)]
    plt.figure(figsize=(8, max(5, 0.35 * len(feat_names))))
    order = sorted(zip(feat_names, rf_feature_importance), key=lambda x: x[1])
    names = [o[0] for o in order]
    vals = [o[1] for o in order]
    plt.barh(names, vals, color='#4c72b0')
    plt.title('Muc do quan trong cua dac trung - Random Forest')
    plt.xlabel('Importance')
    show_and_save(os.path.join(OUT_DIR, "feature_importance.png"), "Feature Importance")

# --- 5.8 Confusion Matrices (3 mo hinh) ---
print("\n[Bieu do 8/10] Confusion Matrix cua 3 mo hinh")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, r) in zip(axes, results.items()):
    cm = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
    ax.set_title(name)
plt.suptitle('Confusion Matrix - 3 mo hinh')
show_and_save(os.path.join(OUT_DIR, "confusion_matrices.png"), "Confusion Matrices")

# --- 5.9 ROC Curve (3 mo hinh) ---
print("\n[Bieu do 9/10] ROC Curve so sanh 3 mo hinh")
plt.figure(figsize=(7, 6))
colors = {"Logistic Regression": "#1f77b4", "Decision Tree": "#ff7f0e", "Random Forest": "#2ca02c"}
for name, pdf_probs in probs_by_model.items():
    fpr, tpr, _ = roc_curve(pdf_probs["Churn"], pdf_probs["prob1"])
    roc_auc = sk_auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=colors.get(name))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - So sanh 3 mo hinh')
plt.legend(loc='lower right')
show_and_save(os.path.join(OUT_DIR, "roc_curve.png"), "ROC Curve")

# --- 5.10 So do kien truc he thong ---
print("\n[Bieu do 10/10] So do kien truc he thong")
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 3)
ax.axis('off')

stages = [
    ("1. Thu thap\ndu lieu", f"{file_name}"),
    ("2. Luu tru\nphan tan", "HDFS\n/input/"),
    ("3. Tien xu ly &\nFeature Engineering", "PySpark\nStringIndexer, OneHot,\nAssembler, Scaler\n(Bo sung bien hanh vi)"),
    ("4. Huan luyen &\ndanh gia mo hinh", "Spark MLlib\nLR / DT / RF"),
    ("5. Truc quan hoa\n& bao cao", "Pandas +\nMatplotlib/Seaborn"),
]

stage_colors = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6"]
box_w, box_h = 1.85, 1.7
gap = 0.35
x = 0.15
for i, (title, sub) in enumerate(stages):
    box = FancyBboxPatch((x, 0.65), box_w, box_h, boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.3, edgecolor="#1e3a8a", facecolor=stage_colors[i])
    ax.add_patch(box)
    ax.text(x + box_w / 2, 0.65 + box_h - 0.35, title, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color="#1e3a8a")
    ax.text(x + box_w / 2, 0.65 + 0.45, sub, ha='center', va='center', fontsize=8, color="#1e3a8a")
    if i < len(stages) - 1:
        arrow = FancyArrowPatch((x + box_w + 0.03, 0.65 + box_h / 2), (x + box_w + gap - 0.03, 0.65 + box_h / 2),
                                 arrowstyle='-|>', mutation_scale=18, color="#1e3a8a", linewidth=1.5)
        ax.add_patch(arrow)
    x += box_w + gap
show_and_save(os.path.join(OUT_DIR, "architecture.png"), "So do kien truc he thong")

# =====================================================================
# BUOC 6: TONG KET
# =====================================================================
print("\n" + "=" * 70)
print("BUOC 6: TONG KET KET QUA CAC MO HINH (CHURN=1)")
print("=" * 70)
for name, r in results.items():
    print(f"{name:22s} | Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} | "
          f"Prec={r['precision']:.4f} | Rec={r['recall']:.4f} | AUC={r['auc']:.4f}")

print(f"\nTat ca bieu do da duoc luu trong thu muc: {OUT_DIR}")
print("Tam dung 10 giay de ban kip xem Spark Web UI (neu can)...")
time.sleep(10)

spark.stop()
print("\nDA HOAN THANH TOAN BO PIPELINE.")

# =====================================================================
# BUOC 7: MO DASHBOARD WEB (STREAMLIT) DE XEM KET QUA
# =====================================================================
print("\n" + "=" * 70)
print("BUOC 7: DANG MO DASHBOARD TREN TRINH DUYET (Streamlit)")
print("=" * 70)

dashboard_path = os.path.join(CURRENT_DIR, "dashboard.py")
if os.path.exists(dashboard_path):
    print("Dang khoi chay Streamlit dashboard tai http://localhost:8501 ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
else:
    print(f"Khong tim thay {dashboard_path}. Hay chay thu cong: streamlit run dashboard.py")
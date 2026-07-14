"""
Dashboard hien thi ket qua Customer Churn Prediction.
Chay bang lenh:  streamlit run dashboard.py
(Script churn_prediction.py se tu dong goi lenh nay sau khi chay xong pipeline)
"""

import os
import json
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CURRENT_DIR, "charts")
RESULTS_PATH = os.path.join(CURRENT_DIR, "results.json")

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* Thu hẹp khung chứa nội dung */
    .block-container {
        max-width: 500px !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    /* Giảm cỡ chữ toàn bộ hệ thống */
    html, body, [data-testid="stAppViewContainer"] {
        font-size: 13px !important;
    }
    /* Thu nhỏ các tiêu đề */
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.0rem !important; }
    
    /* Thu nhỏ font chữ trong bảng biểu số liệu */
    .stDataFrame div, table {
        font-size: 12px !important;
    }
    /* Thu nhỏ khối thẻ chỉ số Metric */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 HỆ THỐNG DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ")
st.caption("Kết quả huấn luyện & phân tích dữ liệu — PySpark MLlib")

if not os.path.exists(RESULTS_PATH):
    st.error(
        "Không tìm thấy results.json. Vui lòng chạy churn_prediction.py trước "
        "để pipeline tạo ra kết quả và biểu đồ."
    )
    st.stop()

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", {})

# ---------------------------------------------------------------
# 1. BANG SO SANH CAC MO HINH (METRIC CARDS + TABLE)
# ---------------------------------------------------------------
st.header("1️⃣ So sánh hiệu suất các mô hình")

if results:
    best_model = max(results.items(), key=lambda kv: kv[1]["auc"])[0]
    cols = st.columns(len(results))
    for col, (name, r) in zip(cols, results.items()):
        with col:
            label = f"🏆 {name}" if name == best_model else name
            st.metric(label=label, value=f"AUC {r['auc']:.3f}",
                       delta=f"Acc {r['accuracy']:.3f}")

    st.subheader("Bảng chi tiết")
    table_rows = []
    for name, r in results.items():
        table_rows.append({
            "Mô hình": name,
            "Accuracy": round(r["accuracy"], 4),
            "F1-score": round(r["f1"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "AUC": round(r["auc"], 4),
            "TP": r["tp"], "TN": r["tn"], "FP": r["fp"], "FN": r["fn"],
        })
    st.dataframe(table_rows, use_container_width=True, hide_index=True)
else:
    st.warning("Chưa có dữ liệu kết quả mô hình.")

st.divider()

# ---------------------------------------------------------------
# 2. PHAN TICH DU LIEU (EDA)
# ---------------------------------------------------------------
st.header("2️⃣ Phân tích dữ liệu khách hàng")

# CẬP NHẬT: Thêm 2 biểu đồ mới vào mảng
eda_charts = [
    ("churn_rate.png", "Tỷ lệ Churn tổng thể"),
    ("churn_by_monthlycharges.png", "Tỷ lệ Churn theo mức cước hàng tháng"),
    ("churn_by_tenure.png", "Tỷ lệ Churn theo thời gian gắn bó"),
    ("churn_by_contract.png", "Tỷ lệ Churn theo loại hợp đồng"),
    ("churn_by_satisfaction.png", "Tỷ lệ Churn theo Điểm hài lòng"),
    ("churn_by_complaints.png", "Tỷ lệ Churn theo Khiếu nại chưa xử lý"),
]

# CẬP NHẬT: Tăng thêm 1 hàng (row3) để hiển thị đủ 6 biểu đồ EDA
row1 = st.columns(2)
row2 = st.columns(2)
row3 = st.columns(2)
for (fname, caption), col in zip(eda_charts, row1 + row2 + row3):
    path = os.path.join(OUT_DIR, fname)
    with col:
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.info(f"Chưa có biểu đồ: {fname}")

st.divider()

# ---------------------------------------------------------------
# 3. KET QUA MO HINH (Feature Importance, Confusion Matrix, ROC)
# ---------------------------------------------------------------
st.header("3️⃣ Kết quả mô hình")

fi_path = os.path.join(OUT_DIR, "feature_importance.png")
if os.path.exists(fi_path):
    st.subheader("Mức độ quan trọng của đặc trưng (Random Forest)")
    st.image(fi_path, use_container_width=True)

cm_path = os.path.join(OUT_DIR, "confusion_matrices.png")
if os.path.exists(cm_path):
    st.subheader("Confusion Matrix — 3 mô hình")
    st.image(cm_path, use_container_width=True)

roc_path = os.path.join(OUT_DIR, "roc_curve.png")
if os.path.exists(roc_path):
    st.subheader("ROC Curve — So sánh 3 mô hình")
    st.image(roc_path, use_container_width=True)

st.divider()

# ---------------------------------------------------------------
# 4. KIEN TRUC HE THONG
# ---------------------------------------------------------------
arch_path = os.path.join(OUT_DIR, "architecture.png")
if os.path.exists(arch_path):
    st.header("4️⃣ Kiến trúc hệ thống")
    st.image(arch_path, use_container_width=True)

st.caption("Dashboard tự động cập nhật mỗi khi chạy lại churn_prediction.py")
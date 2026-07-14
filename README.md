Yêu cầu hệ thống
Phần cứng

Thành phần     Tối thiểu     Khuyến nghị
RAM              8GB             16GB
CPU             4 nhân          8 nhân
Ổ cứng        10 GB trống     20 GB trống

Phần mềm

Công nghệ        Phiên bản         Ghi chú
Java (JDK)      11 hoặc 17       Bắt buộc cho Hadoop & Spark
Python           3.9 +           Khuyến nghị 3.10 hoặc 3.11
Apache Hadoop    3.3.x            HDFS để lưu trữ dữ liệu
Apache Spark    3.4.x hoặc 3.5.x   Pre-built for Hadoop 3.x
pip packages    Xem requirements.txt   pyspark, streamlit, ...

Hướng dẫn cài đặt và chạy chi tiết

BƯỚC 0 — Clone repository về máy

bashgit clone https://github.com/<your-username>/churn-prediction-bigdata.git
cd churn-prediction-bigdata
BƯỚC 1 — Cài đặt Java 11

bash# Ubuntu / Debian
sudo apt update
sudo apt install -y openjdk-11-jdk

# Kiểm tra cài đặt thành công
java -version
# Kết quả mong đợi: openjdk version "11.x.x"

# Thêm vào ~/.bashrc (hoặc ~/.zshrc)
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc


BƯỚC 2 — Cài đặt Apache Hadoop 3.3.x

bash# Tải Hadoop
wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
tar -xzf hadoop-3.3.6.tar.gz
sudo mv hadoop-3.3.6 /opt/hadoop

# Thêm vào ~/.bashrc
echo 'export HADOOP_HOME=/opt/hadoop'         >> ~/.bashrc
echo 'export PATH=$PATH:$HADOOP_HOME/bin'     >> ~/.bashrc
echo 'export PATH=$PATH:$HADOOP_HOME/sbin'    >> ~/.bashrc
echo 'export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop' >> ~/.bashrc
source ~/.bashrc

# Kiểm tra
hadoop version

Cấu hình HDFS — Sửa file /opt/hadoop/etc/hadoop/core-site.xml:

xml<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>

Sửa file /opt/hadoop/etc/hadoop/hdfs-site.xml:

xml<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
</configuration>

bash# Khởi tạo HDFS (chỉ chạy lần đầu)
hdfs namenode -format

# Khởi động HDFS
start-dfs.sh

# Kiểm tra — phải thấy NameNode, DataNode, SecondaryNameNode
jps

# Mở Web UI kiểm tra: http://localhost:9870


BƯỚC 3 — Cài đặt Apache Spark 3.4.x

bash# Tải Spark (pre-built cho Hadoop 3.x)
wget https://archive.apache.org/dist/spark/spark-3.4.3/spark-3.4.3-bin-hadoop3.tgz
tar -xzf spark-3.4.3-bin-hadoop3.tgz
sudo mv spark-3.4.3-bin-hadoop3 /opt/spark

# Thêm vào ~/.bashrc
echo 'export SPARK_HOME=/opt/spark'           >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin'      >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/sbin'     >> ~/.bashrc
echo 'export PYSPARK_PYTHON=python3'          >> ~/.bashrc
source ~/.bashrc

# Khởi động Spark cluster (standalone mode)
$SPARK_HOME/sbin/start-all.sh

# Kiểm tra — phải thấy Master, Worker
jps

# Web UI Spark Master: http://localhost:8080
# Web UI Spark App:    http://localhost:4040  (khi đang chạy job)


⚠️ Lưu ý: Địa chỉ Spark Master mặc định trong code là spark://127.0.0.1:7077.
Nếu máy bạn dùng IP khác, hãy sửa dòng .master(...) trong churn_prediction.py.




BƯỚC 4 — Cài đặt thư viện Python

bash# Tạo môi trường ảo (khuyến nghị)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Cài đặt tất cả thư viện
pip install -r requirements.txt

# Hoặc cài thủ công
pip install pyspark==3.4.3 \
            streamlit \
            pandas \
            matplotlib \
            seaborn \
            scikit-learn

Nội dung requirements.txt:

pyspark==3.4.3
streamlit>=1.32.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0


BƯỚC 5 — Sinh dữ liệu (nếu chưa có file CSV)

bash# Sinh 3 bộ dữ liệu (6.000 bản ghi tổng)
python3 create_data.py

# Kết quả: tạo ra các file
# telco_churn_1.csv  (2000 bản ghi, seed=42)
# telco_churn_2.csv  (2000 bản ghi, seed=43)
# telco_churn_3.csv  (2000 bản ghi, seed=45)


📌 Mỗi file có 26 cột đặc trưng: CustomerID, tenure, Contract, MonthlyCharges,
TotalCharges, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
StreamingTV, StreamingMovies, PaperlessBilling, PaymentMethod,
SatisfactionScore, SupportCallsLast6Months, Data_Usage_Drop_Rate,
Call_Drop_Rate, Slow_Speed_Days, Unresolved_Complaints, Churn (nhãn)




BƯỚC 6 — Đưa dữ liệu lên HDFS

bash# Tạo thư mục /input trên HDFS
hdfs dfs -mkdir -p /input

# Upload file CSV lên HDFS
hdfs dfs -put telco_churn_1.csv /input/
hdfs dfs -put telco_churn_2.csv /input/
hdfs dfs -put telco_churn_3.csv /input/

# Kiểm tra đã upload thành công
hdfs dfs -ls /input/
# Kết quả mong đợi:
# -rw-r--r--   1 ... /input/telco_churn_1.csv
# -rw-r--r--   1 ... /input/telco_churn_2.csv
# -rw-r--r--   1 ... /input/telco_churn_3.csv

# Kiểm tra dung lượng
hdfs dfs -du -h /input/


BƯỚC 7 — Chạy pipeline Spark ML chính

bash# Cú pháp:
# spark-submit churn_prediction.py <tên_file_csv>

# Chạy với telco_churn_1.csv (khuyến nghị, seed=42 → kết quả ổn định nhất)
spark-submit churn_prediction.py telco_churn_1.csv

# Chạy với telco_churn_2.csv
spark-submit churn_prediction.py telco_churn_2.csv

# Chạy với telco_churn_3.csv
spark-submit churn_prediction.py telco_churn_3.csv
Quá trình chạy sẽ in ra màn hình:

══════════════════════════════════════════════════════════════════════
BUOC 1: KHOI TAO SPARK SESSION
══════════════════════════════════════════════════════════════════════
BUOC 2: DOC DU LIEU TU HDFS
TONG SO DONG: 2000
BUOC 3: XAY DUNG PIPELINE TIEN XU LY DU LIEU
...
BUOC 4: HUAN LUYEN VA DANH GIA CAC MO HINH
  [Logistic Regression] acc=0.8631  f1=0.7860  auc=0.8328
  [Decision Tree]       acc=0.7765  f1=0.6522  auc=0.7375
  [Random Forest]       acc=0.8184  f1=0.7162  auc=0.7831
...
BUOC 5: TRUC QUAN HOA (10 bieu do da luu vao charts/)
Da luu results.json
HOAN THANH.

Output sau khi chạy xong:

charts/
  ├── churn_rate.png              ← Tỷ lệ Churn tổng thể
  ├── churn_by_monthlycharges.png ← Churn theo mức cước
  ├── churn_by_tenure.png         ← Churn theo thời gian gắn bó
  ├── churn_by_contract.png       ← Churn theo loại hợp đồng
  ├── churn_by_satisfaction.png   ← Churn theo điểm hài lòng
  ├── churn_by_complaints.png     ← Churn theo khiếu nại chưa xử lý
  ├── feature_importance.png      ← Tầm quan trọng đặc trưng (RF)
  ├── confusion_matrices.png      ← Ma trận nhầm lẫn (3 mô hình)
  ├── roc_curve.png               ← Đường cong ROC
  └── architecture.png            ← Sơ đồ kiến trúc hệ thống

results.json                      ← Metrics + feature names (JSON)


BƯỚC 8 — Mở Dashboard Streamlit


Mở terminal mới (giữ nguyên terminal đang chạy Spark ở trên)



bash# Chuyển vào thư mục dự án
cd churn-prediction-bigdata

# Khởi động dashboard
streamlit run dashboard.py

# Kết quả:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
#   Network URL: http://192.168.x.x:8501

Mở trình duyệt và truy cập: 👉 http://localhost:8501

Dashboard hiển thị 4 mục:


① So sánh hiệu suất mô hình — Metric cards + bảng chi tiết Accuracy/F1/AUC/Confusion Matrix
② Phân tích dữ liệu (EDA) — 6 biểu đồ phân tích đặc trưng
③ Kết quả mô hình — Feature Importance, Confusion Matrix, ROC Curve
④ Kiến trúc hệ thống — Sơ đồ 5 giai đoạn xử lý



💡 Dashboard tự động cập nhật mỗi khi chạy lại churn_prediction.py
vì đọc trực tiếp từ results.json và thư mục charts/
Quy trình chạy lại với bộ dữ liệu khác

bash# Ví dụ: chạy lại với telco_churn_2.csv
hdfs dfs -put -f telco_churn_2.csv /input/
spark-submit churn_prediction.py telco_churn_2.csv
# → Dashboard tự cập nhật tại localhost:8501


🐛 Xử lý lỗi thường gặp

❌ Lỗi: Connection refused khi kết nối Spark Master

bash# Kiểm tra Spark đang chạy chưa
jps
# Nếu không thấy Master/Worker → khởi động lại
$SPARK_HOME/sbin/start-all.sh

❌ Lỗi: No such file or directory khi đọc HDFS

bash# Kiểm tra HDFS đang chạy
hdfs dfsadmin -report

# Nếu không chạy → khởi động lại
start-dfs.sh

# Kiểm tra file đã có trên HDFS chưa
hdfs dfs -ls /input/

❌ Lỗi: Python version mismatch giữa Driver và Worker

bash# Đã được xử lý trong code bằng 2 dòng sau (đầu churn_prediction.py):
# os.environ["PYSPARK_PYTHON"] = sys.executable
# os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Nếu vẫn lỗi, thêm vào ~/.bashrc:
echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc
echo 'export PYSPARK_DRIVER_PYTHON=python3' >> ~/.bashrc
source ~/.bashrc

❌ Lỗi: Out of Memory (OOM)

bash# Tăng bộ nhớ khi submit
spark-submit \
  --driver-memory 4g \
  --executor-memory 4g \
  churn_prediction.py telco_churn_1.csv

❌ Lỗi: streamlit: command not found

bashpip install streamlit
# hoặc nếu đang dùng venv:
source venv/bin/activate
pip install streamlit

❌ Lỗi: randomSplit không đúng 80/20

Spark randomSplit() dùng xấp xỉ Bernoulli — kết quả thực tế có thể lệch
nhẹ so với 80/20. Đây là đặc tính bình thường của Spark, không phải lỗi.
Dùng seed=42 để đảm bảo tái lập kết quả.
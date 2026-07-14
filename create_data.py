import csv
import math
import random


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def tao_file_csv_ngau_nhien(filename, start_id, num_rows=2000, seed=None):
    if seed is not None:
        random.seed(seed)

    contracts = ['Month-to-month', 'One year', 'Two year']
    internet_services = ['DSL', 'Fiber optic', 'No']
    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']

    # Baseline shift ngẫu nhiên nhỏ để tạo độ khác biệt giữa các tập dữ liệu
    baseline_shift = random.uniform(-0.1, 0.1)

    W_CONTRACT = {'Month-to-month': 1.10, 'One year': -0.10, 'Two year': -1.15}
    W_INTERNET = {'Fiber optic': 0.45, 'DSL': -0.05, 'No': -0.40}
    W_PAYMENT = {'Electronic check': 0.40, 'Mailed check': 0.0,
                 'Bank transfer': -0.15, 'Credit card': -0.20}

    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        # Thêm các cột đặc trưng mới vào tiêu đề CSV
        writer.writerow([
            'CustomerID', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges',
            'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
            'PaymentMethod', 'SatisfactionScore', 'SupportCallsLast6Months',
            'Data_Usage_Drop_Rate', 'Call_Drop_Rate', 'Slow_Speed_Days', 'Unresolved_Complaints',
            'Churn'
        ])

        for i in range(num_rows):
            cust_id = f"CUST{start_id + i}"
            
            # 1. Tạo phân phối tenure thực tế hơn (phân phối đều)
            tenure = random.randint(1, 72)
            
            contract = random.choice(contracts)
            internet_service = random.choice(internet_services)
            payment_method = random.choice(payment_methods)

            senior_citizen = 1 if random.random() < 0.16 else 0
            partner = 'Yes' if random.random() < 0.48 else 'No'
            dependents = 'Yes' if random.random() < 0.30 else 'No'
            phone_service = 'Yes' if random.random() < 0.90 else 'No'
            
            multiple_lines = 'No phone service' if phone_service == 'No' else ('Yes' if random.random() < 0.42 else 'No')
            online_security = 'No internet service' if internet_service == 'No' else ('Yes' if random.random() < 0.28 else 'No')
            online_backup = 'No internet service' if internet_service == 'No' else ('Yes' if random.random() < 0.34 else 'No')
            device_protection = 'No internet service' if internet_service == 'No' else ('Yes' if random.random() < 0.34 else 'No')
            tech_support = 'No internet service' if internet_service == 'No' else ('Yes' if random.random() < 0.29 else 'No')
            streaming_tv = 'No internet service' if internet_service == 'No' else ('Yes' if random.random() < 0.38 else 'No')
            streaming_movies = 'No internet service' if internet_service == 'No' else ('Yes' if random.random() < 0.39 else 'No')
            
            paperless_billing = 'Yes' if random.random() < 0.59 else 'No'

            # Tính toán chi phí Monthly & Total Charges logic
            base_charge = 20.0
            if internet_service == 'DSL': base_charge += 35.0
            if internet_service == 'Fiber optic': base_charge += 55.0
            if phone_service == 'Yes': base_charge += 15.0
            if multiple_lines == 'Yes': base_charge += 10.0
            
            monthly_charges = round(base_charge + random.uniform(-5, 15), 2)
            total_charges = round(monthly_charges * tenure * random.uniform(0.85, 1.05), 2)

            # Các biến nền tảng hành vi
            satisfaction_score = random.choices([1, 2, 3, 4, 5], weights=[0.12, 0.18, 0.35, 0.25, 0.1])[0]
            support_calls = random.choices([0, 1, 2, 3, 4, 5], weights=[0.25, 0.3, 0.2, 0.13, 0.08, 0.04])[0]

            # --- MÔ PHỎNG 4 ĐẶC TRƯNG HÀNH VI XU HƯỚNG SÁT THỰC TẾ ---
            # 1. Tỷ lệ sụt giảm sử dụng Data
            if satisfaction_score <= 2 or support_calls >= 3:
                data_usage_drop_rate = round(random.uniform(0.40, 0.85), 2)
            else:
                data_usage_drop_rate = round(random.uniform(0.0, 0.39), 2)
            
            # 2. Tỷ lệ cuộc gọi bị rớt
            if internet_service == 'Fiber optic' and random.random() > 0.6:
                call_drop_rate = round(random.uniform(0.05, 0.16), 3)
            else:
                call_drop_rate = round(random.uniform(0.0, 0.04), 3)

            # 3. Số ngày xảy ra hiện tượng mạng yếu trong tháng
            if satisfaction_score <= 2:
                slow_speed_days = random.randint(6, 24)
            else:
                slow_speed_days = random.randint(0, 5)

            # 4. Số khiếu nại nghiêm trọng chưa giải quyết dứt điểm
            if support_calls >= 3 and random.random() > 0.4:
                unresolved_complaints = random.randint(1, 3)
            else:
                unresolved_complaints = 0

            # --- TÍNH TOÁN ĐIỂM SỐ LOGIT RỜI BỎ (CHURN) ---
            # Khởi tạo điểm số logit cơ bản
            score = baseline_shift + W_CONTRACT[contract] + W_INTERNET[internet_service] + W_PAYMENT[payment_method]
            
            if senior_citizen == 1: score += 0.20
            if partner == 'Yes': score -= 0.15
            if dependents == 'Yes': score -= 0.20
            if paperless_billing == 'Yes': score += 0.10

            score += 0.45 * support_calls
            score -= 0.70 * (satisfaction_score - 3)

            # Cộng hưởng các biến hành vi tương tác động
            score += 1.6 * data_usage_drop_rate
            score += 2.2 * call_drop_rate
            score += 0.07 * slow_speed_days
            score += 0.65 * unresolved_complaints

            # =================================================================
            # SỬA ĐỔI ĐỘT PHÁ ĐỂ ĐẢM BẢO QUY LUẬT NGHIỆP VỤ THỰC TẾ TRONG BIỂU ĐỒ
            # =================================================================
            
            # 1. RÀNG BUỘC TENURE (Cực kỳ quan trọng):
            # Khách hàng thâm niên càng lâu thì càng trung thành (giảm điểm logit Churn mạnh mẽ)
            # Tại tenure = 72 tháng, hệ số này trừ đi 3.6 điểm logit, giảm tỉ lệ Churn về sát 0%
            score -= 0.05 * tenure

            # 2. RÀNG BUỘC MONTHLY CHARGES (Cước phí tháng):
            # Khách hàng chịu mức cước cao sẽ tăng nhẹ nguy cơ rời mạng (đặc biệt là phân khúc cước cao)
            score += 0.008 * (monthly_charges - 60)

            # 3. HẠ BASELINE CHURN CHUNG (Tối ưu hóa tỉ lệ nhãn):
            # Trừ bớt 1.1 để đưa tỷ lệ rời mạng chung của cả tập dữ liệu về mức thực tế (~23% - 27%)
            # thay vì mức cân bằng cơ học 50% như trước, giúp các giỏ cước phí không bị đẩy quá cao.
            score -= 1.1

            # Thêm phương sai nhiễu ngẫu nhiên Gauss để bảo toàn tính chân thực
            score += random.gauss(0, 0.3)

            prob_churn = sigmoid(score)
            churn = 1 if random.random() < prob_churn else 0

            writer.writerow([
                cust_id, tenure, contract, monthly_charges, total_charges,
                senior_citizen, partner, dependents, phone_service, multiple_lines,
                internet_service, online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies, paperless_billing,
                payment_method, satisfaction_score, support_calls,
                data_usage_drop_rate, call_drop_rate, slow_speed_days, unresolved_complaints,
                churn
            ])


if __name__ == '__main__':
    # Sinh dữ liệu ngẫu nhiên có kiểm soát hạt giống (Seed) để tạo sự đồng bộ
    tao_file_csv_ngau_nhien('telco_churn.csv', start_id=1001, num_rows=2000, seed=42)
    print("Xử lý hoàn tất! Đã cập nhật thành công thuật toán sinh dữ liệu viễn thông thực tế.")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("mock_transaction.csv")  # 또는 df = ... (직접 생성한 DataFrame 사용)

# 📌 1. 기본 정보 확인
print(df.info())
print(df.describe())
print(df['isFraud'].value_counts(normalize=True))

# 📌 2. 단변수 분포 시각화
sns.histplot(data=df, x="hour", hue="isFraud", multiple="stack", bins=24)
plt.title("시간대별 거래수")
plt.show()

sns.kdeplot(data=df, x="amount_ratio_to_bank_avg", hue="isFraud", common_norm=False)
plt.title("은행 평균 대비 거래금액 비율")
plt.show()

sns.boxplot(data=df, x="isFraud", y="recent_transaction_gap")
plt.title("최근 거래 간격 분포")
plt.show()

# 📌 3. 범주형 변수 분포
sns.countplot(data=df, x="payment_method", hue="isFraud")
plt.title("지불수단 vs 사기 여부")
plt.show()

sns.countplot(data=df, x="authentication", hue="isFraud")
plt.title("인증방식 vs 사기 여부")
plt.show()

# 📌 4. 보안 관련 변수
df["risk_score"] = df["vpn"].astype(int) + df["rooting"].astype(int)
sns.boxplot(data=df, x="isFraud", y="risk_score")
plt.title("보안 리스크 스코어 vs 사기 여부")
plt.show()

# 📌 5. 상관관계 (수치형)
num_cols = df.select_dtypes(include=["float", "int"]).columns
corr = df[num_cols].corr()
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("수치형 변수 간 상관관계")
plt.show()

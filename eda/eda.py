# import pandas as pd

# df = pd.read_csv("/root/dacon-action-decision/submit/model/roberta_eda_state4/specialist_boosters/val_predictions_with_specialists.csv")

# # list_directory인데 틀린 것들
# wrong = df[(df["true_action"] == "list_directory") & 
#            (df["specialist_pred_action"] != "list_directory")]

# print(wrong["specialist_pred_action"].value_counts())
# print(wrong["current_prompt"].head(20).tolist())

import pandas as pd

df = pd.read_csv("/root/dacon-action-decision/submit/model/roberta_eda_state4/specialist_boosters/val_predictions_with_specialists.csv")

# ------step1: list_directory 정답인 것들의 특징------------
ld = df[df["true_action"] == "list_directory"]
rf = df[df["true_action"] == "read_file"]

print("=== list_directory 정답 문장들 (맞춘 것) ===")
ld_correct = ld[ld["specialist_pred_action"] == "list_directory"]
print(ld_correct["current_prompt"].head(15).tolist())

print("\n=== read_file 정답 문장들 ===")
print(rf["current_prompt"].head(15).tolist())

# ------step2:구분 키워드 분석---------
from collections import Counter
import re

def get_words(texts):
    words = []
    for t in texts:
        words.extend(re.findall(r'\w+', str(t).lower()))
    return Counter(words)

ld_words = get_words(ld["current_prompt"])
rf_words = get_words(rf["current_prompt"])

# list_directory에만 유독 많이 나오는 단어 찾기
print("=== list_directory 특징 단어 ===")
for word, cnt in ld_words.most_common(30):
    rf_cnt = rf_words.get(word, 0)
    # list_directory에 상대적으로 많은 단어
    if cnt >= 5 and cnt > rf_cnt * 1.5:
        print(f"{word}: ld={cnt}, rf={rf_cnt}")

# step3: turn_index 차이 확인 (EDA 활용)
# list_directory와 read_file의 turn_index 분포 비교
print("list_directory turn_index 평균:", ld["turn_index"].mean())
print("read_file turn_index 평균:", rf["turn_index"].mean())

# 첫 턴 비율
print("list_directory 첫턴(turn=1) 비율:", (ld["turn_index"]==1).mean())
print("read_file 첫턴(turn=1) 비율:", (rf["turn_index"]==1).mean())

#-------step4: 정량확인-----
import pandas as pd

df = pd.read_csv("/root/dacon-action-decision/submit/model/roberta_eda_state4/specialist_boosters/val_predictions_with_specialists.csv")

ld = df[df["true_action"] == "list_directory"]["current_prompt"].str.lower()
rf = df[df["true_action"] == "read_file"]["current_prompt"].str.lower()

# list_directory 신호 단어들
signals = ["폴더", "디렉토리", "dir", "뭐뭐", "뭐가", "뭐 있", "어디", "구조", "list", "resources"]

print("단어 | list_dir 비율 | read_file 비율")
for w in signals:
    ld_ratio = ld.str.contains(w).mean()
    rf_ratio = rf.str.contains(w).mean()
    print(f"{w}: {ld_ratio:.1%} vs {rf_ratio:.1%}")
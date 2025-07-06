import numpy as np
import pandas as pd
import json

# Đọc pickle
with open("D:/codeSmell/JSS-EnseSmells/EnseSmells/program/data/gotClass/GodClass_TokenIndexing_metrics.pkl", "rb") as file:
    df = pd.read_pickle(file)

# Đọc JSON
with open('D:/codeSmell/JSS-EnseSmells/EnseSmells/program/data/gotClass/graph_godClass_dictName.json', 'r') as f:
    raw_data = json.load(f)
    sample_data_dict = {int(k): v for k, v in raw_data.items()}

# Gán data từ JSON vào DataFrame
df['data'] = df['sample_id'].map(sample_data_dict)

# ⚠️ Kiểm tra dòng không khớp (dữ liệu không trùng)
missing_rows = df[df['data'].isna()]
print(f"Số dòng không tìm thấy graph_info: {len(missing_rows)}")

# Nếu muốn in chi tiết sample_id không trùng
if not missing_rows.empty:
    print("Các sample_id không có trong JSON:")
    print(missing_rows['sample_id'].tolist())
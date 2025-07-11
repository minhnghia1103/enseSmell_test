import numpy as np
import pandas as pd
import json

# Đọc pickle
with open("D:/codeSmell/JSS-EnseSmells/EnseSmells/program/data/gotClass/GodClass_TokenIndexing_metrics.pkl", "rb") as file:
    df = pd.read_pickle(file)
print(f"Số lượng dòng trong DataFrame: {len(df)}")

print("22222222222: ",df['embedding'][:10])

# # Đọc JSON
# with open('D:/codeSmell/JSS-EnseSmells/EnseSmells/program/data/feature_envy/featureEnvy_dict_graph.json', 'r') as f:
#     raw_data = json.load(f)
#     sample_data_dict = {int(k): v for k, v in raw_data.items()}
# print(f"Số lượng sample_id trong JSON: {len(sample_data_dict)}")

# # Gán data từ JSON vào DataFrame
# df['data'] = df['sample_id'].map(sample_data_dict)

# # ⚠️ Kiểm tra dòng không khớp (dữ liệu không trùng)
# missing_rows = df[df['data'].isna()]
# print(f"Số dòng không tìm thấy graph_info: {len(missing_rows)}")

# # Nếu muốn in chi tiết sample_id không trùng
# if not missing_rows.empty:
#     print("Các sample_id không có trong JSON:")
#     print(missing_rows['sample_id'].tolist())
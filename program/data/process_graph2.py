import numpy as np
import pandas as pd
import json
with open("D:/codeSmell/JSS-EnseSmells/EnseSmells/program/data/gotClass/GodClass_TokenIndexing_metrics.pkl", "rb") as file:
    df = pd.read_pickle(file)
# Đọc file JSON chứa dict {sample_id: data_value}
with open('D:/codeSmell/JSS-EnseSmells/EnseSmells/program/data/gotClass/graph_godClass_dictName.json', 'r') as f:
    raw_data = json.load(f)
    sample_data_dict = {int(k): v for k, v in raw_data.items()}

# Đảm bảo sample_id trong df là string để khớp với key trong JSON

# map lại
df['data'] = df['sample_id'].map(sample_data_dict)

# Kiểm tra kết quả
print(df.head())
import os
import json

# Đọc dữ liệu từ file JSON
with open('D:\\codeSmell\\JSS-EnseSmells\\EnseSmells\\program\\data\\dataclass\\graph.json', 'r') as f:
    data = json.load(f)

# Tạo một dictionary mới để lưu các key mới là tên file
new_data = {}
for key, value in data.items():
    # Lấy tên file từ đường dẫn sử dụng hàm os.path.basename
    filename = os.path.basename(key).split('.')[0]
    new_data[filename] = value

# Ghi dữ liệu mới vào file JSON
with open('graph_godClass_dictName.json', 'w') as f:
    json.dump(new_data, f, indent=2)

print("Chuyển đổi hoàn tất!")
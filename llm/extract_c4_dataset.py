import json
import random

# 加载本地数据集
with open('../data/MedChatZH_train.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 构造方法：instruction + output => text
constructed_data = [{'text': f"{item['instruction']} {item['output']}"} for item in data]

# 随机采样128个样本
sampled_data = random.sample(constructed_data, 128)

# 保存校准数据集
with open('../data/c4_MedChatZH.json', 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=4)

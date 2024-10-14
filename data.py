import pandas as pd

# Đọc dữ liệu từ file
# with open('training_data.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         text, intent = line.strip().split('|')  # Tách text và intent
#         training_data.append({"text": text, "intent": intent})
# asw = []
# with open('answer_data.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         intent, answer = line.strip().split('|')  # Tách text và intent
#         asw.append({"intent": intent, "answer": answer})

# df = pd.DataFrame(asw)
# df.to_excel('answer_data.xlsx', index=False)

def get_training_data():
    training_data = []
    df = pd.read_excel('training_data.xlsx')
    data_array = df.values.tolist()
    for item in data_array:
        text = item[0]
        intent = item[1]
        training_data.append({"text": text, "intent": intent})
    return training_data

def get_answer_data():
    answer_data = {}
    df = pd.read_excel('answer_data.xlsx')
    data_array = df.values.tolist()
    for item in data_array:
        intent = item[0] 
        answer = item[1]
        answer_data[intent] = answer
    return answer_data

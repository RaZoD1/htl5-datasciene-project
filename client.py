import requests
import csv 

BASE_URL="http://localhost:8000"

# Define a function to read N lines from the CSV and store them as dictionaries
def read_csv_to_dict(file_path, num_lines):
    data_list = []
    
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        
        # Read only the first 'num_lines' lines
        for i, row in enumerate(reader):
            if i >= num_lines:
                break
            data_list.append(row)
    
    return data_list

data = read_csv_to_dict("Train_data.csv", 10)

# Categorize into anomaly_samples and normal_samples based on the 'class' field
anomaly_samples = []
normal_samples = []

for row in data:
    if row.get('class') == 'normal':
        row.pop('class')
        normal_samples.append(row)
    elif row.get('class') == 'anomaly':
        row.pop('class')
        anomaly_samples.append(row)


def predict(sample):
    res = requests.post(BASE_URL + "/predict", json = sample)
    return res.json()['prediction']

for sample in normal_samples:
    p = predict(sample)
    s = 'SUCCESS' if p == 'normal' else 'WRONG'
    print(f"{s}: Predicted: '{p}' - Actual: 'normal'")


for sample in anomaly_samples:
    p = predict(sample)
    s = 'SUCCESS' if p == 'anomaly' else 'WRONG'
    print(f"{s}: Predicted: '{p}' - Actual: 'anomaly'")



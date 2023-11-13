import csv
import json
import os

csv_file_path = '/data2/chengjunhao/data/LEAF/train/train_label.csv'  # your csv path


csv_folder = os.path.dirname(csv_file_path)
json_file_path = os.path.join(csv_folder, 'label.json')

data = []

#label match
label_mapping = {
    'healthy': 0,
    'rust': 1,
    'scab': 2,
    'frog_eye_leaf_spot': 3,
    'powdery_mildew': 4,
    'complex': 5,
}


with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  
    ids=0
    for row in csv_reader:
        if len(row) == 2:
            image = row[0]
            labels = row[1].split()  # split multi-labels
            for i in range(len(labels)):
                labels[i] = label_mapping.get(labels[i], -1)
            data.append({'image': image, 'labels': labels, 'id': ids})
            ids+=1

# write in new josn
with open(json_file_path, 'w') as file:
    json.dump(data, file, indent=4)

print('success')
import csv
import os

csv_path = '/data2/chengjunhao/data/LEAF/train/train_label1.csv' #your csv path

with open(csv_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  
    rows = list(reader)  

# expand csv
expanded_rows = []
for row in rows:
    image = row[0]
    label = row[1]
    image_name, image_ext = os.path.splitext(image)
    expanded_image = f"{image_name}{image_ext}"
    expanded_rows.append([expanded_image, label])
    for i in range(1, 4):
        expanded_image = f"{image_name}{i}{image_ext}"
        expanded_rows.append([expanded_image, label])

# write new
with open('expanded_file.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(expanded_rows)
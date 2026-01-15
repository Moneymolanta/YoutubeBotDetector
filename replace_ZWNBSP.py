
import csv

data = []
with open("data/Youtube Spam Dataset/Youtube-Spam-Dataset.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    # imho you are creating a data structure, the file was its (original) source
    # so don't name it 'file' anymore
    data = [[c.replace('\ufeff', '') for c in row] for row in reader]

with open("fixed_csv.csv", "w", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerows(data)
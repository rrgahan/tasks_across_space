import csv
import pandas as pd

def main():
    data = pd.read_csv('output/esmi_tasks.csv', names=['stem', 'readable', 'count'])
    stems = set(data['stem'])
    combined_pairs = {}

    for index, row in data.iterrows():
        if row['stem'] not in combined_pairs.keys():
            combined_pairs[row['stem']] = {
                'readable': row['readable'],
                'count': row['count']
            }
        else:
            combined_pairs[row['stem']]['count'] += row['count']

    with open('output/combined_tasks.csv', 'w+') as f:
        writer = csv.writer(f)
        for key, value in combined_pairs.items():
            writer.writerow([value['readable'], value['count']])

if __name__ == "__main__":
    main()

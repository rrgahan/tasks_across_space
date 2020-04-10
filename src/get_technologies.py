import boto3
import concurrent.futures
import nltk
from bitarray import bitarray
from nltk.corpus import stopwords
import os
import numpy as np
import pandas as pd
import time

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
unique_base = list()
stop_words = set(stopwords.words('english'))
tech_data = pd.DataFrame()


def main():
    s3 = boto3.resource('s3')
    BUCKET_NAME = 'tasksacrossspace'

    global tech_data
    tech_data = pd.read_csv('data/hot_technologies.csv').apply(lambda x: x.str.lower())
    tech_data['technology'] = tech_data['technology'].astype(str) + ' '
    global unique_base
    unique_base = tech_data['base'].unique()

    file_count = 0
    for tsv in os.listdir("data/clean_esmi/"):
        if tsv.endswith(".tsv"):
            print(f"data/clean_esmi/{tsv}")
            t_start = time.time()

            postings = pd.read_csv(f"data/clean_esmi/{tsv}", encoding="latin-1", sep="\t", error_bad_lines=False,
                                   engine='python')
            postings = postings[postings['ad_length'].between(11, 841, inclusive=True)].reset_index(drop=True)
            descriptions = postings['description'].apply(lambda x: x.lower())
            postings_ids = postings["posting_id"]
            del postings

            with concurrent.futures.ProcessPoolExecutor() as executor:
                for postings_id, binary in zip(postings_ids, executor.map(generate_binary, descriptions)):
                    with open(f'output/tech_binaries/binary_{tsv}.csv', 'a+') as f:
                        f.write("%s,%s\n" % (postings_id, binary.to01()))
                        f.close()

            print(f"File #{file_count}: {time.time() - t_start}")
            s3.meta.client.upload_file(f'output/tech_binaries/binary_{tsv}.csv', BUCKET_NAME,
                                       f'tech_vectors_binary/binaries_{tsv}.csv')
            file_count += 1


def generate_binary(description):
    tokens = tokenizer.tokenize(description)
    description = ' '.join([w for w in tokens if w not in stop_words])
    binary = bitarray()

    for tech in tech_data['technology'].values:
        if tech in description:
            binary.append(True)
        else:
            binary.append(False)

    reduced_binary = bitarray()
    for base in unique_base:
        indexes = np.where(tech_data['base'].values == base)
        value = False
        for index in indexes[0]:
            value = value or binary[index]
        reduced_binary.append(value)

    return reduced_binary


if __name__ == "__main__":
    main()

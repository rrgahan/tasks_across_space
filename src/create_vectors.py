import boto3
import concurrent.futures
import csv
import nltk
import numpy as np
import os
import pandas as pd
import time

from bitarray import bitarray

from build_vocabulary import cut_non_task_words


stemmer = nltk.stem.PorterStemmer()
tasks = list()

def main():
    s3 = boto3.resource('s3')
    BUCKET_NAME = 'tasksacrossspace'

    tasks_csv = pd.read_csv('output/combined_tasks.csv')
    tasks_csv.columns = ["task", "count"]

    t0 = time.time()
    tasks.extend(prepare_tasks(tasks_csv))
    del tasks_csv
    t1 = time.time()
    print("Prepare tasks: {}".format(t1 - t0))
    file_count = 0
    for tsv in os.listdir("data/clean_esmi/"):
        if tsv.endswith(".tsv"):
            print(f"data/clean_esmi/{tsv}")
            t_start = time.time()
            postings = pd.read_csv(f"data/clean_esmi/{tsv}", encoding="latin-1", sep="\t", error_bad_lines=False, engine='python')
            postings = postings[postings['ad_length'].between(11, 841, inclusive=True)].reset_index(drop=True)
            postings_ids = postings["posting_id"]
            postings_descriptions = postings["description"]
            del postings

            with concurrent.futures.ProcessPoolExecutor() as executor:
                for posting_id, binary in zip(postings_ids, executor.map(generate_binary, postings_descriptions)):
                    with open(f'output/binaries/binary_{tsv}.csv', 'a+') as f:
                        f.write("%s,%s\n" % (posting_id, binary.to01())) 
                        f.close()

            print(f"File #{file_count}: {time.time() - t_start}")
            s3.meta.client.upload_file(f'output/binaries/binary_{tsv}.csv', BUCKET_NAME, f'vectors_binary/binaries_{tsv}.csv')
            file_count += 1

    s3.meta.client.upload_file('output/tasks_used.csv', BUCKET_NAME, 'vectors_binary/tasks_used.csv')

    tn = time.time()
    print("Total time: {}".format(tn - t0))

    return

def generate_binary(description):
    word_tag_pairs = cut_non_task_words(description)
    possible_tasks = create_possible_tasks(word_tag_pairs)
    binary = bitarray()
    for task in tasks:
        if task in possible_tasks:
            binary.append(True)
        else:
            binary.append(False)
    return binary


def create_possible_tasks(word_tag_pairs):
    possible_tasks = []
    for (k, (word, tag)) in enumerate(word_tag_pairs):
        if tag == 'VERB':
            next_index = k + 1
            while next_index < len(word_tag_pairs) - 1:
                next_pair = word_tag_pairs[next_index]
                if next_pair[1] == 'NOUN':
                    possible_tasks.append(stemmer.stem(word) + ' ' + stemmer.stem(next_pair[0]))
                    break
                else:
                    next_index += 1

    return possible_tasks


def prepare_tasks(tasks_csv):
    # Number of tasks to keep
    threshold = 2000
    tasks_list = list(tasks_csv.nlargest(threshold, "count").dropna().reset_index(drop=True)['task'])
    with open('output/tasks_used.csv', 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(tasks_list)
    defined_task_stems = [stemmer.stem(t.split(' ')[0]) + ' ' + stemmer.stem(t.split(' ')[1]) for t in tasks_list]
    return defined_task_stems


if __name__ == "__main__":
    main()

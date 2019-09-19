import boto3
import csv
import nltk
import numpy as np
import pandas as pd
import time

from bitarray import bitarray

from build_vocabulary import cut_non_task_words


def main():
    s3 = boto3.resource('s3')
    BUCKET_NAME = 'tasksacrossspace'

    tasks_csv = pd.read_csv('data/tasks_large.csv')
    tasks_csv.columns = ["task", "count"]

    stemmer = nltk.stem.PorterStemmer()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    t0 = time.time()
    tasks = prepare_tasks(tasks_csv, stemmer)
    del tasks_csv
    t1 = time.time()
    print("Prepare tasks: {}".format(t1 - t0))

    postings = pd.read_csv('data/job_postings_large.csv', encoding='latin-1')
    postings = postings[postings['ad_length'].between(11, 841, inclusive=True)].reset_index(drop=True)
    postings_ids = postings["posting_id"]
    postings_descriptions = postings["description"]
    del postings
    t2 = time.time()
    print("Load data: {}".format(t2 - t1))

    binaries = {}
    i = 0
    for i in range(len(postings_ids)):
        description = postings_descriptions[i]
        post_id = postings_ids[i]
        binary = generate_binary(description, tasks, tokenizer, stemmer)
        binaries[post_id] = binary.to01()
        i += 1
        if (i > 50):
            break

    print(binaries)

    with open('output/tasks_used.csv', 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(tasks)

    with open('output/binary_test.csv', 'w') as f:
        for key in binaries.keys():
            f.write("%s,%s\n" % (key, binaries[key]))

    # This will probably take up too much memory if we don't chunk
    # df.to_csv('/tmp/output.csv', sep=",")
    # s3.meta.client.upload_file('/tmp/output.csv', BUCKET_NAME, 'vectors_large/output.csv')

    tn = time.time()
    print("Total time: {}".format(tn - t0))

    return


def generate_binary(description, tasks, tokenizer, stemmer):
    word_tag_pairs = cut_non_task_words(description, tokenizer)
    possible_tasks = create_possible_tasks(word_tag_pairs, stemmer)
    binary = bitarray()
    for task in tasks:
        if task in possible_tasks:
            binary.append(True)
        else:
            binary.append(False)
    return binary


def create_possible_tasks(word_tag_pairs, stemmer):
    tasks = []
    for (k, (word, tag)) in enumerate(word_tag_pairs):
        if tag == 'VERB':
            next_index = k + 1
            while next_index < len(word_tag_pairs) - 1:
                next_pair = word_tag_pairs[next_index]
                if next_pair[1] == 'NOUN':
                    tasks.append(stemmer.stem(word) + ' ' + stemmer.stem(next_pair[0]))
                    break
                else:
                    next_index += 1

    return tasks


def prepare_tasks(tasks_csv, stemmer):
    # Number of tasks to keep
    threshold = 1000
    tasks_list = list(tasks_csv.nlargest(threshold, "count").dropna().reset_index(drop=True)['task'])
    defined_task_stems = [stemmer.stem(t.split(' ')[0]) + ' ' + stemmer.stem(t.split(' ')[1]) for t in tasks_list]
    return defined_task_stems


if __name__ == "__main__":
    main()

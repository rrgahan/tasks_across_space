import boto3
import nltk
import numpy as np
import pandas as pd
import time

from build_vocabulary import cut_non_task_words


def main():
    s3 = boto3.resource('s3')
    t0 = time.time()

    v = pd.read_csv('output/all_small.csv')
    v.columns = ["task", "count"]

    t1 = time.time()
    vocabulary = trim_vocab(v)
    t2 = time.time()
    print("Trim vocab: {}".format(t2 - t1))

    BUCKET_NAME = 'tasksacrossspace'
    KEY = 'job_postings.csv'
    s3.Bucket(BUCKET_NAME).download_file(KEY, 'data/job_postings.csv')
    postings = pd.read_csv('data/job_postings.csv')
    postings = postings[postings['ad_length'].between(20, 400, inclusive=True)]
    print(postings.shape)
    posting_ids = postings["posting_id"]
    posting_descs = postings["description"]
    chunk_size = 1000
    chunk_count = round(len(posting_ids.index) / chunk_size)
    posting_ids_splits = np.array_split(posting_ids, chunk_count)
    posting_descs_splits = np.array_split(posting_descs, chunk_count)

    tasks = list(vocabulary["task"])
    col_names = ["posting_id", "description"] + tasks
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    stemmer = nltk.stem.PorterStemmer()
    defined_task_stems = [[stemmer.stem(t.split(' ')[0]), stemmer.stem(t.split(' ')[1])] for t in tasks]

    for i in range(int(chunk_count)):
        t3 = time.time()
        print("Chunk {} out of {}".format(i, chunk_count - 1))
        df = create_dummy_df(vocabulary, posting_ids_splits[i], posting_descs_splits[i], col_names)
        t4 = time.time()
        print("Create dataframe: {}".format(t4 - t3))

        df = fill_df(df, defined_task_stems, tokenizer, stemmer)
        t5 = time.time()
        print("Fill dataframe: {}".format(t5 - t4))

        df.to_csv('output/200k_vectorized/{}.csv'.format(i), sep=",")

    tn = time.time()
    print("Total time: {}".format(tn - t0))

    return


def create_dummy_df(vocabulary, ids, descs, col_names):
    df = pd.DataFrame(columns=col_names)
    df["posting_id"] = ids
    df["description"] = descs

    df.reset_index(drop=True, inplace=True)

    return df


def create_possible_tasks(stems):
    tasks = []
    for (k, (word, tag)) in enumerate(stems):
        if tag == 'VERB':
            next_index = k + 1
            while next_index < len(stems) - 1:
                next_tuple = stems[next_index]
                if next_tuple[1] == 'NOUN':
                    tasks.append(word + ' ' + next_tuple[0])
                    break
                else:
                    next_index += 1

    return tasks


def fill_df(df, defined_task_stems, tokenizer, stemmer):
    for index, row in df.iterrows():
        print("Description #{}".format(index))
        try:
            tuples = cut_non_task_words(row["description"], tokenizer)
            stems = [[stemmer.stem(t[0]), t[1]] for t in tuples]
            possible_tasks = create_possible_tasks(stems)
            for col, defined_stem in enumerate(defined_task_stems):
                combined = defined_stem[0] + " " + defined_stem[1]
                if combined in possible_tasks:
                    df.iloc[index, col + 2] = 1
                else:
                    df.iloc[index, col + 2] = 0
        except TypeError:
            pass
    return df


def trim_vocab(v):
    # Number of tasks to keep
    threshold = 300
    v = v.nsmallest(threshold, "count")
    v = v.dropna()
    v.reset_index(drop=True, inplace=True)
    return v


if __name__ == "__main__":
    main()

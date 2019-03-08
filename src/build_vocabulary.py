import boto3
import csv
import nltk
import pandas as pd
import time

from collections import Counter, OrderedDict
from nltk.tag import map_tag


class OrderedCounter(Counter, OrderedDict):
    pass


def main():
    s3 = boto3.resource('s3')
    t0 = time.time()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    t1 = time.time()
    descriptions = read_all(s3)
    t2 = time.time()
    print("Get descriptions: {}".format(t2 - t1))

    t3 = time.time()
    phrases = get_relevant_phrases(descriptions)
    t4 = time.time()
    print("Get phrases: {}".format(t4 - t3))

    t5 = time.time()
    pair_set = generate_all_noun_pairs(phrases, tokenizer)
    t6 = time.time()
    print("Get pairs: {}".format(t6 - t5))

    filename = '/tmp/all_large.csv'
    t7 = time.time()
    write_set(pair_set, filename)
    BUCKET_NAME = 'tasksacrossspace'
    s3.meta.client.upload_file(filename, BUCKET_NAME, 'tasks_large.csv')
    tn = time.time()
    print("Write file: {}".format(tn - t7))
    print("Total time: {}".format(tn - t0))


def cut_non_task_words(phrases, tokenizer):
    description_token = tokenizer.tokenize(phrases)
    tagged = nltk.pos_tag(description_token)
    # Turn description into simplified tags (noun, verb, adjective, etc)
    simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in tagged]

    only_noun_verb = []
    for (j, (word, tag)) in enumerate(simplifiedTags):
        if tag == 'VERB' or tag == 'NOUN':
            only_noun_verb.append([word, tag])
    return only_noun_verb


def generate_all_noun_pairs(phrases, tokenizer):
    stemmer = nltk.stem.PorterStemmer()
    pair_set = {}
    # for index, description in descriptions.iteritems():
    for index, phrase in enumerate(phrases):
        print("Phrase #{}".format(index))
        try:
            # Dict of verb/noun pairs found in individual descriptions. {stem: readable}
            print("start cutting non task")
            only_noun_verb = cut_non_task_words(phrase, tokenizer)
            print("end cutting non task")

            for (k, (word, tag)) in enumerate(only_noun_verb):
                if tag == 'VERB':
                    next_index = k + 1
                    while next_index < len(only_noun_verb) - 1:
                        next_tuple = only_noun_verb[next_index]
                        if next_tuple[1] == 'NOUN':
                            stemmed_pair = stemmer.stem(word) + ' ' + stemmer.stem(next_tuple[0])
                            if stemmed_pair not in pair_set.keys():
                                pair_set[stemmed_pair] = {
                                    'readable': word + ' ' + next_tuple[0],
                                    'count': 1
                                }
                            else:
                                pair_set[stemmed_pair]['count'] += 1
                            break
                        else:
                            next_index += 1
        except TypeError:
            pass

    return pair_set


# For building the vocabulary, we only want to get tasks from a very specific part of the posting.
# Here, we cut out text not in a sentence that includes a clue word.
def get_relevant_phrases(descriptions):
    clue_words = ['duties', 'responsibilities', 'summary', 'tasks', 'functions']
    relevant_phrases = []
    for description in descriptions:
        print('New description')
        try:
            description = description.lower()
            for word in clue_words:
                if word in description:
                    first_split = description.split(word, 1)[1]
                    second_split = first_split.split('.', 1)[0]
                    relevant_phrases.append(second_split)
                    break

        except AttributeError:
            pass
    return relevant_phrases


def read_all(s3):
    BUCKET_NAME = 'tasksacrossspace'
    KEY = 'job_postings_large.csv'
    print('Getting file')
    s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/job_postings.csv')
    print('file downloaded')
    postings = pd.read_csv('/tmp/job_postings.csv')
    return postings['description']


def write_set(dictionary, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for key, value in dictionary.items():
            writer.writerow([value['readable'], value['count']])


if __name__ == "__main__":
    main()

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
    t0 = time.time()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    t1 = time.time()
    descriptions = read_all()
    t2 = time.time()
    print("Get descriptions: {}".format(t2 - t1))
    # descriptions = read_all_soc()
    # descriptions = read_some()
    # codes, descriptions = read_soc_defs()
    # for i, description in enumerate(descriptions):
    #     pair_set = generate_all_noun_pairs(description, tokenizer)
    #     filename = 'output/soc/{}.csv'.format(codes[i])
    #     write_set(pair_set, filename)

    t3 = time.time()
    phrases = get_relevant_phrases(descriptions)
    t4 = time.time()
    print("Get phrases: {}".format(t4 - t3))

    t5 = time.time()
    pair_set = generate_all_noun_pairs(phrases, tokenizer)
    t6 = time.time()
    print("Get paris: {}".format(t6 - t5))

    filename = 'output/all_small.csv'
    t7 = time.time()
    write_set(pair_set, filename)
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
        try:
            # Dict of verb/noun pairs found in individual descriptions. {stem: readable}
            only_noun_verb = cut_non_task_words(phrase, tokenizer)

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
    non_clue_descriptions = []
    for description in descriptions:
        try:
            flag = True
            description = description.lower()
            for word in clue_words:
                if word in description:
                    first_split = description.split(word, 1)[1]
                    second_split = first_split.split('.', 1)[0]
                    relevant_phrases.append(second_split)
                    flag = False
                    break

            if flag:
                non_clue_descriptions.append(description)
        except AttributeError:
            pass
    with open('non_clue_23-1011.txt', 'w') as f:
        for item in non_clue_descriptions:
            f.write("%s\n" % item)
    print(len(relevant_phrases))
    return relevant_phrases


def read_all():
    s3 = boto3.resource('s3')
    BUCKET_NAME = 'tasksacrossspace'
    KEY = 'job_postings_large.csv'
    s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/job_postings.csv')
    postings = pd.read_csv('/tmp/job_postings.csv')
    return postings['description']


def read_all_soc():
    df = pd.read_csv('data/job_postings.csv')
    onet_code = '23-1011.00'
    df = df.loc[df['onet'] == onet_code]
    print(df.shape)
    descriptions = df['description']
    descriptions.reset_index(drop=True, inplace=True)
    return descriptions


def read_some():
    input_data = 'output/41-1011.00_desc.txt'
    file = open(input_data, 'r')
    descriptions = file.readlines()
    descriptions = [s.strip('\n') for s in descriptions]
    descriptions = list(filter(None, descriptions))
    return descriptions


def read_soc_defs():
    df = pd.read_csv('data/soc_2018_definitions.csv')
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    codes = df['SOC Code']
    descriptions = df['SOC Definition']
    return codes, descriptions


def write_set(dictionary, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for key, value in dictionary.items():
            writer.writerow([value['readable'], value['count']])


if __name__ == "__main__":
    main()

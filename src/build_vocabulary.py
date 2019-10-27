import boto3
import csv
import nltk
import pandas as pd
import time

from collections import Counter, OrderedDict
from nltk.tag import map_tag


tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

def main():
    s3 = boto3.resource('s3')
    t0 = time.time()
    postings = pd.read_csv('data/job_postings_large.csv', encoding='latin-1')
    descriptions = postings['description']
    del postings
    print("Get descriptions done")

    phrases = get_relevant_phrases(descriptions)
    del descriptions
    print("Get phrases done")

    pair_set = generate_all_noun_pairs(phrases)
    del phrases
    print("Get pairs done")

    filename = 'output/large_tasks_test.csv'
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for key, value in pair_set.items():
            writer.writerow([value['readable'], value['count']])

    BUCKET_NAME = 'tasksacrossspace'
    # s3.meta.client.upload_file(filename, BUCKET_NAME, 'tasks_five_percent.csv')
    tn = time.time()
    print("Total time: {}".format(tn - t0))


def cut_non_task_words(phrases):
    description_token = tokenizer.tokenize(phrases)
    tagged = nltk.pos_tag(description_token)
    # Turn description into simplified tags (noun, verb, adjective, etc)
    simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in tagged]

    only_noun_verb = []
    for (j, (word, tag)) in enumerate(simplifiedTags):
        if tag == 'VERB' or tag == 'NOUN':
            only_noun_verb.append([word, tag])
    return only_noun_verb


def generate_all_noun_pairs(phrases):
    stemmer = nltk.stem.PorterStemmer()
    pair_set = {}
    # TODO: Make concurrent
    for index, phrase in enumerate(phrases):
        print("Phrase #{} of {}".format(index, len(phrases)))
        try:
            # Dict of verb/noun pairs found in individual descriptions. {stem: readable}
            only_noun_verb = cut_non_task_words(phrase)

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
    # TODO: Make concurrent
    for description in descriptions:
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


if __name__ == "__main__":
    main()

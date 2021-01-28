import boto3
import csv
import nltk
import os
import pandas as pd
import time
import concurrent.futures

from collections import Counter, OrderedDict
from nltk.tag import map_tag

stemmer = nltk.stem.PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
clue_words = ['duties', 'responsibilities', 'summary', 'tasks', 'functions']

def main():
    s3 = boto3.resource('s3')
    t0 = time.time()

    for tsv in os.listdir("data/clean_esmi/"):
        if tsv.endswith(".tsv"):
            print(f"data/clean_esmi/{tsv}")
            data = pd.read_csv(f"data/clean_esmi/{tsv}", encoding="latin-1", sep="\t", error_bad_lines=False, engine='python')
            descriptions = data['description']

            phrases = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for phrase in executor.map(get_relevant_phrases, descriptions):
                    phrases.append(phrase)

            del descriptions
            print("Get phrases done")

            pair_set = generate_all_noun_pairs(phrases)
            del phrases
            print("Get pairs done")

            filename = 'output/esmi_tasks.csv'
            with open(filename, 'a+') as f:
                writer = csv.writer(f)
                for key, value in pair_set.items():
                    writer.writerow([key, value['readable'], value['count']])
            del data
            print("Get descriptions done")

    tn = time.time()
    print("Total time: {}".format(tn - t0))


def cut_non_task_words(phrases):
    only_noun_verb = []
    if phrases:
        # for phrase in phrases.split():
        description_token = tokenizer.tokenize(phrases)
        tagged = nltk.pos_tag(description_token)
        # Turn description into simplified tags (noun, verb, adjective, etc)
        simplified_tags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in tagged]

        for (j, (word, tag)) in enumerate(simplified_tags):
            if tag == 'VERB' or tag == 'NOUN':
                only_noun_verb.append([word, tag])
        return only_noun_verb
    else:
        return []


def generate_all_noun_pairs(phrases):
    stemmer = nltk.stem.PorterStemmer()
    pair_set = {}
    for index, phrase in enumerate(phrases):
        print("Phrase #{} of {}".format(index, len(phrases)))
        # Dict of verb/noun pairs found in individual descriptions. {stem: readable}
        only_noun_verb = cut_non_task_words(phrase)

        if only_noun_verb:
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

    return pair_set


# For building the vocabulary, we only want to get tasks from a very specific part of the posting.
# Here, we cut out text not in a sentence that includes a clue word.
def get_relevant_phrases(description):
    relevant_phrases = []
    try:
        description = description.lower()
        for word in clue_words:
            if word in description:
                first_split = description.split(word, 1)[1]
                second_split = first_split.split('.', 1)[0]
                relevant_phrases.append(second_split)
                break
        return relevant_phrases

    except AttributeError:
        pass


def prepare_tasks():
    # Number of tasks to keep
    tasks_csv = pd.read_csv('output/combined_tasks.csv')
    tasks_csv.columns = ["task", "count"]
    threshold = 2000
    tasks_list = list(tasks_csv.nlargest(threshold, "count").dropna().reset_index(drop=True)['task'])
    with open('output/tasks_used.csv', 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(tasks_list)
    defined_task_stems = [stemmer.stem(t.split(' ')[0]) + ' ' + stemmer.stem(t.split(' ')[1]) for t in tasks_list]
    return defined_task_stems


if __name__ == "__main__":
    main()

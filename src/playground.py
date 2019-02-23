import nltk
import random

from collections import Counter
from nltk.tag import map_tag


def main():
    return


def getting_counts():
    temp = [{'writ paper': 'writing papers'},
            {'meet client': 'meeting clients'},
            {'writ paper': 'write papers'},
            {'meet client': 'meet clients'}]
    for_counter = []
    readable = {}
    for dictionary in temp:
        key = list(dictionary.keys())[0]
        for_counter.append(key)
        if key not in list(readable.keys()):
            readable[key] = dictionary[key]
    readable_count = {}
    for key in list(readable.keys()):
        readable_count[readable[key]] = dict(Counter(for_counter))[key]

    print(readable_count)


def get_pos(descriptions, tokenizer, output):
    for index, description in enumerate(descriptions):
        description_token = tokenizer.tokenize(description)
        tagged = nltk.pos_tag(description_token)
        simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in tagged]
        desc_values = []
        for i in range(0, len(description_token)):
            desc_values.append([description_token[i], tagged[i], simplifiedTags[i]])
        with open('{}_{}.txt'.format(output, index), 'w') as f:
            f.write("{}\n\n\n".format(description))
            for item in desc_values:
                f.write('{}\n'.format(item))


def get_random_desc(code, df):
    df = df.loc[df['onet'] == code]
    descriptions = df['description']
    length = len(descriptions)
    rand_index = random.sample(range(0, length), 10)
    rand_desc = []
    for i in rand_index:
        rand_desc.append(descriptions.iloc[i])
    with open('output/{}_desc.txt'.format(code), 'w') as f:
        for item in rand_desc:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()

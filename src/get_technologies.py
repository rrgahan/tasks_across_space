import nltk
from nltk.corpus import stopwords
import pandas as pd

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


def main():
    tech_data = pd.read_csv('data/hot_technologies.csv').apply(lambda x: x.str.lower())

    technologies_count = pd.DataFrame(tech_data['base'].unique(), columns=['technology'])
    technologies_count['count'] = 0

    postings_data = pd.read_csv('data/subset.csv')
    descriptions = postings_data['description'].apply(lambda x: x.lower())
    del postings_data

    for description in descriptions:
        # TODO: NLTK stuff
        # Tokenize
        # Remove stop words? What are they? Any chance valid words would be removed?
        # Loop through each word, see if is in tech_data[technology]

        stop_words = set(stopwords.words('english'))
        tokens = tokenizer.tokenize(description)
        description_tokens = [w for w in tokens if w not in stop_words]

        for word in description_tokens:
            if word in tech_data['technology'].values:
                base = tech_data['base'].values[tech_data['technology'] == word][0]
                technologies_count.loc[technologies_count['technology'] == base, 'count'] += 1
        break
    print(technologies_count)

    return


if __name__ == "__main__":
    main()

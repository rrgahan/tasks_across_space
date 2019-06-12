import csv
import nltk
import pandas as pd


def main():
    data = pd.read_csv('data/subset.csv')
    descriptions = data.description
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    keywords = ["hour", "week", "000", "500", "$"]

    salary_words_list = []

    for description in descriptions:
        # Tokenize
        description_tokens = tokenizer.tokenize(description)
        # print(description_tokens)

        sa_ = [i for i, x in enumerate(description_tokens) if x in keywords]
        sa_idx = sorted([x+1 for x in sa_[0:len(sa_)]] +
                        [x+2 for x in sa_[0:len(sa_)]] +
                        [x+3 for x in sa_[0:len(sa_)]] +
                        [x+4 for x in sa_[0:len(sa_)]] +
                        [x-1 for x in sa_[0:len(sa_)]] +
                        [x-2 for x in sa_[0:len(sa_)]] + 
                        [x-3 for x in sa_[0:len(sa_)]] +
                        [x-4 for x in sa_[0:len(sa_)]] +
                        [x-5 for x in sa_[0:len(sa_)]] +
                        [x for x in sa_[0:len(sa_)]])
        sa_idx = list(set(
            [sa_idx[x] for x in range(0, len(sa_idx))
                if sa_idx[x] < len(description_tokens)]))
        salarywords = [
            description_tokens[x] for x in [x for i, x in enumerate(sa_idx)
                                            if sa_idx[i] >= 0]]

        if len(salarywords):
            salary_words_list.append(salarywords)

    with open("output/salary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(salary_words_list)

    return


if __name__ == "__main__":
    main()

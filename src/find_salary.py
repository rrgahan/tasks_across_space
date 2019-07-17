import csv
import itertools
import nltk
import pandas as pd


def main():
    data = pd.read_csv('data/with_benefits.csv')
    descriptions = data.description

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    columns = ["wage", "starting_bonus", "retirement_plans", "insurance"]
    flag_df = pd.DataFrame(columns=columns)

    find_wage(descriptions, tokenizer)
    return
    flag_df.starting_bonus = find_starting_bonus(descriptions)
    flag_df.retirement_plans = find_retirement_plans(descriptions)
    flag_df.insurance = find_insurance(descriptions, tokenizer)
    print(flag_df)


def find_wage(descriptions, tokenizer):
    pre_keywords = ["earn", "pay", "get"]
    
    return


def find_starting_bonus(descriptions):
    bonus_keywords = ["sign_on bonus"]
    flags = []

    for description in descriptions:
        flag = False
        for keyword in bonus_keywords:
            if keyword in description:
                flag = True
                break
        flags.append(flag)

    return flags


def find_retirement_plans(descriptions):
    retirement_keywords = ["401_k", "401 (k)", "401 ( k )", "401_K", "401 (K)", "401 ( K )"]
    flags = []

    for description in descriptions:
        flag = False
        for keyword in retirement_keywords:
            if keyword in description:
                flag = True
                break
        flags.append(flag)

    return flags


def find_insurance(descriptions, tokenizer):
    insurance_keywords_first = ["health", "dental", "vision", "medical", "medi_cal", "health_care"]
    insurance_keywords_second = ["coverage", "insurance", "benefits"]
    insurance_keywords_combined = list(map('_'.join, itertools.product(insurance_keywords_first, insurance_keywords_second)))

    flags = []

    for description in descriptions:
        flag = False
        # TODO: Tokenize here to look for second
        tokens = tokenizer.tokenize(description)
        for i, token in enumerate(tokens):
            if token in insurance_keywords_first:
                next_index = i + 1
                if tokens[next_index] in insurance_keywords_second:
                    flag = True
                    break
            if token in insurance_keywords_combined:
                flag = True
                break
        flags.append(flag)

    return flags


def coauthor_keywords():
    data = pd.read_csv('data/subset.csv')
    descriptions = data.description
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    keywords = ["hour", "week", "000", "500", "$"]

    salary_words_list = []

    for description in descriptions:
        description_tokens = tokenizer.tokenize(description)

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


if __name__ == "__main__":
    main()

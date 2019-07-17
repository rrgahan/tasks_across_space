import csv
import itertools
import nltk
import pandas as pd


def main():
    data = pd.read_csv('data/with_benefits.csv')
    descriptions = data.description

    tokenizer = nltk.tokenize.RegexpTokenizer('\w+|\$[\d\.]+|\S+')

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
    for description in descriptions:
        tokens = tokenizer.tokenize(description)
        for i, token in enumerate(tokens):
            if token in pre_keywords:
                last_index = len(tokens) - 1 if i + 10 >= len(tokens) else i + 10
                for j in range(i, last_index):
                    if "$" in tokens[j]:
                        dollar_range = len(tokens) - 1 if j + 4 >= len(tokens) else j + 4
                        numbers = []
                        for k in range(j + 1, dollar_range):
                            if is_number(tokens[k]):
                                numbers.append(tokens[k])
                            elif "$" in tokens[k]:
                                break
                        wage = ""
                        if len(numbers) == 1:
                            wage = numbers[0]
                        elif len(numbers) == 2:
                            # This would be cents
                            if len(numbers[1]) == 2:
                                wage = "{}.{}".format(numbers[0], numbers[1])
                            # This would be thousands
                            if len(numbers[1]) == 3:
                                wage = "{},{}".format(numbers[0], numbers[1])
                        print(wage)
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


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()

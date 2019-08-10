import csv
import itertools
import nltk
import pandas as pd


def main():
    data = pd.read_csv('data/job_postings_large.csv')
    descriptions = data.description

    tokenizer = nltk.tokenize.RegexpTokenizer('\w+|\$[\d\.]+|\S+')

    columns = ["posting_id", "wage_amount", "wage_frequency", "starting_bonus", "retirement_plans", "insurance"]
    flag_df = pd.DataFrame(columns=columns)

    flag_df.posting_id = data["posting_id"]
    (wage_amount, wage_frequency) = find_wage(descriptions, tokenizer)
    flag_df.wage_amount = wage_amount
    flag_df.wage_frequency = wage_frequency
    flag_df.starting_bonus = find_starting_bonus(descriptions)
    flag_df.retirement_plans = find_retirement_plans(descriptions)
    flag_df.insurance = find_insurance(descriptions, tokenizer)
    flag_df.to_csv("output/salary_subset.csv", index=False)
    print(flag_df)


def find_wage(descriptions, tokenizer):
    pre_keywords = ["earn", "pay", "get", "salary", "earned", "paid", "got", "rate", "rates"]
    wage_frequency_keywords = ["weekend", "weekly", "week", "hourly", "hour", "yearly", "year", "salary", "annual", "annually"]

    wage_amount = []
    wage_frequency = []

    for description in descriptions:
        tokens = tokenizer.tokenize(description)
        tokens_length = len(tokens) - 1
        description_wage_amount = []
        description_wage_frequency = ""
        for i, token in enumerate(tokens):
            if token in pre_keywords:
                # Look for "$", then amount, then frequency
                dollar_index = 0
                dollar_ending_index = len(tokens) if i + 10 >= len(tokens) else i + 10
                for i in range(i, dollar_ending_index):
                    if "$" in tokens[i]:
                        dollar_index = i
                        wage_index, amount = find_amount(dollar_index + 1, tokens)
                        description_wage_amount.append(amount)
                        if wage_index:
                            frequency = find_frequency(wage_index, tokens, wage_frequency_keywords)
                            if frequency:
                                description_wage_frequency = frequency

            if description_wage_amount and description_wage_frequency:
                break
        if description_wage_frequency:
            wage_amount.append(description_wage_amount)
            wage_frequency.append(description_wage_frequency)
        else:
            wage_amount.append([])
            wage_frequency.append("")

    return (wage_amount, wage_frequency)


# Will return index and amounts
def find_amount(starting_index, tokens):
    ending_index = len(tokens) if starting_index + 6 >= len(tokens) else starting_index + 6
    wage_index = 0
    wage = ""
    numbers = []
    # Find near numbers
    for i in range(starting_index, ending_index):
        if is_number(tokens[i]):
            numbers.append(tokens[i])
            wage_index = i
        elif tokens[i].lower() == "k":
            numbers.append("000")
        elif "$" in tokens[i]:
            break

    if len(numbers) == 1:
        wage = numbers[0]
    elif len(numbers) == 2:
        # This would be cents
        if len(numbers[1]) == 2:
            wage = "{}.{}".format(numbers[0], numbers[1])
        # This would be thousands
        if len(numbers[1]) == 3:
            wage = "{},{}".format(numbers[0], numbers[1])
    return (wage_index, wage)


def find_frequency(starting_index, tokens, wage_frequency_keywords):
    ending_index = len(tokens) if starting_index + 8 >= len(tokens) else starting_index + 8
    for i in range(starting_index, ending_index):
        if tokens[i] in wage_frequency_keywords:
            return tokens[i]
    return ""


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
                else:
                    max_search_index = len(tokens) if next_index + 4 >= len(tokens) else next_index + 4
                    for j in range(next_index, max_search_index):
                        if tokens[j] in insurance_keywords_first:
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

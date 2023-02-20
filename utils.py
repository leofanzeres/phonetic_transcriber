import csv
import random

def load_data(data_file, delimiter, to_list=False, reverse=False, to_int=False, item_addition_ratio=1.0):
    reader = csv.reader(open(data_file, mode='r'), delimiter=delimiter)
    add_item = True if random.random() < item_addition_ratio else False
    if to_list:
        dictionary = []
        for row in reader:
            if add_item:
                if to_int and row[0].isdigit():
                    row[0] = int(row[0])
                if reverse:
                    dictionary.append([row[1],row[0]])
                else:
                    dictionary.append([row[0],row[1]])
    else:
        dictionary = {}
        for row in reader:
            if add_item:
                if to_int and row[0].isdigit():
                    row[0] = int(row[0])
                if reverse:
                    dictionary[row[1]] = row[0]
                else:
                    dictionary[row[0]] = row[1]
    return dictionary


def get_swap_dict(dict):
    return {v: k for k, v in dict.items()}


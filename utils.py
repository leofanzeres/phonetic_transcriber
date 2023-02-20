import csv
import torch

def load_dictionary(dict_tsv_file, delimiter, to_list=False, reverse=False, to_int = False):
    reader = csv.reader(open(dict_tsv_file, mode='r'), delimiter=delimiter)
    if to_list:
        dictionary = []
        for row in reader:
            if to_int and row[0].isdigit():
                row[0] = int(row[0])
            if reverse:
                dictionary.append([row[1],row[0]])
            else:
                dictionary.append([row[0],row[1]])
    else:
        dictionary = {}
        for row in reader:
            if to_int and row[0].isdigit():
                row[0] = int(row[0])
            if reverse:
                dictionary[row[1]] = row[0]
            else:
                dictionary[row[0]] = row[1]
    return dictionary


def get_swap_dict(dict):
    return {v: k for k, v in dict.items()}


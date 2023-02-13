import csv
import random

def load_dictionary(dict_tsv_file, delimiter, to_list=False, reverse=False):
    reader = csv.reader(open(dict_tsv_file, mode='r'), delimiter=delimiter)
    if to_list:
        dictionary = []
        for row in reader:
            if reverse:
                dictionary.append([row[1],row[0]])
            else:
                dictionary.append([row[0],row[1]])
    else:
        dictionary = {}
        for row in reader:
            if reverse:
                dictionary[row[1]] = row[0]
            else:
                dictionary[row[0]] = row[1]
    return dictionary

def shuffle_dict(dictionary):
    keys = list(dictionary.keys())
    random.shuffle(keys)
    shuffled_dict = {}
    for key in keys:
        shuffled_dict[key] = dictionary[key]
    return shuffled_dict

def get_pairs(dictionary):
    dictlist = []
    for key, value in dictionary.items():
        dictlist.append([key,value])
    return dictlist

def split_data(dictionary, split):
    if sum(split) > 100:
        raise Exception("Sorry, sum of split ratio parts cannot be higher than 100.")
    ratio = len(dictionary) // sum(split)
    train_dict_length = split[0]*ratio
    val_dict_length = split[1]*ratio
    test_dict_length = split[2]*ratio
    train_dict = dict(list(dictionary.items())[:train_dict_length])
    val_dict = dict(list(dictionary.items())[train_dict_length:train_dict_length+val_dict_length])
    test_dict = dict(list(dictionary.items())[train_dict_length+val_dict_length:train_dict_length+val_dict_length+test_dict_length])

    return train_dict, val_dict, test_dict

def generate_indexes(dictionary):
    letters = set()
    phonemes = set()

    for word in dictionary:
        for letter in word:
            letters.add(letter)

    for word in dictionary:
        for phoneme in dictionary[word].split(" "):
            phonemes.add(phoneme)

    with open('letters.csv', 'w') as f:
        first = True
        idx = 0
        for line in letters:
            if first:
                first = False
            else:
                f.write("\n")
            f.write(f"{str(idx) + ',' + line}")
            idx += 1

    with open('phonemes.csv', 'w') as f:
        first = True
        idx = 0
        for line in phonemes:
            if first:
                first = False
            else:
                f.write("\n")
            f.write(f"{str(idx) + ',' + line}")
            idx += 1


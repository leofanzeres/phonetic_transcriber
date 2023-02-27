import random
import utils as ut
import values as v

class Dataset():
    def __init__(self, pt_br_dictionary_file, item_addition_ratio=1.0) -> None:
        data_dict = ut.load_data(pt_br_dictionary_file,'\t', to_list=False, item_addition_ratio=item_addition_ratio)
        shuffled_data = self.shuffle_dict(data_dict)
        self.train_data, self.val_data, self.test_data = self.split_data(shuffled_data, v.SPLIT)
        self.train_data_length = len(self.train_data)
        self.val_data_length = len(self.val_data)
        self.test_data_length = len(self.test_data)

    def get_train_pairs(self):
        return self.get_pairs(self.train_data)

    def get_val_pairs(self):
        return self.get_pairs(self.val_data)

    def get_test_pairs(self):
        return self.get_pairs(self.test_data)
    
    def get_pairs(self, dictionary):
        dictlist = []
        for key, value in dictionary.items():
            dictlist.append([key,value])
        return dictlist
    

    def shuffle_dict(self, dictionary):
        keys = list(dictionary.keys())
        random.shuffle(keys)
        shuffled_dict = {}
        for key in keys:
            shuffled_dict[key] = dictionary[key]
        return shuffled_dict
    

    def split_data(self, dictionary, split):
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
    
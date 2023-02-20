import torch
import utils as ut
import values as v

class Language():
    def __init__(self, phonemes_file, letters_file) -> None:
        self.phonemes = ut.load_data(phonemes_file,',', reverse=True)
        self.phonemes_length = len(self.phonemes)
        self.letters = ut.load_data(letters_file,',', reverse=True)
        self.letters_length = len(self.letters)
        
    def get_phonemes(self) -> dict:
        return self.phonemes

    def get_letters(self) -> dict:
        return self.letters

    def phonemes_length(self):
        return self.phonemes_length

    def letters_length(self):
        return self.letters_length


    def indexesFromWord(self, word, language, letter=True):
        if letter:
            indexes = []
            for letter in word:
                index = language.get_letters()[letter]
                indexes.append(int(index))
        else:
            indexes = []
            for phoneme in word.split(' '):
                index = language.get_phonemes()[phoneme]
                indexes.append(int(index))
        return indexes


    def tensorFromWord(self, word, device, language, letter=True):
        indexes = self.indexesFromWord(word, language, letter)
        indexes.append(v.EOW_TOKEN)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


    def tensorsFromPair(self, pair, device, language):
        input_tensor = self.tensorFromWord(pair[0], device, language)
        target_tensor = self.tensorFromWord(pair[1], device, language, False)
        return (input_tensor, target_tensor)


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
            idx = 2
            f.write('0,SOW\n')
            f.write('1,EOW\n')
            for line in letters:
                if first:
                    first = False
                else:
                    f.write("\n")
                f.write(f"{str(idx) + ',' + line}")
                idx += 1

        with open('phonemes.csv', 'w') as f:
            first = True
            idx = 2
            f.write('0,SOW\n')
            f.write('1,EOW\n')
            for line in phonemes:
                if first:
                    first = False
                else:
                    f.write("\n")
                f.write(f"{str(idx) + ',' + line}")
                idx += 1


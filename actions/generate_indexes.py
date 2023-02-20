import utils as ut
import values as v
from language import Language

dictionary = ut.load_dictionary(v.PT_BR_DICTIONARY_FILE,'\t')
language = Language(v.PHONEMES_FILE, v.LETTERS_FILE)
language.generate_indexes(dictionary)
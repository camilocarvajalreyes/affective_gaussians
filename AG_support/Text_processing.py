import unicodedata
import os
import glob
import itertools
import errno
import numpy as np
import operator
import re

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
DEFAULT_SPECIAL_SYMBOLS = {PAD_TOKEN, UNK_TOKEN}

#def of tokeniser
#tokenizer = torchtext.data.get_tokenizer("basic_english")

#sentence = "I'd like to go to the café tonight. What do you think?"
#tokens1 = tokenizer(sentence)
#print(tokens1)

# getting text data files from directory
#from https://github.com/abrazinskas/BSG/blob/80089f9ec4302096ca6c81e79145ec5685c8d26e/libraries/utils/paths_and_files.py#L12
def get_file_paths(path, return_file_names=False):
    """
    :param path:
    :return: :rtype: a list of filepaths that are in the folder
    """
    if os.path.isdir(path):
        paths = glob.glob(path + "/*")
    else:
        paths = [path]  # that means there is only one file

    if return_file_names:
        paths = [(p, p.split('/')[-1]) for p in paths]
    return paths

def create_folders_if_not_exist(filename):
    if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

#iterator
class TextDataIterator():
    
    def __init__(self, data_path, tokenizer):
        """
        Text data iterator for open text, that returns tokenized sentences. Assumes that each sentence is separated
        by a new line.
        """
        self.tokenizer = tokenizer
        self.data_path = data_path

    def __iter__(self):
        for filename in get_file_paths(self.data_path):
            with open(filename, encoding="utf8") as f:
                for line in f:
                    tokens = self.tokenizer(line)
                    yield tokens,

"""#Example
path = "C:\\Users\\fast\\camilo\\affective_gaussians\\BSG\\data"
data_iterator = TextDataIterator(path,tokenizer)
s=0
for token in data_iterator:
    s+=1
    print('tokenized sentence '+str(s))
    print(token)
    print('----------------------------------------------')
    if s>20:
        break"""

class Word:
    def __init__(self, token, id, count):
        self.token = token
        self.id = id
        self.count = count


class Vocabulary:
    """
    A general purpose vocabulary class.
    """

    def __init__(self, max_size=None, min_count=5):
        """
        :param data_iterator: an data_iterator over data object
        :param max_size: maximum size of the vocabulary, words that don't fit are discarded.
        :param min_count: minimum frequency count of a word to be added to the vocabulary.
        """
        self.max_size = max_size
        self.min_count = min_count
        self.total_count = 0

        # create data collectors
        self._id_to_word_obj = []
        self._word_to_word_obj = {}
        self.special_symbols = {}

    def load(self, vocab_file_path, sep=" "):
        """
        Loads vocabulary from a file.
        """
        with open(vocab_file_path) as f:
            for entry in itertools.islice(f, 0, self.max_size):
                fields = entry.strip().split(sep)
                token, count = fields[0], int(fields[1])
                # dropping infrequent words
                if count >= self.min_count:
                    word_obj = self.add_word(token, count=count)
                    if match_special_symbol(token):
                        self.special_symbols[token] = word_obj

        self.add_special_symbols(DEFAULT_SPECIAL_SYMBOLS)

    def create(self, data_iterator, vocab_file_path, target_positions=[0], sep=' '):
        """
        Creates a vocabulary from textual data where target_positions is a list.
        :param target_positions: List of positions of the actual text to be considered for the vocabulary creation
        """
        temp_word_to_freq = {}
        print("Creating vocabulary...")

        for data in data_iterator:
            for t_p in target_positions:
                tokens = data[t_p]

                if not isinstance(tokens, (list, np.ndarray)):
                    tokens = [tokens]

                for token in tokens:
                    if isinstance(token, int):
                        raise TypeError("the type of the word '%s' must not be int!" % token)
                    if token == '':
                        continue
                    if token not in temp_word_to_freq:
                        temp_word_to_freq[token] = 0
                    temp_word_to_freq[token] += 1

        # populate the collectors
        for token, count in sort_hash(temp_word_to_freq, by_key=False):
            if self.max_size and len(self) >= self.max_size:
                break
            if count >= self.min_count:
                word_obj = self.add_word(token, count)
                self.total_count += count
                # if the word_obj is actually a special symbol, add it to the proper collection.
                if match_special_symbol(token):
                    self.special_symbols[token] = word_obj

        self.add_special_symbols(DEFAULT_SPECIAL_SYMBOLS)
        self.write(vocab_file_path, sep)

    def write(self, file_path, sep=' '):
        create_folders_if_not_exist(file_path)
        with open(file_path, 'w') as f:
            for word in self:
                if word.token in DEFAULT_SPECIAL_SYMBOLS:
                    continue
                try:
                    f.write(sep.join([str(word.token), str(word.count)]) + "\n")
                except:
                    #raise ValueError("could not process %s" % word.token)
                    pass
        print("Vocabulary written to " + file_path)

    def add_special_symbols(self, special_symbols):
        """
        Appending/updating of special symbols.
        """
        for token in special_symbols:
            count = self[token].count if token in self else 1
            self.special_symbols[token] = self.add_word(token, count)

    def add_word(self, token, count=1):
        """
        Adds a word to collection or update its count if it already there.
        """
        if token in self:
            word_obj = self[token]
            self.total_count += count
            self.total_count -= word_obj.count
            word_obj.count = count
        else:
            n = len(self._word_to_word_obj)
            word_obj = Word(token, id=n, count=count)
            self._word_to_word_obj[token] = word_obj
            self._id_to_word_obj.append(word_obj)
            self.total_count += count
        return word_obj

    def assign_distr(self, pow=0.75):
        # extract frequencies as an array
        counts = np.array([word.count for word in self])
        self.uni_distr = compute_distr(counts, pow)
        # dirty hack to avoid numpy's probabilies do not sum to 1
        s = sum(self.uni_distr)
        eps = 1 - s
        self.uni_distr[len(self.special_symbols)] += eps
        assert self.uni_distr is not None

    def __len__(self):
        return len(self._id_to_word_obj)

    def __contains__(self, item):
        if isinstance(item, Word):
            return item.token in self._word_to_word_obj
        if isinstance(item, str):
            return item in self._word_to_word_obj
        if isinstance(item, int):
            return len(self._id_to_word_obj) >= item + 1
        raise ValueError('input argument is not of a correct type.')

    def __iter__(self):
        for word in self._id_to_word_obj:
            yield word

    def __getitem__(self, item):
        """
        A generic get method in the latter two cases below will return UNK if word(s) are
        not in the vocabulary.
        :param item: either an integer(id) or a string(word) or a list of strings(words)
        """
        if isinstance(item, str):
            return self._word_to_word_obj[item] if item in self else self[UNK_TOKEN]
        if isinstance(item, (int, np.integer)):
            return self._id_to_word_obj[item]
        if isinstance(item, (list, np.ndarray)):
            return [self[w] for w in item]
        print(item.token)
        raise ValueError('input argument is not of a correct type.')


def compute_distr(freq, pow=0.75):
    """
    Computes and returns a unigram distributions over frequency counts.
    """
    p = freq**pow
    return p/np.sum(p, dtype="float32")


def match_special_symbol(token):
    """
    Checks whether the passed token matches the special symbols format.
    """
    return re.match(r'<[A-Z]+>', token)

def sort_hash(hash, by_key=True, reverse=True):
        if by_key:
            indx = 0
        else:
            indx = 1
        return sorted(hash.items(), key=operator.itemgetter(indx), reverse=reverse)

#maybe to use later: https://pytorch.org/text/data_utils.html#ngrams-iterator

import csv
import sys

import numpy as np

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

def get_word_vector_vocab(vocab_dict, dataset):
    '''
    This function convert the words in the dataset to word vectors using bag of word features.

    :param vocab_dict: the dictionary of the vocabulary
    :param dataset: np array containing the data of movie reviews
    :return: a list containing word vectors of each review in the dataset
    '''

    # convert the vocabulary dictionary to list for indexing in generating the word vec
    vocab_list = list(vocab_dict)
    # convert the vocabulary dictionary to set for searching in generating the word vec
    vocab_set = set(vocab_list)
    # result nx2 list containing the vectors of review data
    word_vector = []

    # iterate over the dataset to get the word vector
    for review in dataset:
        # word vector for the current review
        vec = [0]*(len(vocab_list)+1)
        vec[0] = review[0]
        word_list = review[1].split()
        for word in word_list:
            # when the word is in the vocabulary, update the word vector
            if word in vocab_set:
                idx = vocab_list.index(word)+1
                vec[idx] = 1
                continue
        # add the word vector of the current review to the result word vectors list
        word_vector.append(vec)
    return word_vector

def get_word_vector_embedding(word_embedding, dataset):
    '''
    This function computes the word vector of the review dataset using word embedding feature

    :param word_embedding: a dictionary containing the word embedding features
    :param dataset: np array containing the data of movie reviews
    :return: list of word vectors of each review
    '''
    embedding_vector = []
    for review in dataset:
        word_list = review[1].split()
        word_trimmed = {}
        # step 1 in creating the vector using word embedding
        # remove the words in the review that do not appear in word embedding dictionary
        # record the number of times the word appears
        for word in word_list:
            # if the word is already in the trimmed dictionary, increase the count by 1
            if word in word_trimmed:
                word_trimmed[word] += 1
            # if the word is not in the trimmed dictionary, check if it is in the word embedding dict
            else:
                # if the word is in the word embedding features, add the word to the trimmed dictionary
                # else, ignore the word
                if word in word_embedding:
                    word_trimmed[word] = 1

        # step 2 in creating the vector using word embedding
        vector = []
        # add the label
        vector.append(format(review[0],'.6f'))
        length = 0
        # get the overall length of the remaining words in the review
        for key in list(word_trimmed.keys()):
            length += word_trimmed[key]
        # iterate over the features in word embedding dictionary
        for i in range(MAX_WORD_LEN):
            feature = 0
            # sum over the ith feature of all words in the trimmed dictionary times the occurence of the word
            for word in word_trimmed:
                feature += word_trimmed[word]*word_embedding[word][i]
            feature = format(feature/length,'.6f')
            # add the feature into the word vector
            vector.append(feature)
        # add the word vector to the result list
        embedding_vector.append(vector)
    return embedding_vector

def write_file(data, path):
    '''
    Write the word vectors to a tab deliminated file

    :param data: a list containing word vectors
    :param path: output file path
    '''
    with open(path, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for row in data:
            tsv_output.writerow(row)

if __name__ == "__main__":
    flag = int(sys.argv[9])
    # load the training set
    train_set = load_tsv_dataset(sys.argv[1])
    # load the validation set
    valid_set = load_tsv_dataset(sys.argv[2])
    # load the test set
    test_set = load_tsv_dataset(sys.argv[3])
    if flag == 1:
        dict_path = sys.argv[4]

        # load the vocabulary into a dictionary
        vocab_dict = load_dictionary(dict_path).keys()

        # get the training, validation, test word vectors using bag of words features
        train_vector = get_word_vector_vocab(vocab_dict, train_set)
        valid_vector = get_word_vector_vocab(vocab_dict, valid_set)
        test_vector = get_word_vector_vocab(vocab_dict, test_set)

        # write the word vectors to the output file
        write_file(train_vector, sys.argv[6])
        write_file(valid_vector, sys.argv[7])
        write_file(test_vector, sys.argv[8])
    elif flag == 2:
        # load the word embedding dictionary
        word_embedding = load_feature_dictionary(sys.argv[5])

        # compute the word vector for train, validation, test set using word embedding
        train_embedding = get_word_vector_embedding(word_embedding, train_set)
        valid_embedding = get_word_vector_embedding(word_embedding, valid_set)
        test_embedding = get_word_vector_embedding(word_embedding, test_set)

        # write the word vectors to the output file
        write_file(train_embedding, sys.argv[6])
        write_file(valid_embedding, sys.argv[7])
        write_file(test_embedding, sys.argv[8])
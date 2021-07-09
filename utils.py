import random

import numpy as np
import pretty_midi
import music21
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json

base_dir = os.path.dirname(os.path.abspath(__file__))

# Some constant values for further process
chord_dict_path = os.path.join(base_dir, 'dictionaries/chord_dict.npy')
chord_dict_clf_path = os.path.join(base_dir,'dictionaries/chord_dict_clf.npy')
note_dict_path = os.path.join(base_dir,'dictionaries/note_dict.npy')
key_dict_path = os.path.join(base_dir,'dictionaries/key_dict.npy')

with open(os.path.join(base_dir,"lists/adventure.json"), "r") as fp:
    adventure = json.load(fp)
with open(os.path.join(base_dir,"lists/fighting.json"), "r") as fp:
    fighting = json.load(fp)
with open(os.path.join(base_dir,"lists/puzzle.json"), "r") as fp:
    puzzle = json.load(fp)
with open(os.path.join(base_dir,"lists/sport.json"), "r") as fp:
    sport = json.load(fp)
with open(os.path.join(base_dir,"lists/shooting.json"), "r") as fp:
    shooting = json.load(fp)
with open(os.path.join(base_dir,"lists/gen_list.json"), "r") as fp:
    gen_list = json.load(fp)

label_lists = adventure + fighting + puzzle + sport + shooting

# Function used for loading both the dictionaries
def load_dictionaries():

    # Reading from file
    try:
        chord_dict = np.load(chord_dict_path, allow_pickle='TRUE').item()  # Load the dictionary
        musical_dict = np.load(note_dict_path, allow_pickle='TRUE').item()  # Load the dictionary
    except:
        chord_dict = {}
        musical_dict = {}
    finally:
        # Obtaining the inverted dicts for inverse translation
        inverted_dicts = []
        for d in [chord_dict, musical_dict]:
            inverted_dict = {}
            for e in d:
                key = d[e]
                value = e
                inverted_dict[key] = value
            inverted_dicts.append(inverted_dict)

    return chord_dict, musical_dict, inverted_dicts[0], inverted_dicts[1]

# It gets a midi object and returns a normalized piano roll matrix with values 0 or 1
def get_normalized_piano_roll_matrix(midi_instance, fs=100):
    normalized_piano_roll_matrix = midi_instance.get_piano_roll(fs=fs)  # Piano roll matrix size: 128 x times.shape[0]
    normalized_piano_roll_matrix[normalized_piano_roll_matrix > 0.] = 1.

    return normalized_piano_roll_matrix

# It gets a midi object and returns the piano roll matrix with an extra row representing the pause (1 if present)
def get_paused_piano_roll_matrix(midi_instance, fs=100):
    normalized_piano_roll_matrix = get_normalized_piano_roll_matrix(midi_instance, fs)

    num_frames = normalized_piano_roll_matrix.shape[1]
    new_row = np.zeros(num_frames)

    normalized_piano_roll_matrix = np.vstack([normalized_piano_roll_matrix, new_row])  # Add the new row

    # Assign the correct values in the last row (1 if a pause is present, 0 otherwise)
    for i in range(num_frames):
        if sum(normalized_piano_roll_matrix[:, i]) == 0:
            normalized_piano_roll_matrix[128, i] = 1

    return normalized_piano_roll_matrix

# It returns a list of arrays, where each array contains a set of indexes of the normalized_piano_roll_matrix
def compress_piano_roll_matrix(midi_instance, fs=100, pause=False):

    if pause == False:
        normalized_piano_roll_matrix = get_normalized_piano_roll_matrix(midi_instance, fs=fs)
    else:
        normalized_piano_roll_matrix = get_paused_piano_roll_matrix(midi_instance, fs=fs)

    num_frames = normalized_piano_roll_matrix.shape[1]
    active_notes = []

    for i in range(num_frames):
        active_notes.append(np.where(normalized_piano_roll_matrix[:, i] == 1))

    return active_notes

# It gets a list of values and returns a list of pairs of the type <value, count>
def convert_sequence(sequence):
    current_value = sequence[0]
    count = 0
    output_list = []

    for i in range(len(sequence)):
        if current_value == sequence[i]:
            count += 1
        else:
            output_list.append((current_value, count))
            current_value = sequence[i]
            count = 1

        if i == (len(sequence) - 1):
            output_list.append((current_value, count))

    return output_list

# From list of pairs to list of strings
def from_pairs_to_strings(sequence):
    string_list = []

    for i in range(len(sequence)):
        current_string = str(sequence[i])  # Convert the current element as a string
        current_string = current_string.replace(',', '')  # Remove the ',' character from the string
        string_list.append(current_string)

    return string_list

# From list of strings to dictionary
def from_list_to_dictionary(string_list, start=0):
    try:
        dictionary = np.load(note_dict_path, allow_pickle='TRUE').item()  # Load the dictionary (if present)
    except Exception as e:
        dictionary = {}  # Create a new dictionary (if the file does not exist yet)

    # Update the dictionary if needed
    for i in range(len(string_list)):
        if string_list[i] not in dictionary:
            if len(dictionary) > 0:
                dictionary[string_list[i]] = max(dictionary.values()) + 1  # New sequential value
            else:
                dictionary[string_list[i]] = start  # Start from 0

    np.save(note_dict_path, dictionary)  # Save dictionary

    return dictionary

# Term frequency function
def term_frequency(X, len_dictionary=20000):
  term_frequency_matrix = []

  for x in X:
    current_item = np.zeros(len_dictionary)
    for i in x:
      current_item[i] += 1

    term_frequency_matrix.append(current_item)

  term_frequency_matrix = np.matrix(term_frequency_matrix)

  return term_frequency_matrix

# Function used for reweight a probability distribution
def reweight_distribution(original, temperature=0.3):
    dist = np.asarray(original).astype('float64')
    dist = np.log(dist) / temperature
    dist = np.exp(dist)
    dist = dist / np.sum(dist)
    dist = np.random.multinomial(1, dist[0])
    return dist

# Check if a file is one of those of the selected games
def is_present(file_name, search_list=label_lists):
    present = False
    for name in search_list:
        if name in file_name:
            present = True
    return present

# Function used to assign a label with respect to the file name
def assign_label(file_name):

    if is_present(file_name, adventure):
        return 0
    elif is_present(file_name, sport):
        return 1
    elif is_present(file_name, fighting):
        return 2
    elif is_present(file_name, shooting):
        return 3
    elif is_present(file_name, puzzle):
        return 4

# Function used for loading the classifier training and test set
def load_clf_data_kfold(sample_len, fs, k=10, feature_extraction = "standard"):

    # Declaring test and train sets
    folds_X, folds_y = [], []

    # Trying to read the chord dict
    try:
        chord_dict = np.load(chord_dict_clf_path, allow_pickle='TRUE').item()  # Load the dictionary (if present)
    except:
        chord_dict = {}

    file_list = []
    label_distrib = np.zeros(shape=(5,))
    # Reading the midi files
    for file in os.listdir(os.path.join(base_dir, 'dataset')):
        if is_present(file):
            file_list.append(file)
            label_distrib[assign_label(file)] += 1

    #random.shuffle(file_list)
    print(len(file_list))
    print(label_distrib)

    fold_dim = int(len(file_list)/k)
    # print(file_list)

    folds = []
    for i in range(k):
        curr_fold = []
        for j in range(fold_dim):
            curr_fold.append(file_list[i*fold_dim + j])
        folds.append(curr_fold)

    # Iteration over folds
    for j in range(k):

        # Current fold X and y
        curr_X, curr_y = [], []

        print('loading data file fold {}...'.format(j))
        # Iteration over fold j
        for file in folds[j]:

            filepath = os.path.join('dataset', file)
            midi_data = pretty_midi.PrettyMIDI(os.path.join(base_dir, filepath))

            # Computing each chord
            piano_roll = []
            for elem in compress_piano_roll_matrix(midi_data, fs=fs):
                chord = str(elem[0])
                if chord not in chord_dict:
                    chord_dict[chord] = len(chord_dict)
                piano_roll.append(chord_dict[chord])

            # Adding each sample of the song to the training or test set
            temp_X, temp_y = extract_features(piano_roll, sample_len, file, feature_extraction)
            curr_X.extend(temp_X)
            curr_y.extend(temp_y)
            # for i in range(int(len(piano_roll) / sample_len)):
            #     curr_sample = piano_roll[(i * sample_len):((i + 1) * sample_len)]
            #     curr_label = assign_label(file)
            #     curr_X.append(curr_sample)
            #     curr_y.append(curr_label)

        # Converting into arrays the data
        print(np.array(curr_X))
        print(np.array(curr_y))
        curr_X, curr_y = np.array(curr_X), np.array(curr_y)
        # print(curr_X.shape)
        # curr_X = term_frequency(curr_X)
        # print(curr_X.shape)
        # Adding the fold j
        folds_X.append(curr_X)
        folds_y.append(curr_y)

    np.save(chord_dict_clf_path, chord_dict)  # Saving the dictionary

    print(len(folds_X))

    # x_train = np.concatenate((folds_X[0], folds_X[1], folds_X[2], folds_X[3], folds_X[4], folds_X[5]), axis=0)
    # y_train = np.concatenate((folds_y[0], folds_y[1], folds_y[2], folds_y[3], folds_y[4], folds_y[5]), axis=0)
    # x_train, y_train = folds_X[1], folds_y[1]
    # x_test, y_test = folds_X[6], folds_y[6]

    return folds_X, folds_y, chord_dict
    # return x_train, x_test, y_train, y_test, chord_dict

# Function used for loading the classifier training and test set
def load_clf_data(sample_len, fs):

    # Declaring test and train sets
    x_train, x_test, y_train, y_test = [], [], [], []

    # Trying to read the chord dict
    try:
        chord_dict = np.load(chord_dict_clf_path, allow_pickle='TRUE').item()  # Load the dictionary (if present)
    except:
        chord_dict = {}

    train_vs_test_index = 0

    # Reading the midi files
    for file in os.listdir('dataset'):
        if is_present(file):
            print('loading data file...')
            train_vs_test_index = (train_vs_test_index + 1) % 2
            midi_data = pretty_midi.PrettyMIDI(os.path.join('dataset', file))

            # Computing each chord
            piano_roll = []
            for elem in compress_piano_roll_matrix(midi_data, fs=fs):
                chord = str(elem[0])
                if chord not in chord_dict:
                    chord_dict[chord] = len(chord_dict)
                piano_roll.append(chord_dict[chord])

            # Adding each sample of the song to the training or test set
            for i in range(int(len(piano_roll) / sample_len)):
                curr_sample = piano_roll[(i * sample_len):((i + 1) * sample_len)]
                curr_label = assign_label(file)
                if train_vs_test_index == 1:
                    x_train.append(curr_sample)
                    y_train.append(curr_label)
                else:
                    x_test.append(curr_sample)
                    y_test.append(curr_label)

    # Converting into arrays the data
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # print(x_train.shape)
    # x_train_0, y_train_0 = x_train[np.where(y_train == 0)], y_train[np.where(y_train == 0)]
    # x_train_1, y_train_1 = x_train[np.where(y_train == 1)], y_train[np.where(y_train == 1)]
    # x_train_2, y_train_2 = x_train[np.where(y_train == 2)], y_train[np.where(y_train == 2)]
    # x_train_3, y_train_3 = x_train[np.where(y_train == 3)], y_train[np.where(y_train == 3)]
    # x_train_4, y_train_4 = x_train[np.where(y_train == 4)], y_train[np.where(y_train == 4)]
    # cut_len = min(x_train_0.shape[0], x_train_1.shape[0], x_train_2.shape[0],x_train_3.shape[0],x_train_4.shape[0])
    # y_train = np.ones(shape=(cut_len*5, 100))
    # x_train_0, y_train_0 = x_train_0[:cut_len], y_train_0[:cut_len]
    # x_train_1, y_train_1 = x_train_1[:cut_len], y_train_1[:cut_len]
    # x_train_2, y_train_2 = x_train_2[:cut_len], y_train_2[:cut_len]
    # x_train_3, y_train_3 = x_train_3[:cut_len], y_train_3[:cut_len]
    # x_train_4, y_train_4 = x_train_4[:cut_len], y_train_4[:cut_len]
    # print(cut_len)
    # x_test_0, y_test_0 = x_test[np.where(y_test == 0)], y_test[np.where(y_test == 0)]
    # x_test_1, y_test_1 = x_test[np.where(y_test == 1)], y_test[np.where(y_test == 1)]
    # x_test_2, y_test_2 = x_test[np.where(y_test == 2)], y_test[np.where(y_test == 2)]
    # x_test_3, y_test_3 = x_test[np.where(y_test == 3)], y_test[np.where(y_test == 3)]
    # x_test_4, y_test_4 = x_test[np.where(y_test == 4)], y_test[np.where(y_test == 4)]
    # cut_len = min(x_train_0.shape[0], x_train_1.shape[0], x_train_2.shape[0], x_train_3.shape[0], x_train_4.shape[0])
    # y_test = np.ones(shape=(cut_len*5, 100))
    # x_test_0, y_test_0 = x_test_0[:cut_len], y_test_0[:cut_len]
    # x_test_1, y_test_1 = x_test_1[:cut_len], y_test_1[:cut_len]
    # x_test_2, y_test_2 = x_test_2[:cut_len], y_test_2[:cut_len]
    # x_test_3, y_test_3 = x_test_3[:cut_len], y_test_3[:cut_len]
    # x_test_4, y_test_4 = x_test_4[:cut_len], y_test_4[:cut_len]
    #
    # x_train = np.concatenate((x_test_0, x_test_1, x_test_2, x_test_3, x_test_4), axis=0)
    # x_test = np.concatenate((x_train_0, x_train_1, x_train_2, x_train_3, x_train_4), axis=0)
    # y_train = np.concatenate((y_test_0, y_test_1, y_test_2, y_test_3, y_test_4), axis=0)
    # y_test = np.concatenate((y_train_0, y_train_1, y_train_2, y_train_3, y_train_4), axis=0)

    np.save(chord_dict_clf_path, chord_dict)  # Saving the dictionary

    return x_train, x_test, y_train, y_test, chord_dict

# Function used for loading the dataset form midi files
def load_data(fix_len=25, instrument=1, fs=50, convert=True, complete_partition=False, attention=False, legay=False):

    samples = []
    labels = []
    try:
        chord_dict = np.load(chord_dict_path, allow_pickle='TRUE').item()  # Load the dictionary (if present)
        key_dict = np.load(key_dict_path, allow_pickle='TRUE').item()  # Load the dictionary (if present)
    except:
        chord_dict = {}
        key_dict = {}

    # Listing the midi files
    for file in os.listdir('dataset'):
        # Loading the desired files
        if is_present(file, gen_list):
            print('loading data file...')
            midi_data = pretty_midi.PrettyMIDI(os.path.join('dataset', file))

            if file not in key_dict:
                score = music21.converter.parse(os.path.join('dataset', file))
                key = score.analyze('key')
                key_dict[file] = key
            else:
                key = key_dict[file]

            # Loading only the files with 3 instruments
            if len(midi_data.instruments) >= 3 and key.name == 'C major':

                # Choosing between single instrument or full score loading
                piano_roll = []
                if complete_partition:
                    compressed_piano_roll = compress_piano_roll_matrix(midi_data, fs=fs)
                else:
                    compressed_piano_roll = compress_piano_roll_matrix(midi_data.instruments[instrument], fs=fs)

                # Computing each chord on the piano roll
                for elem in compressed_piano_roll:
                    chord = str(elem[0])
                    if chord not in chord_dict:
                        chord_dict[chord] = len(chord_dict)
                    piano_roll.append(chord_dict[chord])

                if convert:
                    if not legay:
                        piano_roll = from_pairs_to_strings(convert_sequence(piano_roll))

                    if attention:
                        dict = from_list_to_dictionary(piano_roll, start=1)
                    else:
                        dict = from_list_to_dictionary(piano_roll, start=0)

                    for i, p in enumerate(piano_roll):
                        piano_roll[i] = dict[p]

                if attention:
                    # Using the attention mechanism there is an explicit masking
                    for i in range(len(piano_roll)-1):
                        curr_sample = piano_roll[:(i+1)]
                        curr_label = piano_roll[i+1]
                        samples.append(curr_sample)
                        labels.append(curr_label)
                else:
                    # Without the attention we need the same sample size
                    for i in range(len(piano_roll)-fix_len):
                        curr_sample = piano_roll[i:(i+fix_len)]
                        curr_label = piano_roll[i+fix_len]
                        samples.append(curr_sample)
                        labels.append(curr_label)

    samples, labels = np.array(samples), np.array(labels)
    np.save(chord_dict_path, chord_dict)
    np.save(key_dict_path, key_dict)

    samples = tf.keras.preprocessing.sequence.pad_sequences(samples, padding="post", dtype='float32')
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, padding="post", dtype='float32')
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, padding="post", dtype='float32')
    feed_shape = x_train[0].shape

    return x_train, x_test, y_train, y_test, feed_shape

# This function generate a midi file from the given piano midi
def from_piano_roll_to_midi(piano_roll, dictionary, fs, single_column=False, legacy=False, path='generated.midi'):

    # Creating the piano midi
    midi_file = pretty_midi.PrettyMIDI()

    # 81 and 86 are a high and mid synth wave sound, 39 in a synth bass
    instruments = [81, 86, 39]
    range_dim = 3
    if single_column:
        range_dim = 1
    for i in range(range_dim):

        # Creating an instrument with the given sound timber
        instrument = pretty_midi.Instrument(program=instruments[i])

        start = 0
        notes = []

        if single_column:
            current_piano_roll = piano_roll
        else:
            print('current i: {}'.format(i))
            current_piano_roll = piano_roll[i]

        # Constructing the midi for a single voice
        for element in current_piano_roll:

            if element == '[]':
                print("EMPTY!")
            else:
                # Todo Remove this!
                # measure_test = True
                # if measure_test:
                #     duration = 1
                #     chord = dictionary[element]
                #     chord = chord[1:-1]
                #     print(chord)
                print(element)
                elements = element[1:-1].split(' ')

                print('---')
                print(elements)
                if legacy:
                    duration = 1
                    chord = element
                else:
                    duration = int(elements[1])             # note duration
                    chord = dictionary[int(elements[0])]    # note chord in form of string (i.e. "[61 62]")
                chord = chord[1:-1]                         # removing the parenthesis from the string

                # Only if the chord is not empty (i.e. it was a "pause")
                if chord:
                    if legacy:
                        note_list = [int(s) for s in chord.split(' ')]
                    else:
                        note_list = [int(s) for s in chord.split(' ')]      # Converting the string to list of int

                    print(note_list)
                    # Adding each note of the chord
                    for n in note_list:
                        print('Note: {}'.format(n))
                        curr_note = pretty_midi.Note(1, n, start/fs, (start+duration)/fs)
                        notes.append(curr_note)

            # Updating the next frame
            start = start + duration

        # Adding the note list to the instrument and adding it to the midi file
        instrument.notes = notes
        midi_file.instruments.append(instrument)

    # Generating a midi file
    midi_file.write(path)

# This function is used for generating the core new song
def generate_piano_roll(model, example, gen_len=250, single_column=False, attention=False, attention_factor=1):

    # Three piano rolls one for each voice
    piano_roll1, piano_roll2, piano_roll3 = [], [], []
    original_song = []

    # if single_column:
    #     ex = np.array([example[example != 0]], dtype='uint16')
    #     for x in ex[0]:
    #         piano_roll1.append(int(x))
    #         original_song.append(int(x))
    # else:
    #     # Populating the piano rolls with the initial melody
    #     for x in example[0][0]:
    #         piano_roll1.append(x)
    #         original_song.append(x)
    #     for x in example[1][0]:
    #         piano_roll2.append(x)
    #     for x in example[2][0]:
    #         piano_roll3.append(x)

    # if attention:
    #     example = [example[example != 0]]

    print(example)
    predictions = model.model.predict(example)

    if single_column:
        pred1 = np.argmax(reweight_distribution(predictions))
        piano_roll1.append(pred1)
    else:
        # Adding predictions to the piano roll
        pred1 = np.argmax(reweight_distribution(predictions[0]))           # First instrument (main voice)
        pred2 = np.argmax(reweight_distribution(predictions[1]))           # Second instrument (harmony 1)
        pred3 = np.argmax(reweight_distribution(predictions[2]))           # Third instrument (harmony 2)

        piano_roll1.append(pred1)
        piano_roll2.append(pred2)
        piano_roll3.append(pred3)

    for i in range(gen_len-1):

        if single_column:
            if attention:
                attention_dim = 648 #Todo: qui ci va la grandezza del dict
                print(example)
                example = [example[example != 0]]
                if len(example[0]) == attention_dim:
                    example = np.concatenate((example[0][1:], np.array([pred1])), axis=0)
                else:
                    example = np.concatenate((example[0], np.array([pred1])), axis=0)
                current_dim = example.shape[0]
                print(current_dim)
                example = np.pad(example, (0, attention_dim-current_dim), 'constant', constant_values=(0, 0))
                example = np.array([example])

            else:
                example = np.concatenate((example[0][1:], np.array([pred1])), axis=0)
                example = np.array([example])

        else:
            # Shifting voice themes
            voice1, voice2, voice3 = example[0][0], example[1][0], example[2][0]
            voice1 = np.concatenate((voice1[1:], np.array([pred1])), axis=0)
            voice2 = np.concatenate((voice2[1:], np.array([pred2])), axis=0)
            voice3 = np.concatenate((voice3[1:], np.array([pred3])), axis=0)
            example = [np.array([voice1]), np.array([voice2]), np.array([voice3])]

        # Predicting
        predictions = model.model.predict(example)

        if single_column:
            pred1 = np.argmax(reweight_distribution(predictions))
            piano_roll1.append(pred1)
        else:
            # Adding predictions to the piano roll
            pred1 = np.argmax(reweight_distribution(predictions[0]))           # First instrument (main voice)
            pred2 = np.argmax(reweight_distribution(predictions[1]))           # Second instrument (harmony 1)
            pred3 = np.argmax(reweight_distribution(predictions[2]))           # Third instrument (harmony 2)

            piano_roll1.append(pred1)
            piano_roll2.append(pred2)
            piano_roll3.append(pred3)

    if single_column:
        to_return = piano_roll1
    else:
        to_return = [piano_roll1, piano_roll2, piano_roll3]

    return to_return, original_song

def extract_standard_feature(piano_roll, sample_len, file):
    curr_X, curr_y = [], []
    for i in range(int(len(piano_roll) / sample_len)):
        curr_sample = piano_roll[(i * sample_len):((i + 1) * sample_len)]
        curr_label = assign_label(file)
        curr_X.append(curr_sample)
        curr_y.append(curr_label)
    return curr_X, curr_y

def walk_forward_feature_engineering(piano_roll, sample_len, file):
    curr_X, curr_y = [], []
    for i in range(0, len(piano_roll)-sample_len):
        curr_sample = piano_roll[i:i + sample_len]
        curr_label = assign_label(file)
        curr_X.append(curr_sample)
        curr_y.append(curr_label)
    return curr_X, curr_y

def full_standard_feature_engineering(piano_roll, sample_len, file):
    curr_X, curr_y = [], []
    for i in range(int(len(piano_roll) / sample_len)):
        #print(f"Slicing item in piano from from {i*sample_len} to {(i+1)*sample_len}")
        curr_sample = piano_roll[(i * sample_len):((i + 1) * sample_len)]
        curr_label = assign_label(file)
        curr_X.append(curr_sample)
        curr_y.append(curr_label)
        #print(i, curr_sample)
    if len(piano_roll) % sample_len != 0 and not len(piano_roll) < 50:
        last_sample = piano_roll[-(sample_len + 1):-1]
        last_label = assign_label(file)
        if len(last_sample) < 50:
            print(file)
        curr_X.append(last_sample)
        curr_y.append(last_label)
    return curr_X, curr_y

#this function is used for the features engineering, it allows 3 kind of extraction types:
# 1) standard -> split the piano roll in len(piano_roll)//sample_len parts and get the sample_len roll for each parts
# 2) full-standard -> since the standard part will cut the last piece of piano_roll here we include them
# 3) walk-forward -> we move of one time-frame for cycle and took the window of sample_len next elements
def extract_features(piano_roll, sample_len, file, extraction_type="standard"):
    curr_X, curr_y = [], []
    if extraction_type == "standard":
        curr_X, curr_y = extract_standard_feature(piano_roll, sample_len, file)
    elif extraction_type == "walk-forward":
        curr_X, curr_y = walk_forward_feature_engineering(piano_roll, sample_len, file)
    elif extraction_type == "full-standard":
        curr_X, curr_y = full_standard_feature_engineering(piano_roll, sample_len, file)
    return curr_X, curr_y
from deep_nes import DeepNES
from attention_nes import AttentionNES
from utils import *
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


single_column_test = True      # If we want all the instrument at the same time
fs = 10                        # Frame sampling rate
initial_attention = 10         # Number of elements used ad X

if single_column_test:
    x_train, x_test, y_train, y_test, feed_shape = load_data(fs=fs,
                                                             fix_len=initial_attention,
                                                             complete_partition=True,
                                                             attention=True)

else:
    # Loading the training and test set for every instrument
    x_train1, x_test1, y_train1, y_test1, feed_shape1 = load_data(instrument=0, fix_len=initial_attention, fs=fs)
    x_train2, x_test2, y_train2, y_test2, feed_shape2 = load_data(instrument=1, fix_len=initial_attention, fs=fs)
    x_train3, x_test3, y_train3, y_test3, feed_shape3 = load_data(instrument=2, fix_len=initial_attention, fs=fs)

    # Chooding the minimum dimension for not exceeding the tensor size
    dim_train = np.min([len(y_train1), len(y_train2), len(y_train3)])
    dim_test = np.min([len(y_test1), len(y_test2), len(y_test3)])
    dimension = min(dim_test, dim_train)

    # Resizing the entire dataset
    x_train1, y_train1 = x_train1[:dimension], y_train1[:dimension]
    x_train2, y_train2 = x_train2[:dimension], y_train2[:dimension]
    x_train3, y_train3 = x_train3[:dimension], y_train3[:dimension]
    x_train, y_train = [x_train1, x_train2, x_train3], [y_train1, y_train2, y_train3]

    # Resizing the test set
    x_test1, y_test1 = x_test1[:dimension], y_test1[:dimension]
    x_test2, y_test2 = x_test2[:dimension], y_test2[:dimension]
    x_test3, y_test3 = x_test3[:dimension], y_test3[:dimension]
    x_test, y_test = [x_test1, x_test2, x_test3], [y_test1, y_test2, y_test3]

    feed_shape = [feed_shape1, feed_shape2, feed_shape3]

# Loading the two dictionaries for the further conversions
chord_dict, musical_dict, chord_dict_inverted, musical_dict_inverted = load_dictionaries()
len_dict = len(musical_dict)
inverted_dict = musical_dict_inverted

# Choosing the correct model with respect to the single or multiple column
if single_column_test:
    model = AttentionNES(feed_shape, vocabulary_size=len_dict)
else:
    model = DeepNES(feed_shape, vocabulary_sizes=[len_dict, len_dict, len_dict])

# Choosing whether training or testing the model
model.load_model()

# Preparing an example with the beginning of the first sample
ran = np.random.randint(100)
if single_column_test:
    example = np.array([x_test[ran]])
else:
    example = [np.array([x_train1[ran]]), np.array([x_train2[ran]]), np.array([x_train3[ran]])]

# Generating the music
music, original = generate_piano_roll(model, example, single_column=single_column_test, attention=single_column_test)

# Converting each value of the generated music matrix
if single_column_test:
    for i, e in enumerate(music):
        supp = inverted_dict[e]
        music[i] = supp
else:
    for k in range(3):
        for i, e in enumerate(music[k]):
            supp = inverted_dict[e]
            music[k][i] = supp

# Converting the dict words into the real chords
for i, e in enumerate(original):
    supp = inverted_dict[e]
    original[i] = supp

# Creating the midi file
from_piano_roll_to_midi(music,
                        dictionary=chord_dict_inverted,
                        fs=fs,
                        single_column=single_column_test,
                        path='generated.midi')





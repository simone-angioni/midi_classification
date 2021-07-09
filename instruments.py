from numpy.lib.histograms import histogram
import pretty_midi as pm
import os, json, pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
midi_classification = {}
instrument_histogram = {}
for file in os.listdir(dataset_dir):

    if file.endswith(".mid"):
        print(f"Working of file {file}", end="\r")
        filename = os.path.join(dataset_dir, file)
        try:
            current_midi = pm.PrettyMIDI(filename)
        except ValueError:
            current_midi = pm.PrettyMIDI()
        midi_classification[file] = []
        for instrument in current_midi.instruments:
            prg = int(instrument.program)
            name = pm.program_to_instrument_name(instrument.program)
            midi_classification[file].append({'program': prg,'name': name})
            if not prg in instrument_histogram:
                instrument_histogram[prg] = {}
                instrument_histogram[prg]['name'] = name
                instrument_histogram[prg]['occurrences'] = 0
            instrument_histogram[prg]['occurrences'] += 1
    # print(midi_classification)
    # print(instrument_histogram)

with open("./dataset_instruments/all_midi.json", "w") as f:
    json.dump(midi_classification, f)

with open("./dataset_instruments/instrument_histogram.json", "w") as f:
    json.dump(instrument_histogram, f)

to_csv = []
for prg in instrument_histogram:
    to_csv.append(instrument_histogram[prg])

df = pd.DataFrame(to_csv)
df.to_excel("./dataset_instrument/instrument_histogram.xlsx")
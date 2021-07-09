# NESMusicGenerator
Generatinon and Classification of NES midi music files using deep learning techniques.

We evaluate our model using a k-fold validation with k=10 and we use the term frequecies for encoding the musical samples.

## Requirements
You need a version of Python 3 installed on your machine.

## Run the experiments
For running the experiment it is sufficient to install the `requirements.txt` and then run either the `main_classifier.py` or the `main_generator.py`.
The main generator will generate a midi file; the main classifier will train the model for 100 epochs and will show the plot results.

```
git clone https://github.com/EmanueleLedda97/NESMusicGenerator
cd NESMusicGenerator
pip install -r requirements.txt
python main_classifier.py
```

After doing this the program will be succesfully ran.

## Troubleshooting
If after the installation of the requirements the code does not run, it is possible to install all the main libraries by using the file `install.sh`. 

```
bash install.sh
```

If you want to manually install everything it can be done by `pip` (or `pip3` if the first does not work).
The dependencies comprehend `tensorflow`, `pandas`, `numpy`, `matplotlib`, `pretty_midi` and `music21`.

# Music Prediction
Predicting music catchiness with machine learning by mimicking human behaviour.

## Prerequisites
- [Matlab version 7 or higher](https://www.mathworks.com/products/matlab.html "Matlab")
- [The MatLab MIRtoolbox](https://www.jyu.fi/hytk/fi/laitokset/mutku/en/research/materials/mirtoolbox "MIRtoolbox")
- [Python version 3.6](https://www.python.org/downloads/ "Python")
- Numpy (install with pip)
- SciPy (install with pip)
- scikit-learn (install with pip)

## Running the program
Firstly, to remove any unprocessable files, run the following in MatLab:
<pre>
find_bad_files(<i>music_folder_name</i>)
</pre>
This makes a file named bad_files.txt, which contains all unprocessable files.
Now run the following in the command prompt:
```
python3 filter_music.py
```
Then run this in MatLab, while in the folder containing the music files:
```
struct = mirfeatures('Folder', 'Stat')
mirexport('music_features.txt', struct)
```
Now run the main file in the command prompt:
<pre>
python3 main.py <i>participant_data.csv</i> music_features.txt <i>n_folds</i>
</pre>

## Authors
- Dries Fransen
- Geerten Rijsdijk
- Hannah Min
- Ko Schoemaker
- Stefan Klut

## Acknowledgements
- John Ashley Burgoyne (ILLC)
- Victor Milewski (TA)

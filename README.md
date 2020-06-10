# LSTM - MELODY GENERATOR
Python/Tensorflow based Long-Short-Term-Memory Recurrent Network Implementation for Music Melody Generation

## Install
run from terminal---> $pip install -r ./requirements.txt

Download and install MuseScore for play and edit symbolic music represented data
https://musescore.org/en

## Usage
Inside deutschl/test theres is a small set of data for testing.
To create a better and deeper model Download and unzip from ESAC dataset http://www.esac-data.org/ (or similar)  with the dataset of your intersest

Run preprocess.py
(data loading, homogeneous note transposition, music notation to numeric time series and one hot encoding representation)

Run train.py
(build train and compile the RNN-LSTM  network)

Run melody_generator.py
(generate the model, tweak the parameters, especially "temperature"-ie the degree of creativity/unpredictability of the model- and create your own melody




Data and general guidance from Valerio Velardo (The Sound of AI) https://github.com/musikalkemist/

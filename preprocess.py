import os
import music21 as m21
import json
import numpy as np
import tensorflow.keras as keras


m21.environment.set('musicxmlPath', '/usr/bin/musescore')


KERN_DATASET_PATH="./deutschl/test"
SAVE_DIR =  "./dataset"
SINGLE_FILE_DATASET="./single_file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = "./mapping.json"



# durations expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]

# kern, MIDI, MusicXML <-> m21 #

def load_songs_in_kern(dataset_path):
    # parse and load pieces in the dataset with music21

    songs=[]
    for path,subdir,files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song =  m21.converter.parse(os.path.join(path,file))
                songs.append(song)

    return songs

def duration_check(song, ACCEPTABLE_DURATIONS):
    for note in song.flat.notesAndRests:    #flatten the object notes into one list
        if note.duration.quarterLength not in ACCEPTABLE_DURATIONS:
            return False
        return True

def encode_song(song, time_step = 0.25):
    encoded_song = []
    for event in song.flat.notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        #handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        #conver note/rest into time series notation
        steps = int(event.duration.quarterLength/ time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str,encoded_song))

    return encoded_song



def transpose(song):
   #Transposes song to C maj/A min

    # get song key
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure) #measure and bars are synonims
    key = measures_part0[0][4]  #the 4th element is tipically the key in m21 encoding

    # estimate key  if not present using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song


def preprocess(dataset_path):
    pass

    # load the folk song
    print("loading songs..")
    songs = load_songs_in_kern(dataset_path)
    print(f"{len(songs)} Song Loaded")

    #filter out songs by duration
    for i,song in enumerate(songs):
        if not duration_check(song, ACCEPTABLE_DURATIONS):
            continue
        #transpose song to Cmaj/Amin
        song = transpose(song)

        # encode songs with music time series representation
        encoded_song = encode_song(song)

         # save songs to .txt
        save_path = os.path.join(SAVE_DIR,str(i))

        with open(save_path, "w") as fp:
            fp.write(encoded_song)

def load(file_path):
    with open(file_path,"r") as fp:
        song = fp.read()

        return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded song and delimiters
    for path, _,files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path,file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    songs = songs[:-1] # clean last delimiter space

    # save one string containing all dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs

def create_mapping(songs,mapping_path):
    mappings = { }

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    #create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save vocabulary to json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings,fp, indent = 4)


def convert_songs_to_int(songs):
    int_songs = []

    # load the mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # cast song strings to list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generate_training_sequences(sequence_length):
    # example with sequence_length=2 ---> int_song=[11, 12, 13, ...] -> i1: [11,12],t1:13 | i2:[12, 13] , t2:14
    songs = load(SINGLE_FILE_DATASET)
    # load songs and map them to int
    int_songs = convert_songs_to_int(songs)

    # generate training sequences
    # dataset with 100 symbol, seq length =64----> 100-64 = 36 sequences will be considered
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    # one-hot encode the sequences
    # inputs dimension before one hot encoding: (# of sequences, sequence_length)
    # example  [ [0,1,2] , [1,1,2] ]  ||| values can be equal to 0,1,2
    # one hot encode [ [ [ 1,0,0 ] , [0,1,0], [0,0,1]] , [ [0,1,0], [0,1,0],[0,0,1] ] ]
    # so after encoding input dimension: (# of sequences, sequence_length, vocabulary size)
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs,num_classes=vocabulary_size)

    targets = np.array(targets)
    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)



if __name__ == "__main__":
    main()





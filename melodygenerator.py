import json
import tensorflow.keras as keras
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import numpy as np
import music21 as m21


class MelodyGenerator:

    def __init__(self, model_path = "./model.h5"):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def _sample_with_temperature(self,probabilities,temperature):

        # temperature = 0 ---> p is reshaped such that maximum incoming probability is mapped to 1 and all the other to 0 - DETERMINISTIC
        # temperature = 1 ---> as if temperature was not used
        # temperature = INF ---> p is reshaped as with all indexes with same probability (i.e. all the network training would be useless)
        # temperature thus helps to let the net be more, or less, creative/unpredictable
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p =probabilities)

        return index

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

        # create seed with start symbol
        seed = seed.split() #string to list
        melody = seed
        seed = self._start_symbols + seed

        # map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit the seed to the last max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes = len(self._mappings))
            # dimension onehot_seed = (max_sequence_length, num_classes)
            onehot_seed = onehot_seed[np.newaxis, ...]
            # dimension onehot_seed for keras = (1, max_sequence_length, num_classes)

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # p=[.1 , .2, .1, .... ,.6] ---> sum(p) = 1

            output_int = self._sample_with_temperature(probabilities,temperature)

            # update the seed
            seed.append(output_int)

            # map int to encoding
            output_symbol = [k for k,v in self._mappings.items() if v == output_int][0]

            # check if melody end has been reached
            if output_symbol == "/":
                break

            # update the melody
            melody.append(output_symbol)

            return melody


    def save_melody(self, melody, step_duration =  0.25, format = "midi", file_name = "gen_mel.mid"):

        # create a music21 string
        stream = m21.stream.Stream()  #default 4/4 in C major

        # parse all melody symbol and create note/rest objects
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            # note/rest cases
            if symbol != "_" or i+1 == len(melody):
                # ensure not dealing with first note/rest event
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # rest case
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength = quarter_length_duration)
                    # note case
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength = quarter_length_duration)

                    stream.append(m21_event)

                    # reset step counter
                    step_counter = 1

                start_symbol = symbol
            # prolungation case
            else:
                step_counter += 1

        # write the m21 stream to midi file
        stream.write(format, file_name)


if __name__ == "__main__":
        mg = MelodyGenerator()
        seed = "60 _ 67 _ 67 _ 67 _ 69 _ 67 _ 65 _"
        melody = mg.generate_melody(seed,500,SEQUENCE_LENGTH, 0.7)

        print(melody)
        mg.save_melody(melody)


import torch #cuda counterpart for nvidia gpus?
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from music21 import converter, instrument, note, chord, stream, tempo, meter, key, duration
import glob
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
import pickle
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_music_data(data_dir):
    #Loads midi files into lists for unique note set 
    notes = set()
    sequences = []
    for file in glob.glob(data_dir + '/**/*.midi') + glob.glob(data_dir + '/**/*.mid'):
            print("Initial Parsing: %s" % file)
            sequence = []
            midi = converter.parse(file)
            new_stream = midi.flatten().chordify().stripTies()
            notes_to_parse = new_stream
            notes_from_chords = []
            chord_to_remove_from = []
            for i, element in enumerate(notes_to_parse.notes): #loop through notes in stream
                if element.tie is not None and element.tie.type == "start": #check for start ties
                    if isinstance(element, chord.Chord): #if start tie element is chord
                        for start_current_note in element: #loop through notes in chord to check for start ties
                            if start_current_note.tie is not None and start_current_note.tie.type == "start": #look for specific notes with start ties
                                tied_duration = start_current_note.duration.quarterLength #get duration of start tie note
                                #print(f" - DURATION:{start_current_note.quarterLength} - NOTE:{start_current_note.name}: TIE TYPE:{start_current_note.tie.type}")
                                for end_element in notes_to_parse.notes[i:]: #loop through notes after start tie
                                    break_loop = False
                                    if end_element.tie is not None and end_element.tie.type == "stop": #check for end ties
                                        for end_current_note in end_element: #look for specific notes with end ties
                                            if end_current_note.tie is not None and end_element.tie.type == "stop":
                                                #print(f"   - DURATION:{end_current_note.quarterLength} -NOTE:{end_current_note}: TIE TYPE:{end_current_note.tie.type}")
                                                # print(f"    - TIED DURATION: {tied_duration}")
                                                tied_duration += end_current_note.duration.quarterLength #add duration of end tie note
                                                chord_to_remove_from.append(end_element.derivation.client) #add chord to list of chords to remove notes from
                                                notes_from_chords.append(end_current_note) #add note to list of notes to remove
                                                new_chord_notes = [n if n != start_current_note else note.Note(start_current_note.pitch.midi, quarterLength=tied_duration) for n in element.notes]
                                                new_chord = chord.Chord(new_chord_notes)
                                                notes_to_parse.replace(element, new_chord)
                                                element = new_chord
                                                start_current_note.tie = None
                                                break_loop = True
                                                break
                                    elif end_element.tie is not None and end_element.tie.type == "continue": #check for continue ties
                                        for end_current_note in end_element:
                                            if end_element.tie is not None and end_element.tie.type == "continue": 
                                                #print(f"   - DURATION:{end_current_note.quarterLength} -NOTE:{end_current_note}: TIE TYPE:{end_current_note.tie.type}")
                                                # print(f"    - TIED DURATION: {tied_duration}")
                                                tied_duration += end_current_note.duration.quarterLength
                                                chord_to_remove_from.append(end_element.derivation.client)
                                                notes_from_chords.append(end_current_note)
                                                break
                                    if break_loop: #BREAK OUT OF END TIE CHECK LOOP
                                        break

            #print("Removing:\n")
            for i, element in enumerate(notes_from_chords):
                element_chord = chord_to_remove_from[i]
                new_chord = chord.Chord([n for n in element_chord if n != element])
                try:
                    notes_to_parse.replace(element_chord, new_chord)
                    # print(f" - Removing {element.nameWithOctave}")
                    # print(f" - Remaining: {new_chord}")
                except:
                    # print(f" - {element.nameWithOctave} could not be removed")
                    pass
            notes_to_parse.write('midi', fp='parse_test.mid')
            #print("---------------------\n")
            for i, element in enumerate(notes_to_parse):
                if isinstance(element, note.Note): #Single note
                    #print("note-" + str(element.duration) + "-" + str(element.pitch.midi))#
                    notes.add("note-" + str(element.duration) + "-" + str(element.pitch.midi)) #keeps inversions as same
                    sequence.append("note-" + str(element.duration) + "-" + str(element.pitch.midi))
                elif isinstance(element, chord.Chord): #Chord
                    #print(str("chord-" + ",".join(str(n.duration) for n in element) + "-" + ".".join(str(n.pitch.midi) for n in element)))
                    notes.add(str("chord-" + ",".join(str(n.duration) for n in element) + "-" + ".".join(str(n.pitch.midi) for n in element))) #keeps inversions as same
                    sequence.append(str("chord-" + ",".join(str(n.duration) for n in element) + "-" + ".".join(str(n.pitch.midi) for n in element)))
                elif isinstance(element, note.Rest): #Rest
                    #print("rest"+"-"+str(element.duration))
                    notes.add("rest"+"-"+str(element.duration))
                    sequence.append("rest"+"-"+str(element.duration))
                elif isinstance(element, tempo.MetronomeMark): #Tempo
                    #print("tempo"+"-"+str(element.number))
                    notes.add("tempo"+"-"+str(element.number))
                    sequence.append("tempo"+"-"+str(element.number))
                elif isinstance(element, meter.TimeSignature): #Time Signature
                    #print("time"+"-"+str(element.ratioString))
                    notes.add("time"+"-"+str(element.ratioString))
                    sequence.append("time"+"-"+str(element.ratioString))
                elif isinstance(element, key.Key): #Key Signature
                    #print("key"+"-"+str(element))
                    notes.add("key"+"-"+str(element))
                    sequence.append("key"+"-"+str(element))
            sequences.append(sequence)
            #print(sequences)
    note_list = list(notes)
    pickle.dump(note_list, open('diff_notes.p', 'wb'))
    return note_list, sequences



data_dir = "GRADED PIECES DATASET"

# music_data, sequences = load_music_data(data_dir)
# pickle.dump(music_data, open('full_music_data.p', 'wb'))
# pickle.dump(sequences, open('full_sequences.p', 'wb'))

music_data = pickle.load(open('full_music_data.p', 'rb'))
sequences = pickle.load(open('full_sequences.p', 'rb'))
# print(music_data)

vocab = sorted(list(music_data)) #set of unique notes
n_vocab = len(vocab) #number of unique notes
print(f"There are {n_vocab} unique notes")
notes_to_num = dict((note, number) for number, note in enumerate(vocab)) #dict from turning unique notes to index
# for key, value in notes_to_num.items():
#     print(f"{key}: {value}")


def seq_prep(sequences):
    #Splits sequences up into sets of 8
    seq_length = 8
    network_input = []
    network_output = []
    #print(len(sequences))
    for sequence in sequences:
        #print(f"\n\n{sequence}\n\n")  
        for i in range(len(sequence) - seq_length):   
            seq_in = sequence[i:i+seq_length]
            seq_out = sequence[i+seq_length]
            #Convers to numeric
            network_input.append([notes_to_num[char] for char in seq_in])
            network_output.append(notes_to_num[seq_out])


    #print(f"Network Input: {network_input}")
    #print(f"Network Output: {network_output}")
    
    return network_input, network_output




class MusicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_vocab, layers=3):
        #input_size = number of features
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True, bidirectional=True,dropout=0.3)
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 192)
        self.fc3 = nn.Linear(192, n_vocab)


    def forward(self, x):
        #print(x)

        out, _ = self.lstm(x)
        out = self.fc(out)
        out = self.fc2(out)
        out = self.fc3(out[:, -1, : ])
        return out


def train_network(training_sequence, vocab):
    #print(f"Note Length: {n_vocab}") 
    network_input, network_output = seq_prep(training_sequence)

    model = MusicRNN(input_size=n_vocab, hidden_size=hidden_size, n_vocab=n_vocab)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    print(len(network_input))
    network_input = np.reshape(network_input, (len(network_input), 8, 1)) # reshapes to 3d
    print(network_input.shape)
    network_output = np.reshape(network_output, (len(network_output), 1))

    input_tensor = torch.tensor(network_input, dtype=torch.float32) #converts to tensor
    input_tensor = (input_tensor) / (n_vocab) # normalise input values

    output_tensor = torch.tensor(network_output, dtype=torch.long) #converts to tensor

    batch_size = 64 
    num_epochs = 100
    checkpoint_interval = 10
    

    loader = DataLoader(list(zip(input_tensor, output_tensor)), shuffle=True, batch_size=batch_size)

    # Training loop
    
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        model.train()
        for batch, (input_batch, output_batch) in enumerate(loader):
            #print(input_batch.size)   
            output = model(input_batch)
            # Reshape the output to match the shape of the output_batch
            output = output.view(-1, n_vocab)

            # Reshape the output_batch to match the reshaped output
            output_batch = output_batch.view(-1)


            #loss = nn.NLLLoss()(torch.log(output), output_batch)
            loss = criterion(output, output_batch) # generate loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 50 == 0:
                #print(f"Generated output: {output}")
                #print(f"Expected output batch: {output_batch}")
                print(f"Batch: {batch} - Loss: {loss.item():>7f}")
                losses.append(loss.detach().numpy())
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join("checkpoints", f'new_model_state_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

        scheduler.step(loss)
    end_time = time.time()-start_time
    print(f"Time taken: {end_time}")
    torch.save(model.state_dict(), "dur_fix_model_state.pth")



vocabulary = music_data
seq_prep(sequences)
losses = []
hidden_size = 384

# train_network(sequences, vocab)#
# plt.plot(losses, label = "Loss")
# plt.show()

from numpy.random import choice

def get_seed_sequence(midi_file):
    #Takes midi file as input and extracts 16 note sequence for prediction
    midi = converter.parse(midi_file)
    # midi.flatten().notes.show('text')
    print("Parsing Seed  %s" % midi_file)
    seed_sequence = []
    num_notes = 0
    new_stream = midi.flatten().chordify().stripTies()
    notes_to_parse = new_stream

    notes_from_chords = []
    chord_to_remove_from = []
    for i, element in enumerate(notes_to_parse.notes): #loop through notes in stream
        if element.tie is not None and element.tie.type == "start": #check for start ties
            if isinstance(element, chord.Chord): #if start tie element is chord
                for start_current_note in element: #loop through notes in chord to check for start ties
                    if start_current_note.tie is not None and start_current_note.tie.type == "start": #look for specific notes with start ties
                        tied_duration = start_current_note.duration.quarterLength #get duration of start tie note
                        #print(f" - DURATION:{start_current_note.quarterLength} - NOTE:{start_current_note.name}: TIE TYPE:{start_current_note.tie.type}")
                        for end_element in notes_to_parse.notes[i:]: #loop through notes after start tie
                            break_loop = False
                            if end_element.tie is not None and end_element.tie.type == "stop": #check for end ties
                                for end_current_note in end_element: #look for specific notes with end ties
                                    if end_current_note.tie is not None and end_element.tie.type == "stop":
                                        #print(f"   - DURATION:{end_current_note.quarterLength} -NOTE:{end_current_note}: TIE TYPE:{end_current_note.tie.type}")
                                        # print(f"    - TIED DURATION: {tied_duration}")
                                        tied_duration += end_current_note.duration.quarterLength #add duration of end tie note
                                        chord_to_remove_from.append(end_element.derivation.client) #add chord to list of chords to remove notes from
                                        notes_from_chords.append(end_current_note) #add note to list of notes to remove
                                        new_chord_notes = [n if n != start_current_note else note.Note(start_current_note.pitch.midi, quarterLength=tied_duration) for n in element.notes]
                                        new_chord = chord.Chord(new_chord_notes)
                                        notes_to_parse.replace(element, new_chord)
                                        element = new_chord
                                        start_current_note.tie = None
                                        break_loop = True
                                        break
                            elif end_element.tie is not None and end_element.tie.type == "continue": #check for continue ties
                                for end_current_note in end_element:
                                    if end_element.tie is not None and end_element.tie.type == "continue": 
                                        #print(f"   - DURATION:{end_current_note.quarterLength} -NOTE:{end_current_note}: TIE TYPE:{end_current_note.tie.type}")
                                        # print(f"    - TIED DURATION: {tied_duration}")
                                        tied_duration += end_current_note.duration.quarterLength
                                        chord_to_remove_from.append(end_element.derivation.client)
                                        notes_from_chords.append(end_current_note)
                                        break
                            if break_loop: #BREAK OUT OF END TIE CHECK LOOP
                                break
    
    #print("Removing:\n")
    for i, element in enumerate(notes_from_chords):
        element_chord = chord_to_remove_from[i]
        new_chord = chord.Chord([n for n in element_chord if n != element])
        try:
            notes_to_parse.replace(element_chord, new_chord)
        except:
            pass
    #print("---------------------\n")
    for i, element in enumerate(notes_to_parse):
        if isinstance(element, note.Note): #Single note
            #print("note-" + str(element.duration) + "-" + str(element.pitch.midi))#
            seed_sequence.append("note-" + str(element.duration) + "-" + str(element.pitch.midi))
            num_notes += 1
        elif isinstance(element, chord.Chord): #Chord
            #print(str("chord-" + ",".join(str(n.duration) for n in element) + "-" + ".".join(str(n.pitch.midi) for n in element)))
            seed_sequence.append(str("chord-" + ",".join(str(n.duration) for n in element) + "-" + ".".join(str(n.pitch.midi) for n in element)))
            num_notes += 1
        elif isinstance(element, note.Rest): #Rest
            #print("rest"+"-"+str(element.duration))
            seed_sequence.append("rest"+"-"+str(element.duration))
        elif isinstance(element, tempo.MetronomeMark): #Tempo
            #print("tempo"+"-"+str(element.number))
            seed_sequence.append("tempo"+"-"+str(element.number))
        elif isinstance(element, meter.TimeSignature): #Time Signature
            #print("time"+"-"+str(element.ratioString))
            seed_sequence.append("time"+"-"+str(element.ratioString))
        elif isinstance(element, key.Key): #Key Signature
            #print("key"+"-"+str(element))
            seed_sequence.append("key"+"-"+str(element))
        if num_notes >= 16:
            break
    return seed_sequence

def prepare_sequences_prediction(vocabulary, sequence):
    #Prepares sequence for prediction

    sequence_length = 8
    network_input = []

    sequence_in = sequence[0][:sequence_length]


    network_input.append([notes_to_num[char] for char in sequence_in])
    
    return tuple(network_input)


generated_length = 200

def generate_notes(model, network_input, vocabulary): 
    #Generates notes from set seed

    n_vocab = len(vocabulary)
    pattern = network_input[0]
    #print(pattern)
    prediction_output = []
    raw_prediction_output = []

    for note_index in range(generated_length): #process is repeated 200 times until a long composition is created
        #print(f"-----  NOTE: {note_index+1} -----")
        prediction_input = torch.tensor(pattern, dtype=torch.long)
        prediction_input = (prediction_input) / (n_vocab)
        #print(f"- Pred Input: {prediction_input}")
        prediction_input = prediction_input.view(1, -1, 1)
        #print(f"Reshaped Pred Input: {prediction_input}")
        prediction = model(prediction_input) 
        prediction = prediction.view(-1)
        pred_prob = torch.softmax(prediction, dim=0)
        # print(f"- Prediction Probabilities: {pred_prob}")
        index = torch.multinomial(pred_prob, 1).item()
        # print(f"- Index is: {index}")
        result = list(notes_to_num.keys())[list(notes_to_num.values()).index(index)]
        prediction_output.append(result)
        raw_prediction_output.append(index)
        pattern = np.append(pattern, index)
        pattern = pattern[1:]

        # print(f"- Result: {result}")
        # print(f"- Updated Pattern: {pattern}")


    
    return prediction_output, raw_prediction_output
    
def generate(seed_path):
    seed_sequence = get_seed_sequence(seed_path)

    network_input = prepare_sequences_prediction(vocabulary, [seed_sequence])

    #Loads model from checkpoint
    model = MusicRNN(input_size=n_vocab,hidden_size=hidden_size, n_vocab=n_vocab)
    model.to(device)
    model_dict = torch.load('final_model_state.pth')
    model.load_state_dict(model_dict)
    
    prediction_output, raw_prediction_output = generate_notes(model, network_input, vocabulary)
    # Print the first 16 notes of the generated output vertically
    # for note in prediction_output[:16]:
    #     print(note)

    # Update the seed sequence with the first 16 notes of the generated output
    seed_sequence = seed_sequence[:16] + prediction_output[16:]
    return seed_sequence, raw_prediction_output


from fractions import Fraction


def create_midi(prediction_output, grade):
    offset = 0
    output_notes = []
    output_stream= stream.Stream()
    def duration_extract(duration_parts):
        #print(duration_parts)
        duration_value = duration_parts[1][:-1]  # Extract the duration value
        if '/' in duration_value:  # Check if the duration is a fraction
            duration = float(Fraction(duration_value))  # Convert the fraction string to a float
        else:
            duration = float(duration_value)  # Convert the duration string to a float
        return duration

    for pattern_outer in prediction_output:
        type = pattern_outer.split("-")[0]
        value = pattern_outer.split("-")[1]
        duration_parts = value.split()
        if type == 'rest':
            #print("- rest")

            duration = duration_extract(duration_parts)
            new_rest = note.Rest(quarterLength=duration)
            new_rest.offset = offset
            output_stream.insert(offset, new_rest)
            offset += duration
        elif type == 'tempo':
            #print("- tempo")

            new_tempo = tempo.MetronomeMark(number=value)
            new_tempo.offset = offset
            output_stream.insert(offset, new_tempo)
        elif type == 'time':
            #print("- time")

            new_time = meter.TimeSignature(value)
            new_time.offset = offset
            output_stream.insert(offset, new_time)
        elif type == 'key':
            #print("- key")

            adjusted_key_string = ""
            #print(value.split(" "))
            if len(value.split(" ")) == 1:
                adjusted_key_string = value.split(" ")[0]
            elif value.split(" ")[1] == "major": #reformats key signature
                adjusted_key_string = value.split(" ")[0]
            else:
                adjusted_key_string = value.split(" ")[0].lower()
            new_key = key.Key(adjusted_key_string)
            new_key.offset = offset
            output_stream.insert(offset, new_key)
        elif type == "note":
            # print("\n- note")
            duration = duration_extract(duration_parts)
            note_pitch = pattern_outer.split("-")[2]
            new_note = note.Note(int(note_pitch), quarterLength=duration)
            new_note.offset = offset
            output_stream.insert(offset, new_note)
            offset += duration
            
        elif type == "chord":
            #print("\n- chord")

            if pattern_outer.split("-")[2] == "":
                continue
            notes_in_chord = pattern_outer.split("-")[2].split(".")
            long_chord_note_durations = value.split(",")
            chord_note_durations = []
            for current_duration in long_chord_note_durations:
                chord_note_durations.append(duration_extract(current_duration.split()))
            chord_note_pitches = []
            for i, current_note in enumerate(notes_in_chord):
                new_note = note.Note(int(current_note), quarterLength=float(chord_note_durations[i]))
                output_stream.insert(offset, new_note)
                chord_note_pitches.append(new_note)
            new_chord = chord.Chord(chord_note_pitches)
            new_chord.offset = offset
            offset += float(min(chord_note_durations))    
    file_path = f"generated midi\\{grade}\\generated_piece_{grade}_2.mid"
    output_stream.write('midi', fp=file_path)


#Classifier

from torch.nn.utils.rnn import pad_sequence
grades = pickle.load(open('full_grades.p', 'rb'))
labels=[0,1,2,3,4,5,6,7,8]
class DifficultyClassifer(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=384):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        #print(out.shape)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

def predict_difficulty(network_input, max_length=3114):
    model = DifficultyClassifer(input_size=max_length, num_classes=9)
    model.eval()
    model.to(device)
    model_dict = torch.load('fix_difficulty_pred.pth')
    model.load_state_dict(model_dict)

    input_tensor = torch.tensor(network_input)
    input_tensor = F.pad(torch.tensor(network_input), (0, max_length - len(network_input)), value = -1)[:max_length]

    input_tensor = (input_tensor) / (n_vocab)
    pred_input = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(pred_input)
    pred_prob = torch.softmax(output, dim=1)
    _, predicted_class = torch.max(pred_prob, 1)

    print(f"Probabilities for grades: {pred_prob}")
    #print(f"Predicted class: {predicted_class.item()}")
    predicted_grade = predicted_class.item()
    return predicted_grade


def request():
    requested_grade = -2
    predicted_grade = -1
    valid = False
    while (requested_grade != predicted_grade):
        #INPUT
        while requested_grade not in labels: #request grade until valid input
            requested_grade = input("Enter a grade between 0 and 8: ")
            try:
                requested_grade = int(requested_grade)
                if requested_grade not in labels:
                    print("Invalid input. Please enter a number between 0 and 8.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 8.")
                continue
        #GENERATE
            
        seed_file_path = seed_request(requested_grade)
        generated_sequence, raw_generated_sequence = generate(seed_file_path)
        #PREDICT

        predicted_grade = predict_difficulty(raw_generated_sequence)
        print(f"Requested Grade: {requested_grade} - Predicted Grade: {predicted_grade}")
        create_midi(generated_sequence, requested_grade)



import random
def seed_request(grade):
    #request random seed based on grade
    requested_grade = str(grade)
    base_folder_path = "GRADED PIECES DATASET"
    grade_folder_path = os.path.join(base_folder_path, requested_grade)
    midi_files = [file for file in os.listdir(grade_folder_path)]
    random_file = random.choice(midi_files)
    midi_file_path = os.path.join(grade_folder_path, random_file)
    return midi_file_path

request()     
        
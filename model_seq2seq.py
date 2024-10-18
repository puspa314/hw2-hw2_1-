
import os
import sys
import time
import json
import re
import random
import numpy as np
from scipy.special import expit
import pandas as pd
from os import listdir
from os.path import join, isfile
from bleu_eval import BLEU
col=['modelName','AverageBleu Score']


# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable



# Attention Mechanism

class attention(nn.Module):
    def __init__(self, hidden_size):
        """
        Initializes the Attention mechanism.
        """
        super(attention, self).__init__()
        self.hidden_size = hidden_size
        
        # Linear layers to process concatenated encoder and hidden states
        self.match1 = nn.Linear(2 * hidden_size, hidden_size)
        self.match2 = nn.Linear(hidden_size, hidden_size)
        self.match3 = nn.Linear(hidden_size, hidden_size)
        self.match4 = nn.Linear(hidden_size, hidden_size)
        
        # Converts the hidden representation to a scalar weight
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        """
        Computes attention weights and applies them to encoder outputs.
        """
        batch_size, seq_len, feat_n = encoder_outputs.size()

        # Repeat hidden state to match sequence length
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)

        # Concatenate encoder outputs with repeated hidden state
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2 * self.hidden_size)

        # Pass through linear layers
        x = self.match1(matching_inputs)
        x = self.match2(x)
        x = self.match3(x)
        x = self.match4(x)

        # Compute attention weights and normalize with softmax
        attention_weights = self.to_weight(x).view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context  # Return context vector


# Helper class: Builds vocabulary, manages word frequency, and mappings.

class dictionary(object):

    def __init__(self, filepath, min_word_count=10):
        """
        Initialize dictionary with file path and minimum word count.
        """
        self.filepath = filepath
        self.min_word_count = min_word_count
        self._word_count = {}  # Stores word frequencies
        self.vocab_size = None
        self._good_words = None
        self._bad_words = None
        self.i2w = None  # Index-to-word mapping
        self.w2i = None  # Word-to-index mapping

        # Initialize the dictionary and build mappings
        self._initialize()
        self._build_mapping()
        self._sanitycheck()

    def _initialize(self):
        """
        Load data and count word frequencies.
        """
        with open(self.filepath, 'r') as f:
            file = json.load(f)

        for d in file:
            for s in d['caption']:
                word_sentence = re.sub('[.!,;?]]', ' ', s).split()
                for word in word_sentence:
                    word = word.replace('.', '') if '.' in word else word
                    self._word_count[word] = self._word_count.get(word, 0) + 1

        # Separate good and bad words based on frequency
        self._bad_words = [k for k, v in self._word_count.items() if v <= self.min_word_count]
        self._good_words = [k for k, v in self._word_count.items() if v > self.min_word_count]

    @staticmethod
    def tokenizer_eng(self, text):
        """
        Tokenizes English text using spacy.
        """
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]

    def _build_mapping(self):
        """
        Create word-to-index and index-to-word mappings.
        """
        useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
        self.i2w = {i + len(useful_tokens): w for i, w in enumerate(self._good_words)}
        self.w2i = {w: i + len(useful_tokens) for i, w in enumerate(self._good_words)}

        # Add special tokens to mappings
        for token, index in useful_tokens:
            self.i2w[index] = token
            self.w2i[token] = index

        self.vocab_size = len(self.i2w) + len(useful_tokens)

    def _sanitycheck(self):
        """
        Check if all essential attributes are initialized properly.
        """
        attrs = ['vocab_size', '_good_words', '_bad_words', 'i2w', 'w2i']
        for att in attrs:
            if getattr(self, att) is None:
                raise NotImplementedError(f'Attribute "{att}" is None.')

    def reannotate(self, sentence):
        """
        Replace rare words with <UNK> and add <SOS>/<EOS> tokens.
        """
        sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
        return ['<SOS>'] + [w if self._word_count.get(w, 0) > self.min_word_count else '<UNK>' for w in sentence] + ['<EOS>']

    def word2index(self, w):
        """Convert word to index."""
        return self.w2i[w]

    def index2word(self, i):
        """Convert index to word."""
        return self.i2w[i]

    def sentence2index(self, sentence):
        """Convert a sentence to a list of indices."""
        return [self.w2i[w] for w in sentence]

    def index2sentence(self, index_seq):
        """Convert a list of indices to a sentence."""
        return [self.i2w[int(i)] for i in index_seq]



# Dataset class: Loads and organizes data for training/testing.

class Creating_Dataset(Dataset):
    def __init__(self, label_json, training_data_path, helper, load_into_ram=False):
        # Check if input paths exist
        if not os.path.exists(label_json):
            raise FileNotFoundError(f'File path {label_json} does not exist.')
        if not os.path.exists(training_data_path):
            raise FileNotFoundError(f'File path {training_data_path} does not exist.')

        self.training_data_path = training_data_path
        self.data_pair = []
        self.load_into_ram = load_into_ram
        self.helper = helper

        # Load and process captions
        with open(label_json, 'r') as f:
            label = json.load(f)
        for d in label:
            for s in d['caption']:
                s = self.helper.reannotate(s)
                s = self.helper.sentence2index(s)
                self.data_pair.append((d['id'], s))

        # Load video features into RAM if required
        if load_into_ram:
            self.avi = {}
            files = os.listdir(training_data_path)
            for file in files:
                key = file.split('.npy')[0]
                value = np.load(os.path.join(training_data_path, file))
                self.avi[key] = value

    def __len__(self):
        # Return the total number of data pairs
        return len(self.data_pair)

    def __getitem__(self, idx):
        # Fetch item by index
        assert idx < self.__len__()
        avi_file_name, sentence = self.data_pair[idx]
        avi_file_path = os.path.join(self.training_data_path, f'{avi_file_name}.npy')

        # Load data from RAM or disk
        data = torch.Tensor(self.avi[avi_file_name]) if self.load_into_ram else torch.Tensor(np.load(avi_file_path))
        data += torch.Tensor(data.size()).random_(0, 2000) / 10000.

        return torch.Tensor(data), torch.Tensor(sentence)



# Encoder class

class encoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_percentage=0.3):
        super(encoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Linear layer to adjust input feature size
        self.compress = nn.Linear(input_size, hidden_size)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout_percentage)
        
        # GRU layer for sequence modeling
        self.lstm = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        # Reshape and compress input
        batch_size, seq_len, feat_n = input.size()
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, self.hidden_size)
        
        # Pass through GRU
        output, hidden_state = self.lstm(input)
        return output, hidden_state



# Decoder class

class decoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, helper=None, dropout_percentage=0.2):
        super(decoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.helper = helper

        # Define embedding, dropout, GRU, attention, and output layers
        self.embedding = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(dropout_percentage)
        self.lstm = nn.GRU(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        # Initialize hidden state and input word
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = self.initialize_hidden_state(encoder_last_hidden_state)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        if torch.cuda.is_available():
            decoder_current_input_word = decoder_current_input_word.cuda()

        seq_logProb, seq_predictions = [], []
        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len - 1):
            # Choose input word based on teacher forcing
            threshold = self._get_teacher_learning_ratio(tr_steps)
            current_input_word = (targets[:, i] if random.uniform(0.05, 0.995) > threshold 
                                  else self.embedding(decoder_current_input_word).squeeze(1))

            # Compute context and LSTM output
            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, decoder_current_hidden_state = self.lstm(lstm_input, decoder_current_hidden_state)

            # Calculate log probabilities and update input word
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        # Concatenate results and return
        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def infer(self, encoder_last_hidden_state, encoder_output):
        # Initialize hidden state and <SOS> token
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = self.initialize_hidden_state(encoder_last_hidden_state)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        if torch.cuda.is_available():
            decoder_current_input_word = decoder_current_input_word.cuda()

        seq_logProb, seq_predictions = [], []
        assumption_seq_len = 28

        for i in range(assumption_seq_len - 1):
            # Compute context and LSTM output
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, decoder_current_hidden_state = self.lstm(lstm_input, decoder_current_hidden_state)

            # Calculate log probabilities and update input word
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        # Concatenate results and return
        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def initialize_hidden_state(self, last_encoder_hidden_state):
        return last_encoder_hidden_state if last_encoder_hidden_state is not None else None

    def initialize_cell_state(self, last_encoder_cell_state):
        return last_encoder_cell_state if last_encoder_cell_state is not None else None

    def _get_teacher_learning_ratio(self, training_steps):
        return expit(training_steps / 20 + 0.85)



# Loss function

class LossFun(nn.Module):
    def __init__(self):
        super(LossFun, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()  # Define loss function
        self.loss = 0
        self.avg_loss = None

    def forward(self, x, y, lengths):
        batch_size = len(x)
        predict_cat, groundT_cat = None, None
        flag = True

        # Process each batch and align predictions with ground truth
        for batch in range(batch_size):
            predict = x[batch][:lengths[batch] - 1]
            ground_truth = y[batch][:lengths[batch] - 1]

            if flag:
                predict_cat, groundT_cat = predict, ground_truth
                flag = False
            else:
                predict_cat = torch.cat((predict_cat, predict), dim=0)
                groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

        # Ensure predictions and ground truth have the same length
        try:
            assert len(predict_cat) == len(groundT_cat)
        except AssertionError:
            print(f'Prediction length: {len(predict_cat)}, Ground truth length: {len(groundT_cat)}')

        # Calculate and return the loss
        self.loss = self.loss_fn(predict_cat, groundT_cat)
        self.avg_loss = self.loss / batch_size
        return self.loss



# MODELS class: Combines encoder and decoder into a single model.

class MODELS(nn.Module):
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, avi_feats, mode, target_sentences=None, tr_steps=None):
        # Get encoder outputs and hidden state
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feats)

        # Choose between training or inference mode
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(
                encoder_last_hidden_state=encoder_last_hidden_state,
                encoder_output=encoder_outputs,
                targets=target_sentences,
                mode=mode,
                tr_steps=tr_steps
            )
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(
                encoder_last_hidden_state=encoder_last_hidden_state,
                encoder_output=encoder_outputs
            )
        else:
            raise KeyError('Invalid mode')

        return seq_logProb, seq_predictions




# Model training

class training(object):
    def __init__(self, model, train_dataloader=None, test_dataloader=None, helper=None):
        # Initialize dataloaders for training and testing
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader

        # Check if CUDA is available for GPU acceleration
        self.__CUDA__ = torch.cuda.is_available()
        self.model = model.cuda() if self.__CUDA__ else model.cpu()

        # Initialize parameters, loss function, and optimizer
        self.parameters = model.parameters()
        self.loss_fn = LossFun()
        self.loss = None
        self.optimizer = optim.Adam(self.parameters, lr=0.001)
        self.helper = helper

    def train(self, epoch):
        # Set the model to training mode
        self.model.train()

        for batch_idx, batch in enumerate(self.train_loader):
            # Extract features, ground truths, and sequence lengths from the batch
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()

            avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

            # Reset gradients before backpropagation
            self.optimizer.zero_grad()

            # Forward pass through the model
            seq_logProb, seq_predictions = self.model(
                avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch
            )

            # Remove the <SOS> token from ground truth for loss calculation
            ground_truths = ground_truths[:, 1:]

            # Calculate loss and perform backpropagation
            loss = self.loss_fn(seq_logProb, ground_truths, lengths)
            loss.backward()
            self.optimizer.step()  # Update model weights

            # Print training progress for each batch
            if (batch_idx + 1):
                info = self.get_training_info(
                    epoch=epoch, batch_id=batch_idx, batch_size=len(lengths),
                    total_data_size=len(self.train_loader.dataset),
                    n_batch=len(self.train_loader), loss=loss.item()
                )
                print(info, end='\r')
                sys.stdout.write("\033[K")

        # Print summary information after each epoch
        info = self.get_training_info(
            epoch=epoch, batch_id=batch_idx, batch_size=len(lengths),
            total_data_size=len(self.train_loader.dataset),
            n_batch=len(self.train_loader), loss=loss.item()
        )
        print(info)
        self.loss = loss.item()  # Save the final loss for the epoch

    def eval(self):
        # Set the model to evaluation mode
        self.model.eval()

        for batch_idx, batch in enumerate(self.test_loader):
            # Extract features and ground truths from the batch
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()

            avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

            # Perform inference
            seq_logProb, seq_predictions = self.model(avi_feats, mode='inference')

            # Remove <SOS> token from ground truths for comparison
            ground_truths = ground_truths[:, 1:]

            # Store a sample of predictions and ground truths
            test_predictions, test_truth = seq_predictions[:3], ground_truths[:3]
            break  # Exit after one batch for evaluation

    def test(self):
        # Set the model to evaluation mode for testing
        self.model.eval()
        ss = []  # Store results

        for batch_idx, batch in enumerate(self.test_loader):
            # Extract ID and features from the batch
            id, avi_feats = batch
            if self.__CUDA__:
                avi_feats = avi_feats.cuda()

            # Convert features to float tensors
            id, avi_feats = id, Variable(avi_feats).float()

            # Perform inference
            seq_logProb, seq_predictions = self.model(avi_feats, mode='inference')

            # Replace <UNK> tokens and extract predictions
            result = [
                [x if x != '<UNK>' else 'something' for x in self.helper.index2sentence(s)]
                for s in seq_predictions
            ]
            result = [' '.join(s).split('<EOS>')[0] for s in result]

            # Collect results with corresponding IDs
            ss.extend(zip(id, result))

        return ss  # Return the list of results

    def get_training_info(self, **kwargs):
        # Format and return training progress information
        ep, bID, bs, tds, nb, loss = (
            kwargs.pop("epoch", None),
            kwargs.pop("batch_id", None),
            kwargs.pop("batch_size", None),
            kwargs.pop("total_data_size", None),
            kwargs.pop("n_batch", None),
            kwargs.pop("loss", None),
        )
        return f"Epoch: {ep} [{(bID + 1) * bs}/{tds} ({100. * bID / nb:.0f}%)]\tLoss: {loss:.6f}"



# Main Execution Function



def minibatch(data):
    # Sort data by caption length in descending order
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data)  # Separate features and captions
    avi_data = torch.stack(avi_data, 0)  # Stack features into a tensor

    # Pad captions to the same length
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

def append_log_file(fileLoc, filename, model_name, time):
    # Append model name and training time to log file
    with open(f"{fileLoc}/{filename}.txt", "a") as f:
        f.write(f"\n {model_name}, {time}")

def main_execution():
    # Set paths for data and labels
    training_json = 'training_data/training_label.json'
    training_feats = 'training_data/feat'
    testing_json = 'testing_data/testing_label.json'
    testing_feats = 'testing_data/feat'

    # Initialize dataset and helper
    helper = dictionary(training_json, min_word_count=3)
    train_dataset = Creating_Dataset(training_json, training_feats, helper, load_into_ram=True)
    test_dataset = Creating_Dataset(testing_json, testing_feats, helper, load_into_ram=True)

    # Set model hyperparameters
    inputFeatDim, output_dim = 4096, helper.vocab_size
    batch_sizes, hidden_sizes, word_dims = [32], [128], [2048]
    dropout_percentages, epochs_n = [0.2], 50
    ModelSaveLoc = 'Trained_Models'

    # Create directory for saved models
    if not os.path.exists(ModelSaveLoc):
        os.mkdir(ModelSaveLoc)

    # Initialize log file
    log_filename = str(time.time()) + "_ModelTime_log"
    with open(f"{ModelSaveLoc}/{log_filename}.txt", "x") as f:
        f.write("Model Name, time")

    # Train models with different hyperparameters
    for batch_size in batch_sizes:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=minibatch)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=minibatch)

        for hidden_size in hidden_sizes:
            for dropout_percentage in dropout_percentages:
                for word_dim in word_dims:
                    encoder = encoderRNN(inputFeatDim, hidden_size, dropout_percentage)
                    decoder = decoderRNN(hidden_size, output_dim, output_dim, word_dim, dropout_percentage)
                    model = MODELS(encoder, decoder)
                    trainer = training(model, train_loader, test_loader, helper)

                    # Train and evaluate the model
                    start = time.time()
                    for epoch in range(epochs_n):
                        trainer.train(epoch + 1)
                        trainer.eval()
                    end = time.time()

                    # Save the trained model and log training time
                    model_name = f"model_batchsize_{batch_size}_hidsize_{hidden_size}_DP_{dropout_percentage}_worddim_{word_dim}"
                    torch.save(model, f"{ModelSaveLoc}/{model_name}.h5")
                    append_log_file(ModelSaveLoc, log_filename, model_name, end - start)

class test_data(Dataset):
    def __init__(self, test_data_path):
        # Load video features into memory
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])

    def __len__(self):
        return len(self.avi)

    def __getitem__(self, idx):
        # Retrieve a specific item by index
        assert idx < self.__len__()
        return self.avi[idx]

def testmodel(arg):
    ModelSaveLoc = "Best_Model"
    result_pd = pd.DataFrame(columns=['modelName', 'AverageBleu Score'])

    # Load all trained models
    file_paths = [join(ModelSaveLoc, f) for f in listdir(ModelSaveLoc) if isfile(join(ModelSaveLoc, f)) and f.endswith(".h5")]
    dataset = test_data(f'{arg[0]}/feat')
    test_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
    helper = dictionary('training_data/training_label.json', min_word_count=3)

    # Evaluate each model
    for model_loc in file_paths:
        model = torch.load(model_loc, map_location='cpu') if not torch.cuda.is_available() else torch.load(model_loc)
        trainer = training(model, test_dataloader=test_loader, helper=helper)

        # Run testing and log results
        for _ in range(1):
            results = trainer.test()

        with open(arg[1], 'w') as f:
            for id, s in results:
                f.write(f'{id},{s}\n')

        # Calculate BLEU scores
        test = json.load(open('testing_data/testing_label.json', 'r'))
        result = {}
        with open(arg[1], 'r') as f:
            for line in f:
                test_id, caption = line.strip().split(',', 1)
                result[test_id] = caption

        bleu_scores = [BLEU(result[item['id']], [x.rstrip('.') for x in item['caption']], True) for item in test]
        average_score = sum(bleu_scores) / len(bleu_scores)
        print(f"Average BLEU score: {average_score}")

        # Log BLEU scores
        result_pd = pd.concat([result_pd, pd.DataFrame([{'modelName': model_loc, 'AverageBleu Score': average_score}])], ignore_index=True)
        result_pd.to_csv("output_temp.csv", index=False)

    result_pd.to_csv("output_temp.csv", index=False)

if __name__ == "__main__":
    train, test = False, True
    if train:
        main_execution()
    if test:
        arg = [sys.argv[1], sys.argv[2]]
        testmodel(arg)

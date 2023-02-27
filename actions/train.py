import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.encoderrnn import EncoderRNN
from networks.decoderrnn import DecoderRNN
from networks.attentiondecoderrnn import AttnDecoderRNN
from actions.evaluate import evaluateRandomly, evaluate
from dataset import Dataset
from language import Language
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
import math
import random
import csv
import values as v


def main():
    parser = argparse.ArgumentParser(prog='Train', description='Training of RNNs to perform text-to-phonemes transcription.')
    parser.add_argument("num_epochs", help="Number of training iterations.", type=int)
    parser.add_argument('--att', default=False, action='store_true')
    args = parser.parse_args()
    attention = args.att
    if attention:
        training_save_path = ('trained_models/encoder_rnn_att.pt', 'trained_models/decoder_rnn_att.pt', 'trained_models/training_evaluation_att_history.csv')
    else:
        training_save_path = ('trained_models/encoder_rnn_test.pt', 'trained_models/decoder_rnn_test.pt', 'trained_models/training_evaluation_history.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = args.num_epochs
    l_rate = 0.001 if attention else 0.01

    transcrip_data = Dataset(v.PT_BR_DICTIONARY_FILE)
    train_set_size = transcrip_data.train_data_length

    language = Language(v.PHONEMES_FILE, v.LETTERS_FILE)

    encoder = EncoderRNN(language.letters_length, v.HIDDEN_SIZE, device).to(device)
    if attention:
        decoder = AttnDecoderRNN(v.HIDDEN_SIZE, language.phonemes_length, device, dropout_p=0.1).to(device)
    else:
        decoder = DecoderRNN(v.HIDDEN_SIZE, language.phonemes_length, device).to(device)

    

    trainIters(transcrip_data, language, encoder, decoder, attention, training_save_path, device, n_epochs=num_epochs, 
    plot_every=train_set_size, learning_rate=l_rate, save_models_bool=True)


def train(input_tensor, target_tensor, encoder, decoder, attention, encoder_optimizer, decoder_optimizer, criterion, device, max_length=v.MAX_LENGTH):
    encoder.train()
    decoder.train()
    
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    accuracy = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[v.SOW_TOKEN]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < v.TEACHER_FORCING_RATIO else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            loss += criterion(decoder_output, target_tensor[di])
            if topi.squeeze().detach() == target_tensor[di].squeeze().detach():
                accuracy += 1
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if topi.squeeze().detach() == target_tensor[di].squeeze().detach():
                accuracy += 1
            if decoder_input.item() == v.EOW_TOKEN:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return (loss.item() / target_length, accuracy / target_length)


def trainIters(transcrip_data, language, encoder, decoder, attention, training_save_path, device, n_epochs=15, plot_every=100, 
               learning_rate=0.01, save_models_bool=False):
    start = time.time()
    MIN_ACCURACY = 0.98
    train_pairs = transcrip_data.get_train_pairs()
    val_pairs = transcrip_data.get_val_pairs()
    train_set_size = len(train_pairs)
    print('Train set size: ' + str(train_set_size))
    n_iters = train_set_size * n_epochs
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    print_accuracy_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    plot_accuracy_total = 0  # Reset every plot_every
    csv_row = ()
    csv_str = []

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    
    training_pairs = []
    for i in range(n_epochs):
        random.shuffle(train_pairs)
        for i in range(len(train_pairs)):
            training_pairs.append(language.tensorsFromPair(train_pairs[i], device, language))

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss, accuracy = train(input_tensor, target_tensor, encoder,
                     decoder, attention, encoder_optimizer, decoder_optimizer, criterion, device)
        
        print_accuracy_total += accuracy
        plot_accuracy_total += accuracy
        print_loss_total += loss
        plot_loss_total += loss

        if iter % train_set_size == 0: # Prints result every epoch
            print_accuracy_avg = print_accuracy_total / train_set_size
            print_accuracy_total = 0
            print_loss_avg = print_loss_total / train_set_size
            print_loss_total = 0
            print('(%d %d%%) Avg loss: %.4f, Accuracy: %.2f%%' % (iter, iter / n_iters * 100, print_loss_avg, print_accuracy_avg * 100))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            plot_accuracy_avg = plot_accuracy_total / plot_every
            plot_accuracy_total = 0
            eval_accuracy = evaluateRandomly(val_pairs,language, encoder, decoder, attention, device)
            if save_models_bool:
                csv_row = (round(plot_loss_avg,8), round(plot_accuracy_avg * 100,8), round(eval_accuracy * 100,8))
                csv_str.append(csv_row)
                count = int(iter / plot_every)
                if plot_accuracy_avg > MIN_ACCURACY: save_models(encoder, decoder, training_save_path, count)
    
    if save_models_bool:
        with open(training_save_path[2], 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_str)
        save_models(encoder, decoder, training_save_path)        

    showPlot(plot_losses)


def showPlot(points):
    #plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def save_models(encoder, decoder, training_save_path, epoch=-1):
    if epoch < 0:
        torch.save(encoder.state_dict(), training_save_path[0])
        torch.save(decoder.state_dict(), training_save_path[1])
    else:
        encoder_path = training_save_path[0].split('.')[0] + '_' + str(epoch) + '.pt'
        decoder_path = training_save_path[1].split('.')[0] + '_' + str(epoch) + '.pt'
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)


if __name__ == "__main__":
    main()
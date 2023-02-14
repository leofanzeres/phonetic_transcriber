import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from encoderrnn import EncoderRNN
from decoderrnn import DecoderRNN
import utils as ut
import time
import math
import random
import values as v

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_iterations = 45000
l_rate = 0.01
print_every=3000


dictionary = ut.load_dictionary('pt-br.dic','\t', False)
shuffled_dict = ut.shuffle_dict(dictionary)

train_dict, _, __ = ut.split_data(shuffled_dict, v.split)

train_pairs = ut.get_pairs(train_dict)

teacher_forcing_ratio = 0.0

letter2index = ut.load_dictionary('letters.csv',',',reverse=True)
phoneme2index = ut.load_dictionary('phonemes.csv',',',reverse=True)
index2phoneme = ut.load_dictionary('phonemes.csv',',')


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=v.MAX_LENGTH):
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

    decoder_input = torch.tensor([[v.SOW_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_output == target_tensor[di]:
                accuracy += 1
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if topi.squeeze().detach() == target_tensor[di].squeeze().detach():
                accuracy += 1
            if decoder_input.item() == v.EOW_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return (loss.item() / target_length, accuracy / target_length)


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, save_models=False):
    start = time.time()
    print('Train set size: ' + str(len(train_pairs)))
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    print_accuracy_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    plot_accuracy_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [ut.tensorsFromPair(random.choice(train_pairs), device, letter2index, phoneme2index)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss, accuracy = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        print_accuracy_total += accuracy
        plot_accuracy_total += accuracy
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_accuracy_avg = print_accuracy_total / print_every
            print_accuracy_total = 0
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f %.2f' % (iter, iter / n_iters * 100, print_loss_avg, print_accuracy_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    if save_models:
        torch.save(encoder.state_dict(), 'models/encoder_rnn.pt')
        torch.save(decoder.state_dict(), 'models/decoder_rnn.pt')

    showPlot(plot_losses)





import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()



encoder1 = EncoderRNN(len(letter2index), v.hidden_size, device).to(device)
decoder1 = DecoderRNN(v.hidden_size, len(phoneme2index), device).to(device)

trainIters(encoder1, decoder1, n_iters=n_iterations, print_every=print_every, learning_rate=l_rate, save_models=True)
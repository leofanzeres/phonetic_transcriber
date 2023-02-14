import torch
import random
import utils as ut
from encoderrnn import EncoderRNN
from decoderrnn import DecoderRNN
import values as v

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dictionary = ut.load_dictionary('pt-br.dic','\t', False)
shuffled_dict = ut.shuffle_dict(dictionary)
_, val_dict, __ = ut.split_data(shuffled_dict, v.split)
val_pairs = ut.get_pairs(val_dict)
letter2index = ut.load_dictionary('letters.csv',',',reverse=True)
phoneme2index = ut.load_dictionary('phonemes.csv',',',reverse=True)
index2phoneme = ut.load_dictionary('phonemes.csv',',')

# Load models
encoder1 = EncoderRNN(len(letter2index), v.hidden_size, device).to(device)
encoder1.load_state_dict(torch.load('models/encoder_rnn.pt'))
encoder1.eval()
decoder1 = DecoderRNN(v.hidden_size, len(phoneme2index), device).to(device)
decoder1.load_state_dict(torch.load('models/decoder_rnn.pt'))
decoder1.eval()
 
def evaluate(encoder, decoder, word, max_length=v.MAX_LENGTH):
    with torch.no_grad():
        input_tensor = ut.tensorFromWord(word, device, letter2index, phoneme2index)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[v.SOW_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_phonemes = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden) # attention
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == v.EOW_token:
                decoded_phonemes.append('<EOW>')
                break
            else:
                decoded_phonemes.append(index2phoneme[str(topi.item())])

            decoder_input = topi.squeeze().detach()

        return decoded_phonemes, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(val_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_phonemes, attentions = evaluate(encoder, decoder, pair[0])
        output_phonemes = ' '.join(output_phonemes)
        print('<', output_phonemes)
        print('')

evaluateRandomly(encoder1, decoder1, len(val_pairs))
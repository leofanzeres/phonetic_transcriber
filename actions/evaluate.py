import argparse
import torch
import random
import utils as ut
import logging
from networks.encoderrnn import EncoderRNN
from networks.decoderrnn import DecoderRNN
from networks.attentiondecoderrnn import AttnDecoderRNN
from dataset import Dataset
from language import Language
import values as v


def main():
    parser = argparse.ArgumentParser(prog='Evaluate', description='Evaluation of RNN models performing text-to-phonemes transcription.')
    parser.add_argument("encoder_file", help="Encoder file path.", type=str)
    parser.add_argument("decoder_file", help="Decoder file path.", type=str)
    parser.add_argument('--att', default=False, action='store_true')
    args = parser.parse_args()
    attention = args.att

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transcrip_data = Dataset(v.PT_BR_DICTIONARY_FILE, v.SPLIT)
    val_pairs = transcrip_data.get_val_pairs()

    language = Language(v.PHONEMES_FILE, v.LETTERS_FILE)

    # Load models
    encoder = EncoderRNN(language.letters_length, v.HIDDEN_SIZE, device).to(device)
    encoder.load_state_dict(torch.load(args.encoder_file))
    if attention:
        decoder = AttnDecoderRNN(v.HIDDEN_SIZE, language.phonemes_length, device, dropout_p=0.1).to(device)
    else:
        decoder = DecoderRNN(v.HIDDEN_SIZE, language.phonemes_length, device).to(device)
    decoder.load_state_dict(torch.load(args.decoder_file))

    evaluateRandomly(val_pairs, language, encoder, decoder, attention, device, True, len(val_pairs))
 
def evaluate(language, encoder, decoder, attention, device, word, max_length=v.MAX_LENGTH):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor, target_tensor = language.tensorsFromPair(word, device, language)
        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        index2phoneme = ut.get_swap_dict(language.get_phonemes())

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[v.SOW_TOKEN]], device=device)

        decoder_hidden = encoder_hidden

        decoded_phonemes = []
        decoder_attentions = torch.zeros(max_length, max_length)
        accuracy = 0
        for di in range(max_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if di < target_length and topi.squeeze().detach() == target_tensor[di].squeeze().detach():
                accuracy += 1
            if topi.item() == v.EOW_TOKEN:
                decoded_phonemes.append('<EOW>')
                break
            else:
                decoded_phonemes.append(index2phoneme[str(topi.item())])

            decoder_input = topi.squeeze().detach()

        return decoded_phonemes, decoder_attentions[:di + 1], accuracy / target_length

def evaluateRandomly(eval_pairs, language, encoder, decoder, attention, device, print_res=False, n=10):
    accuracy_total = 0
    random.shuffle(eval_pairs)
    for i in range(n):
        pair = eval_pairs[i]
        if print_res:
            print('>', pair[0])
            print('=', pair[1])
        output_phonemes, attentions, accuracy = evaluate(language, encoder, decoder, attention, device, pair)
        if print_res:
            output_phonemes = ' '.join(output_phonemes)
            print('<', output_phonemes)
            print('')
        accuracy_total += accuracy
    accuracy_avg = accuracy_total / n
    if print_res:
        #print('Test set size: %d' % (n))
        logging.info('Test set size: %d' % (n))
        #print('Accuracy: %.2f' % (accuracy_avg * 100))
        logging.info('Accuracy: %.2f' % (accuracy_avg * 100))
    return accuracy_avg

if __name__ == "__main__":
    main()
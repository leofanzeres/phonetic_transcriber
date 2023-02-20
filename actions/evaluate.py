import argparse
import torch
import random
import utils as ut
from networks.encoderrnn import EncoderRNN
from networks.decoderrnn import DecoderRNN
from dataset import Dataset
from language import Language
import values as v


def main():
    parser = argparse.ArgumentParser(prog='Evaluate', description='Evaluation of RNN models performing text-to-phonemes transcription.')
    parser.add_argument("encoder_file", help="Encoder file path.", type=str)
    parser.add_argument("decoder_file", help="Decoder file path.", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transcrip_data = Dataset(v.PT_BR_DICTIONARY_FILE)
    val_pairs = transcrip_data.get_val_pairs()

    language = Language(v.PHONEMES_FILE, v.LETTERS_FILE)

    # Load models
    encoder = EncoderRNN(language.letters_length, v.HIDDEN_SIZE, device).to(device)
    encoder.load_state_dict(torch.load(args.encoder_file))
    encoder.eval()
    decoder = DecoderRNN(v.HIDDEN_SIZE, language.phonemes_length, device).to(device)
    decoder.load_state_dict(torch.load(args.decoder_file))
    decoder.eval()

    evaluateRandomly(val_pairs, language, encoder, decoder, device, len(val_pairs))
 
def evaluate(language, encoder, decoder, device, word, max_length=v.MAX_LENGTH):
    with torch.no_grad():
        input_tensor = language.tensorFromWord(word, device, language)
        input_length = input_tensor.size()[0]
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

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden) # attention
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == v.EOW_TOKEN:
                decoded_phonemes.append('<EOW>')
                break
            else:
                decoded_phonemes.append(index2phoneme[str(topi.item())])

            decoder_input = topi.squeeze().detach()

        return decoded_phonemes, decoder_attentions[:di + 1]

def evaluateRandomly(eval_pairs, language, encoder, decoder, device, n=10):
    for i in range(n):
        pair = random.choice(eval_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_phonemes, attentions = evaluate(language, encoder, decoder, device, pair[0])
        output_phonemes = ' '.join(output_phonemes)
        print('<', output_phonemes)
        print('')


if __name__ == "__main__":
    main()
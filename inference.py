from pathlib import Path

import torch

from train import get_model, get_ds


def beam_search_inference(beam_size: int = 4):
    seq_len = 384
    d_model = 384
    N = 6
    heads = 8
    dff = d_model * 4
    dropout = 0.1
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_train, dataloader_val, tokenizer_src, tokenizer_dst = get_ds('en', 'es', seq_len, batch_size)
    model = get_model(tokenizer_src.get_vocab_size(), tokenizer_dst.get_vocab_size(), seq_len, d_model, N, heads,
                      dropout, dff).to(device)
    model.eval()
    model_file = Path('./data/model').joinpath('model.pt')
    print(f'Loading model from {model_file}')
    torch.load(model_file)
    pad = tokenizer_src.token_to_id('<pad>')
    eos = tokenizer_dst.token_to_id('</s>')
    text_eng = f'<s>{input("Enter the english text to be translated: ")}</s>'
    input_tokens_encoder = tokenizer_src.encode(text_eng).ids
    pad_tokens_input = seq_len - len(input_tokens_encoder)
    input_encoder = torch.tensor(input_tokens_encoder + [pad] * pad_tokens_input).to(device)
    mask_encoder = (input_encoder == pad).unsqeeze(0).to(device)
    input_tokens_decoder = tokenizer_dst.encode('<s>').ids
    input_decoder = torch.tensor(input_tokens_decoder).to(device)
    mask_decoder = (torch.tril(torch.ones(input_decoder.size(0), input_decoder.size(0))) == 0).to(device)
    with torch.no_grad():

        output_encoder = model.encode(input_encoder, mask_encoder)
        search = [input_decoder]
        while search:
            input_decoder = search[0]
            search = search[1:]
            if input_decoder.size(-1) >= seq_len:
                break
            output_decoder = model.decode(input_decoder, output_encoder, mask_encoder, mask_decoder)
            prob = model.project(output_decoder[:, -1])
            _, indices = torch.topk(prob, beam_size)
            if eos in indices:
                break
            search += [torch.cat((input_decoder, i)) for i in indices]
        out = tokenizer_dst.decode(output_decoder[:, -1])
        print(out)


if __name__ == '__main__':
    beam_search_inference(4)

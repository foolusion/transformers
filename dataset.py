import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, data: list[dict[str, dict[str, str]]], tokenizer_src: Tokenizer, tokenizer_dst: Tokenizer, lang_src: str, lang_dst: str, seq_len: int):
        super().__init__()
        self.data = data
        self.tokenizer_src = tokenizer_src
        self.tokenizer_dst = tokenizer_dst
        self.lang_src = lang_src
        self.lang_dst = lang_dst
        self.seq_len = seq_len

        self.token_sos = torch.tensor([tokenizer_src.token_to_id('<s>')], dtype=torch.int64)
        self.token_eos = torch.tensor([tokenizer_src.token_to_id('</s>')], dtype=torch.int64)
        self.token_pad = torch.tensor([tokenizer_src.token_to_id('<pad>')], dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        translation_pair = self.data[idx]
        text_src = translation_pair['translation'][self.lang_src]
        text_dst = translation_pair['translation'][self.lang_dst]
        input_tokens_enc = self.tokenizer_src.encode(text_src).ids
        input_tokens_dec = self.tokenizer_dst.encode(text_dst).ids
        num_padding_tokens_enc = self.seq_len - len(input_tokens_enc) - 2
        num_padding_tokens_dec = self.seq_len - len(input_tokens_dec) - 1
        assert num_padding_tokens_enc >= 0
        assert num_padding_tokens_dec >= 0
        input_encoder = torch.cat(
            [
                self.token_sos,
                torch.tensor(input_tokens_enc, dtype=torch.int64),
                self.token_eos,
                torch.tensor([self.token_pad] * num_padding_tokens_enc, dtype=torch.int64),
            ]
        )
        input_decoder = torch.cat(
            [
                self.token_sos,
                torch.tensor(input_tokens_dec, dtype=torch.int64),
                torch.tensor([self.token_pad] * num_padding_tokens_dec, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(input_tokens_dec, dtype=torch.int64),
                self.token_eos,
                torch.tensor([self.token_pad] * num_padding_tokens_dec, dtype=torch.int64)
            ]
        )

        assert input_encoder.size(0) == self.seq_len
        assert input_decoder.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        mask = torch.tril(torch.ones(self.seq_len, self.seq_len)) == 1
        return {
            "input_encoder": input_encoder,
            "input_decoder": input_decoder,
            "mask_encoder": (input_encoder != self.token_pad).unsqueeze(0).unsqueeze(0).int(),
            "mask_decoder": (input_decoder != self.token_pad).unsqueeze(0).unsqueeze(0).int() & mask,
            "label": label,
            "text_src": text_src,
            "text_dst": text_dst,
        }

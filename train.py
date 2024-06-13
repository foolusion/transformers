from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tqdm import tqdm

from dataset import BilingualDataset
from model import build_transformer


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(ds, lang):
    tokenizer_path = Path(f'data/tokenizer_{lang}.json')
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordPiece(unk_token='<unk>'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(special_tokens=['<s>', '</s>', '<pad>', '<unk>'])
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(lang_src, lang_dst, seq_len, batch_size):
    ds_raw = load_dataset('opus_books', f'{lang_src}-{lang_dst}', split='train')
    tokenizer_src = get_or_build_tokenizer(ds_raw, lang_src)
    tokenizer_dst = get_or_build_tokenizer(ds_raw, lang_dst)
    ds_raw = [x for x in ds_raw
              if
              len(tokenizer_src.encode(x['translation'][lang_src]).ids) <=
              len(tokenizer_dst.encode(x['translation'][lang_dst]).ids) <=
              seq_len-2
              ]
    train_ds_size = int(0.9 * len(ds_raw))
    ds_raw_train, ds_raw_val = random_split(ds_raw, [train_ds_size, len(ds_raw) - train_ds_size])
    dataset_train = BilingualDataset(ds_raw_train, tokenizer_src, tokenizer_dst, lang_src, lang_dst, seq_len)
    dataset_val = BilingualDataset(ds_raw_val, tokenizer_src, tokenizer_dst, lang_src, lang_dst, seq_len)

    max_len_src = 0
    max_len_dst = 0

    for item in ds_raw:
        ids_src = tokenizer_src.encode(item['translation'][lang_src]).ids
        ids_dst = tokenizer_dst.encode(item['translation'][lang_dst]).ids
        max_len_src = max(max_len_src, len(ids_src))
        max_len_dst = max(max_len_dst, len(ids_dst))
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of destination sentence: {max_len_dst}')

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    return dataloader_train, dataloader_val, tokenizer_src, tokenizer_dst


def get_model(vocab_size_src, vocab_size_dst, seq_len, d_model, N, heads, dropout, dff):
    model = build_transformer(vocab_size_src, vocab_size_dst, seq_len, d_model, N, heads, dropout, dff)
    return model


def train_model():
    preload = True
    seq_len = 384
    d_model = 384
    N = 6
    heads = 8
    dff = d_model * 4
    dropout = 0.1
    batch_size = 32
    training_epochs = 100000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    Path('./data/model').mkdir(parents=True, exist_ok=True)

    dataloader_train, dataloader_val, tokenizer_src, tokenizer_dst = get_ds('en', 'es', seq_len, batch_size)
    model = get_model(tokenizer_src.get_vocab_size(), tokenizer_dst.get_vocab_size(), seq_len, d_model, N, heads, dropout, dff).to(device)
    writer = SummaryWriter('logs', flush_secs=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    initial_epoch = 0
    global_step = 0
    model_file = Path('./data/model').joinpath('model.pt')
    if preload:
        print(f'Loading model from {model_file}')
        state = torch.load(model_file)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('<pad>'), label_smoothing=0.1).to(device)
    for epoch in range(initial_epoch, training_epochs):
        model.train()
        batch_iterator = tqdm(dataloader_train, desc=f'Epoch {epoch:02d}')
        for batch in batch_iterator:
            input_encoder = batch['input_encoder'].to(device)
            input_decoder = batch['input_decoder'].to(device)
            mask_encoder = batch['mask_encoder'].to(device)
            mask_decoder = batch['mask_decoder'].to(device)

            encoder_output = model.encode(input_encoder, mask_encoder)
            decoder_output = model.decode(input_decoder, encoder_output, mask_encoder, mask_decoder)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_dst.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():.4f}'})
            writer.add_scalar('loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_file)


if __name__ == '__main__':
    train_model()

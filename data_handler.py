import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split, Dataset
from datasets import load_dataset
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from data_handler import Opus_Dataset, causal_mask

from pathlib import Path
from typing import Any


class Opus_Dataset(Dataset):
    def __init__(self, config,ds_raw, src_lng, tgt_lng, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, seq_length:int):
        self.config = config

        self.ds_raw = ds_raw
        self.src_lng = src_lng
        self.tgt_lng = tgt_lng

        self.seq_length = seq_length

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.sos_num_token = [tokenizer_src.token_to_id("[SOS]")]
        self.eos_num_token = [tokenizer_src.token_to_id("[EOS]")]
        self.pad_num_token = [tokenizer_src.token_to_id("[PAD]")]


    def __len__(self):
        return len(self.ds_raw)


    def __getitem__(self, index: int):
        # To get the token ids corresponding to each word of the index's sentence

        # get the sentence in both languages
        src_text = self.ds_raw[index][self.config["key"]][self.src_lng]
        tgt_text = self.ds_raw[index][self.config["key"]][self.tgt_lng]

        # list of ids related to each word - in the sentence.
        src_ids_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_ids_tokens = self.tokenizer_src.encode(tgt_text).ids

        src_padding_size = self.seq_length - len(src_ids_tokens) - 2
        tgt_padding_size = self.seq_length - len(tgt_ids_tokens) - 1

        assert src_padding_size > 0 and tgt_padding_size > 0, "src or tgt _ids_tokens exceed the expected size: (seq_length-1 or seq_length -2)."

        # build tensors of (seq_length) with SOS, EOS, PAD.
        src_ids_tensor = torch.cat(
            [
                torch.tensor(self.sos_num_token, dtype= torch.int64),
                torch.tensor(src_ids_tokens, dtype = torch.int64),
                torch.tensor(self.eos_num_token, dtype = torch.int64),
                torch.tensor(self.pad_num_token * src_padding_size, dtype= torch.int64)
            ]
        )
        tgt_ids_tensor = torch.cat(
            [
                torch.tensor(self.sos_num_token, dtype=torch.int64),
                torch.tensor(tgt_ids_tokens, dtype=torch.int64),
                torch.tensor(self.pad_num_token * tgt_padding_size, dtype= torch.int64)
            ]
        ).type(torch.int64)

        label_ids_tensor = torch.cat(
            [
                torch.tensor(tgt_ids_tokens, dtype=torch.int64),
                torch.tensor(self.eos_num_token, dtype=torch.int64),
                torch.tensor(self.pad_num_token * tgt_padding_size, dtype = torch.int64)
            ]
        )

        assert src_ids_tensor.size(0) == self.seq_length
        assert tgt_ids_tensor.size(0) == self.seq_length
        assert label_ids_tensor.size(0) == self.seq_length

        # build (seq_length) and (1,seq_length, seq_length) masks respectively for encoder_mask, and decoder_mask
        encoder_mask = (src_ids_tensor != self.pad_num_token[0]).unsqueeze(0).unsqueeze(0).type(torch.int64)
        decoder_mask = (tgt_ids_tensor != self.pad_num_token[0]).unsqueeze(0).unsqueeze(0).type(torch.int64) & causal_mask(self.seq_length)

        response = {
            "encoder_input": src_ids_tensor,
            "decoder_input": tgt_ids_tensor,
            "label_input": label_ids_tensor,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

        return response


def causal_mask(size):
    # lower matrix of ones in size of (1, size, size)
    return torch.tril(torch.ones(1,size,size)).type(torch.int64)








def opusbooks_ds_loader(config):
    ds = load_dataset(config['dataset_name'], f'{config["Subset"]}', split="train")
    config["key"] = "translation"
    return ds

def ds_iterator(ds, key, lang):
    # related to Opus_Books dataset's structure
    for item in ds[key]:
        yield item[lang]


def get_build_tokenizer(config, ds, lng):
    tokenizer_path = Path(config["tokenizer_file"].format(lng))
    Path(str(Path(config["tokenizer_file"]).parent)).mkdir(parents=True, exist_ok=True)
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(ds_iterator(ds, config['key'], lng), trainer= trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def sequence_length_finder(config, ds_raw, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer):
    # Built for Opus Books Dataset
    max_src_length = 0
    max_tgt_length = 0
    for item in ds_raw[config["key"]]:
        src_item_ids = tokenizer_src.encode(item[config["src_lang"]]).ids
        tgt_item_ids = tokenizer_tgt.encode(item[config["tgt_lang"]]).ids
        max_src_length = max(max_src_length, len(src_item_ids))
        max_tgt_length = max(max_tgt_length, len(tgt_item_ids))

    # set seq_length
    margin = 20
    seq_length = max(max_src_length, max_tgt_length) + margin

    return max_src_length, max_tgt_length, seq_length


def create_ds_dl(config, batch_size, train_num_workers):
    # create torch dataset and DataLoader

    ds_raw = opusbooks_ds_loader(config)
    tokenizer_src = get_build_tokenizer(config, ds_raw, config["src_lang"])
    tokenizer_tgt = get_build_tokenizer(config, ds_raw, config["tgt_lang"])

    # split dataset: 0.9 training and 0.1 validation
    train_size = int(0.9 * len(ds_raw))
    valid_size = len(ds_raw) - train_size
    training_raw_ds, validation_raw_ds = random_split(ds_raw, [train_size, valid_size])

    # find sequence length
    src_seq, tgt_seq, seq_length = sequence_length_finder(config, ds_raw, tokenizer_src, tokenizer_tgt)
    config["src_seq"] = src_seq
    config["tgt_seq"] = tgt_seq
    config["seq_length"] = seq_length
    # create torch Dataset for training and valid dataset
    training_DS = Opus_Dataset(config, training_raw_ds, config["src_lang"], config["tgt_lang"], tokenizer_src, tokenizer_tgt, seq_length)
    valid_DS = Opus_Dataset(config, validation_raw_ds, config["src_lang"], config["tgt_lang"], tokenizer_src, tokenizer_tgt, seq_length)

    # DataLoader
    training_DL = DataLoader(training_DS, batch_size= batch_size, num_workers = train_num_workers, shuffle= True)
    valid_DL = DataLoader(valid_DS, batch_size= 1, shuffle= True)

    return training_DS, valid_DS, training_DL, valid_DL, tokenizer_src, tokenizer_tgt


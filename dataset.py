import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
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

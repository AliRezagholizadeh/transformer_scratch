import torch
import torch.nn as nn
from tokenizers import Tokenizer
from data_handler import causal_mask
from torch.utils.data import DataLoader

class TrModelNLPValidation:
    def __init__(self, model: nn.Module, tokenizer_src:Tokenizer, tokenizer_tgt:Tokenizer, device):
        # SOS and PAD of target lang
        self.start_token = [tokenizer_tgt.token_to_id("[SOS]")]
        self.padding_token = [tokenizer_tgt.token_to_id("[PAD]")]
        self.end_token = [tokenizer_tgt.token_to_id("[EOS]")]

        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_src = tokenizer_src

        self.model = model
        self.device = device

    def predict_next(self, source_input: torch.tensor, tgt_current_tokens_ts: torch.tensor, encoder_mask):
        """Predict the next word given the model, the source input, and current target words (actual or predicted ones)."""
        '''
        Parameters:
            tgt_current_tokens_ts: A tensor of the shape (,n) which contains n tokens.
        Return: 
            next_token: string 
        '''
        self.model.eval()

        next_token_index = tgt_current_tokens_ts.size(1)
        # print(f"Current tokens: {tgt_current_tokens_ts} \n>> To predict the next token.")
        # padding current predicted tokens
        tgt_tokens_ts = torch.cat(
            [
                tgt_current_tokens_ts,
                torch.tensor(self.padding_token * (self.max_length - tgt_current_tokens_ts.size(1))).type_as(source_input).unsqueeze(0)
            ], dim = -1
        ).to(self.device)
        # print(f"Current tokens: {tgt_current_tokens_ts} \n>> padded size {tgt_tokens_ts.shape}")

        # find decoder mask
        decoder_mask = (tgt_tokens_ts != self.padding_token[0]).unsqueeze(0).unsqueeze(0).type(torch.int64) & causal_mask(
            self.max_length).to(self.device)


        with torch.no_grad():   # ****
            encode_x = model.encode(source_input, encoder_mask)  # (B, seq_length, d_model)
            decode_x = model.decode(tgt_tokens_ts, encode_x, encoder_mask, decoder_mask)  # (B, seq_length, d_model)
            proj_x = model.project(decode_x)  # (B, seq_length, vocab_size)

            next_token = self.greedy_projection(proj_x, next_token_index)

        return next_token

    def progressive_prediction(self, source_input: torch.tensor, encoder_mask: torch.tensor, max_length):
        """Predict entire target sentence given a source input and the model.

        Parameters:
            pass
        Return:
            predicted_tokens: A list of tokens (string) predicted.
            tgt_current_tokens_ts: A tensor containing the token ids predicted.

        """

        self.max_length = max_length
        # tgt_token_ts =
        # predicted tokens - starting from SOS
        tgt_current_tokens_ts = torch.tensor(self.start_token).type_as(source_input).unsqueeze(0)  # (1, 1) shape
        predicted_tokens = []
        while True:
            # stop when reach to the max_length
            if(tgt_current_tokens_ts.size(1) >= max_length):
                break

            # predict the next token (string) given the model, source_input, and the current predicted tokens.
            next_predicted_token = self.predict_next(source_input, tgt_current_tokens_ts, encoder_mask)
            predicted_tokens.append(next_predicted_token)
            tgt_current_tokens_ts = torch.cat(
                [
                    tgt_current_tokens_ts,
                    torch.tensor([self.tokenizer_tgt.token_to_id(next_predicted_token)]).type_as(source_input).unsqueeze(0)
                ], dim = -1
            )

            # stop when end token seen
            if(next_predicted_token == self.end_token[0]):
                break

        return predicted_tokens, tgt_current_tokens_ts

    def validate(self, valid_dl: DataLoader, max_length: int, num_examples:int = 2):
        """
        iterate over validation data set and find predicted sentence given the model by the mean of the next token finding strategy.
        Then, compare the predicted token with the actual tokens.

        Parameters:
            valid_dl: Validation Data Loader
            max_length: max length to generate tokens.
            num_examples: number of samples to evaluate.
        Return:
        """


        print(f"{'-' * 120} \nInference on Validation DS.")
        source_texts = []
        target_texts = []
        predicted_texts = []
        progress_indx = 0
        # predict new data
        for batch_valid in valid_dl: # batch length is 1.
            progress_indx += 1
            # encoder data
            encoder_input = batch_valid["encoder_input"].to(self.device)
            encoder_mask = batch_valid["encoder_mask"].to(self.device)
            src_text = batch_valid["src_text"]
            print(f"Validation {progress_indx}/{num_examples} - input src_text : {src_text} \n encoder_input shape: {encoder_input.shape}")

            assert encoder_input.shape[0] == 1, "validation batch size is not one."
            # Predicted tokens
            predicted_tokens, tgt_tokens_ts = self.progressive_prediction(encoder_input, encoder_mask, max_length)

            # Actual Data / decoder
            # decoder_input = batch_valid["decoder_input"]
            tgt_text = batch_valid["tgt_text"]
            predicted_text = ' '.join(predicted_tokens)
            # print(f"Validation {progress_indx}/{valid_dl.batch_size} - predicted tokens: {predicted_tokens} \n Actual text: {tgt_text}")
            print(f"\n- Source txt: {src_text} \n- Actual text: {tgt_text} \n- predicted tokens: {predicted_text}")


            source_texts.append(src_text)
            target_texts.append(tgt_text)
            predicted_texts.append(predicted_text)

            # TODO: you can bring more NLP evaluation metrics.

            if(progress_indx == num_examples):
                break
        print(f"End of Inferencing/Validation \n{'-' * 120}")


    def greedy_projection(self, projection: torch.tensor, token_index):
        """
        Greedily selecting a token from the output of the projection layer.

        Parameters:
            projection: (B, seq, vocab_size). Here B is assumed to be 1.
            token_index: indicates the index of word demanded within the sentence.
        Return:
            next_token getting from tokenizer_tgt.id_to_token
        """
        assert projection.size(0) == 1, f"Expected to have batch size of 1. shape is {projection.shape}"
        # print(f"Greedy selection - among {len(projection[0][token_index])} vocabolary size. shape is: {projection[0][token_index].shape}")
        # vocab index of the token with max predicted probability
        selected_token_VocabIndx = torch.argmax(projection[0][token_index])
        # find the token
        next_token = self.tokenizer_tgt.id_to_token(selected_token_VocabIndx)[0]
        # print(f"Greedy selection - next token found: {next_token}")
        return next_token




def validation2():
    pass


from config import get_config, find_model_conf_dir
from pathlib import Path
# import sys
# PROJECT_PATH = Path(".").absolute().parent
# sys.path.append(str(PROJECT_PATH))
from data_handler import create_ds_dl
from model import get_model

if __name__ == "__main__":
    config = get_config("config.yml")
    config = find_model_conf_dir(config)

    # device
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    # get model object
    train_num_workers = 1
    batch_size = config["batch_size"]
    training_DS, valid_DS, training_DL, valid_DL, tokenizer_src, tokenizer_tgt = create_ds_dl(config, batch_size, train_num_workers)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # load last model
    model_dir = config["model"]["model_dir"]
    model_basename = config["model"]["model_basename"]
    assert isinstance(config['model'],
                      dict), f"config['model'] expected to be dict but is {type(config['model'])} - config['model']: {config['model']}"

    if ((last_epoch := config['model'].get('last_epoch', None))):  # there is already trained model
        # build model file path
        model_file_name = f"{model_basename}_{last_epoch}.pt"
        model_file = str(Path(model_dir) / model_file_name)

        # load last model
        state = torch.load(model_file)
        model.load_state_dict(state["model_state_dict"])
        print(f"model trained obver {last_epoch + 1} epochs loaded.")
    else:
        raise Exception("There is no model trained with this config.")

    # evaluate
    valid_obj = TrModelNLPValidation(model, tokenizer_src, tokenizer_tgt, device)
    valid_obj.validate(valid_DL, config["seq_length"])




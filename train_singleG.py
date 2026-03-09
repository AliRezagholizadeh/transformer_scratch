import yaml
from torch import nn
import torch
# import validators
from torch.utils.data import DataLoader, random_split, Dataset

from torch.utils.tensorboard import SummaryWriter
from model import get_model
from data_handler import create_ds_dl

from tokenizers import Tokenizer

from pathlib import Path

from config import find_model_conf_dir, get_config, update_lepoch_config
from tqdm import tqdm

import warnings

def train_model(config, tokenizer_src:Tokenizer, tokenizer_tgt:Tokenizer, traindataloader: DataLoader):
    # set device
    # torch.mps.empty_cache()
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if(device_name == "cpu"):
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    # cpu_dev = torch.device("cpu")

    print(f">> Device to train on the data: {device}")

    Path(config["model"]["checkpoint_dir_name"]).mkdir(parents=True, exist_ok=True)
    # get the model on the device
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # tensorboard
    writer = SummaryWriter(str(Path(config['experiment']['dir'])/"single_gpu"))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config['model']['lr'], eps= 1e-9)

    # preload existing model
    initial_epoch = 0
    global_step = 0
    # Load previous model with the same configuration as config if exists. Otherwise, create new model dir and copy updated config.
    # change the config to the corresponding model related config.
    config = find_model_conf_dir(config)
    # make sure config is adjusted to the checkpoint related config.
    assert (m_dir := config["model"].get("model_dir", None)), f"model dir in the corresp. config is not the same: config[model][model_dir]:{m_dir}."
    assert (c_dir := config.get("config_dir", None)), f"config dir in the corresp. config is not the same: config[config_dir]:{c_dir}. "

    model_dir = config["model"]["model_dir"]
    model_basename = config["model"]["model_basename"]
    assert isinstance(config['model'], dict), f"config['model'] expected to be dict but is {type(config['model'])} - config['model']: {config['model']}"

    if((last_epoch:=config['model'].get('last_epoch',None))): # there is already trained model
        # build model file path
        # model_file_name = f"{model_basename}_model_{last_epoch}.pt"
        model_file_name = f"{model_basename}_{last_epoch}.pt"

        model_file = str(Path(model_dir) / model_file_name)

        # load previous model
        state = torch.load(model_file)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]


    # loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index= tokenizer_tgt.token_to_id("[PAD]"), label_smoothing= 0.1).to(device)

    # train loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        # over batches
        batches_iterator = tqdm(traindataloader, desc= f"epoch {epoch} - step over batches - ")

        effective_batch = config["effective_batch"]
        gradient_accumulation_steps = round(effective_batch / config["batch_size"])
        accumulation_count = 0
        batch_indx = 0
        # for i in batches_iterator:
        #     batch_indx+= 1
        #     if(batch_indx > 14290):
        #         break
        # skip_batch_nun = 14290
        skip_batch_nun = 0
        for batch in batches_iterator:         # idle iterating over batches
            batch_indx += 1
            if (batch_indx > skip_batch_nun):
                encoder_input = batch["encoder_input"].to(device)   # (B, seq_length)
                encoder_mask = batch["encoder_mask"].to(device)     # (B, seq_length)

                # print("Encode")
                encode_x = model.encode(encoder_input, encoder_mask)  # (B, seq_length, d_model)
                batches_iterator.set_postfix({"- after encoder: x shape": encode_x.shape, "dtype": encode_x.dtype}, refresh= True)
                # encoder_input.to(cpu_dev)
                # encoder_mask.to(cpu_dev)

                # print("Decode")
                decoder_input = batch["decoder_input"].to(device)   # (B, seq_length)
                decoder_mask = batch["decoder_mask"].to(device)     # (B, seq_length, seq_length)
                decode_x = model.decode(decoder_input, encode_x, encoder_mask, decoder_mask) # (B, seq_length, d_model)
                # decoder_input.to(cpu_dev)
                # decoder_mask.to(cpu_dev)
                # print("Project")
                batches_iterator.set_postfix({"- after decoder: x shape": decode_x.shape, "dtype": decode_x.dtype}, refresh= True)

                proj_x = model.project(decode_x) # (B, seq_length, vocab_size)
                batches_iterator.set_postfix({"- after proj_x: x shape": proj_x.shape, "dtype": proj_x.dtype}, refresh= True)


                label_input = batch["label_input"].to(device)  # (B, seq_length)
                batches_iterator.set_postfix({"- label_input: x shape": label_input.shape, "dtype": label_input.dtype}, refresh= True)

                # calculate the cross entropy loss
                loss = loss_fn(proj_x.view(-1, tokenizer_tgt.get_vocab_size()), label_input.view(-1))
                loss = loss / gradient_accumulation_steps
                # label_input.to(cpu_dev)
                # back propagate and accumulate the losses

                accumulation_count += 1
                if((accumulation_count == gradient_accumulation_steps) or (batch_indx == len(batches_iterator))): # if it is the time of optimizer step: on the effective batch num or on the last batch
                    # remaining smaller batches which hasn't been considered in larger Efficient batch size (in the last batch_index)
                    if((step_diff := gradient_accumulation_steps - accumulation_count) > 0): # in the case of the last batch occured before effective batch num
                        batches_iterator.set_postfix(
                            {"- There is step_diff in efficient batch calculator. step_diff:": step_diff}, refresh=True)
                        try:
                            loss = step_diff * loss / gradient_accumulation_steps
                            loss.backward()
                        except Exception as e:
                            print(f"Error in calculating loss and backward inside : {e}")
                    else:   # current batch occured at effective batch num
                        loss.backward()

                    batches_iterator.set_postfix({" - loss": f"{loss.item():6.3f}"})
                    # print(f"- loss: ", f"{loss.item():6.3f}")
                    # print(f"batch {batch_indx}/{len(batches_iterator)} - loss: {loss}")
                    # log train loss
                    writer.add_scalar("train_loss", loss.item(), global_step)
                    writer.flush()


                    # update the weights
                    optimizer.step()
                    # clear the loss
                    optimizer.zero_grad()

                    global_step += 1
                    accumulation_count = 0
                else:  # this batch is not at the effective batch num or is not the last batch
                    loss.backward()

        # if(epoch % 5 == 0):
        batches_iterator.set_postfix({"model saved": None})
        print("model saved")
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        # save the model
        model_file_name = f"{model_basename}_{epoch}.pt"
        model_file = str(Path(model_dir) / model_file_name)
        torch.save(state, model_file)
        # update the config
        update_lepoch_config(config, epoch)
        print("✅ model and the config stored/updates. ")



def monitor_model(config, tokenizer_src:Tokenizer, tokenizer_tgt:Tokenizer, valid_dataloader: DataLoader):
    # checkpoint_dir_path = Path(str(Path(get_model_file_path(config, 0)).parent))
    # iterate over checkpoints in the dir
    pass



def train_single_gpu_interface(config_path):
    warnings.filterwarnings("ignore")
    config = get_config(config_path)
    train_num_workers = 1
    batch_size = config["batch_size"]
    training_DS, valid_DS, training_DL, valid_DL, tokenizer_src, tokenizer_tgt = create_ds_dl(config, batch_size, train_num_workers)
    train_model(config, tokenizer_src, tokenizer_tgt, training_DL)



if __name__ == "__main__":
    config_path = "config.yml"
    train_single_gpu_interface(config_path)



class ModelOnCloud:
    def __init__(self, data_model_name: str, data_shape: list, model: nn.Module, train_iter: int, batch_size: int, model_repository_url):
        assert validators.url(model_repository_url), "url is not valid."


class LifeTimeLearning:
    def __init__(self, train_iter: int, batch_size: int, model_repository_url):
        assert validators.url(model_repository_url), "url is not valid."

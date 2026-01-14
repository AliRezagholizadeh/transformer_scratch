"""
A script to provide a way to distribute the training phase over
Single node - multiple GPUs (NVIDIA supported, maybe MPS supported later).
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# import validators
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from model import get_model
from data_handler import create_ds_dl
from tokenizers import Tokenizer

from pathlib import Path
# from torchvision import datasets, transforms
import yaml
from config import find_model_conf_dir, get_config, update_lepoch_config
from tqdm import tqdm

import warnings


def setup(rank, world_size):
    # check the Multi-GPUs available, since it is not yet provided for Mac unified chips.
    if torch.backends.mps.is_available():
        raise Exception("Multi-training is not applied in MPS Macbook chip environment.")

    MASTER_ADDR = 'localhost'
    MASTER_PORT = '12355'
    # put variables as env
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT

    BACKEND_ENGINE = "nccl"
    # initialize the process group
    dist.init_process_group(BACKEND_ENGINE, rank=rank, world_size=world_size)
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)

    print(f"--> Multi GPU is set on pytorch.distributed framework. MASTER_ADDR: {MASTER_ADDR}, MASTER_PORT: {MASTER_PORT}, BACKEND_ENGINE: {BACKEND_ENGINE}, world_size: {world_size}, rank: {rank}")


def cleanup():
    dist.destroy_process_group()


# DDP Training Function
def train(rank, world_size, initial_epoch, global_step, model, traindataloader, tokenizer_tgt, loss_fn, optimizer, config):
    print(f"{'-'*120}")
    print(f"{' '*60 + 'TRAIN ON GPU:' + rank }")
    print(f"{'-'*120}")
    # set multi-gpu env variables and NVIDIA compatible backend engine
    setup(rank, world_size)
    device = rank




    # DDP encapsulates the model
    model = DDP(model.to(device), device_ids=[rank])
    loss_fn = loss_fn.to(device)

    # load model_dir and model_basename
    model_dir = config["model"]["checkpoint_dir_name"]
    model_basename = config["model"]["model_basename"]

    # set tensorboard writer
    writer = SummaryWriter(str(Path(config['experiment']['dir'])/f"Rank_{rank}"))

    # Train Loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        # over batches
        batches_iterator = tqdm(traindataloader, desc=f"rank {rank} , epoch {epoch} - step over batches - ")


        for batch in batches_iterator:  # idle iterating over batches
            optimizer.zero_grad()

            # batch_indx += 1
            # if (batch_indx > skip_batch_nun):
            encoder_input = batch["encoder_input"].to(device)  # (B, seq_length)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, seq_length)

            # print("Encode")
            encode_x = model.encode(encoder_input, encoder_mask)  # (B, seq_length, d_model)
            # batches_iterator.set_postfix({"- after encoder: x shape": encode_x.shape, "dtype": encode_x.dtype},
            #                              refresh=True)
            # encoder_input.to(cpu_dev)
            # encoder_mask.to(cpu_dev)

            # print("Decode")
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_length)
            decoder_mask = batch["decoder_mask"].to(device)  # (B, seq_length, seq_length)
            decode_x = model.decode(decoder_input, encode_x, encoder_mask, decoder_mask)  # (B, seq_length, d_model)
            # decoder_input.to(cpu_dev)
            # decoder_mask.to(cpu_dev)
            # print("Project")
            # batches_iterator.set_postfix({"- after decoder: x shape": decode_x.shape, "dtype": decode_x.dtype},
            #                              refresh=True)

            proj_x = model.project(decode_x)  # (B, seq_length, vocab_size)
            # batches_iterator.set_postfix({"- after proj_x: x shape": proj_x.shape, "dtype": proj_x.dtype},
            #                              refresh=True)

            label_input = batch["label_input"].to(device)  # (B, seq_length)
            # batches_iterator.set_postfix({"- label_input: x shape": label_input.shape, "dtype": label_input.dtype},
            #                              refresh=True)

            # calculate the cross entropy loss
            loss = loss_fn(proj_x.view(-1, tokenizer_tgt.get_vocab_size()), label_input.view(-1))
            # loss = loss / gradient_accumulation_steps
            # label_input.to(cpu_dev)
            # back propagate and accumulate the losses

            # accumulation_count += 1
            # if ((accumulation_count == gradient_accumulation_steps) or (batch_indx == len(
            #         batches_iterator))):  # if it is the time of optimizer step: on the effective batch num or on the last batch
            #     # remaining smaller batches which hasn't been considered in larger Efficient batch size (in the last batch_index)
            #     if ((
            #             step_diff := gradient_accumulation_steps - accumulation_count) > 0):  # in the case of the last batch occured before effective batch num
            #         batches_iterator.set_postfix(
            #             {"- There is step_diff in efficient batch calculator. step_diff:": step_diff}, refresh=True)
            #         try:
            #             loss = step_diff * loss / gradient_accumulation_steps
            #             loss.backward()
            #         except Exception as e:
            #             print(f"Error in calculating loss and backward inside : {e}")
            #     else:  # current batch occured at effective batch num
            #         loss.backward()
            #
            #     batches_iterator.set_postfix({" - loss": f"{loss.item():6.3f}"})
            #     # print(f"- loss: ", f"{loss.item():6.3f}")
            #     # print(f"batch {batch_indx}/{len(batches_iterator)} - loss: {loss}")
            #     # log train loss
            #     writer.add_scalar("train_loss", loss.item(), global_step)
            #     writer.flush()
            #
            #     # backward
            #     loss.backward()
            #     # update the weights
            #     optimizer.step()
            #     # clear the loss
            #     optimizer.zero_grad()
            #
            #     global_step += 1
            #     accumulation_count = 0
            # else:  # this batch is not at the effective batch num or is not the last batch
            #     loss.backward()

            batches_iterator.set_postfix({f" Rank: {rank} - loss": f"{loss.item():6.3f}"})
            # print(f"- loss: ", f"{loss.item():6.3f}")
            # print(f"batch {batch_indx}/{len(batches_iterator)} - loss: {loss}")
            # log train loss
            writer.add_scalar(f"Rank: {rank} - train_loss: ", loss.item(), global_step)
            writer.flush()

            # backward
            loss.backward()
            # update the weights
            optimizer.step()
            # clear the loss

            global_step += 1

        # if(epoch % 5 == 0):
        if(rank == 0):
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

        # need to update the config[preload] and store it in Config file in Checkpoint


def main(world_size: int, tokenizer_src:Tokenizer, tokenizer_tgt:Tokenizer, traindataloader: DataLoader, config: dict):
    """Make it ready to call train. Provides adjusted config, model, and optimizer to load previous model if available."""
    
    # distributed_dir = "distributed/multiNode_multiGPU"
    distributed_dir = f"distributed/singleNode_multiGPU({world_size})"
    dist_checkpoint_dir_path = Path(config["model"]["checkpoint_dir_name"])/ distributed_dir
    dist_checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    # update checkpoint_dir_name
    config["model"]["checkpoint_dir_name"] = str(dist_checkpoint_dir_path)

    # change the config to the corresponding model _if exists_ to the related config. Otherwise, create new model dir and copy updated config.
    config = find_model_conf_dir(config)

    # make sure config is adjusted to the checkpoint related config.
    assert (m_dir := config["model"].get("model_dir",
                                         None)), f"model dir in the corresp. config is not the same: config[model][model_dir]:{m_dir}."
    assert (c_dir := config.get("config_dir",
                                None)), f"config dir in the corresp. config is not the same: config[config_dir]:{c_dir}. "


    # load model_dir
    model_dir = config["model"]["model_dir"]
    model_basename = config["model"]["model_basename"]

    assert isinstance(config['model'],
                      dict), f"config['model'] expected to be dict but is {type(config['model'])} - config['model']: {config['model']}"


    # tensorboard
    assert(config['experiment'].get('dir', None), "ERROR: experiment dir is not set.")
    # writer = SummaryWriter(config['experiment']['dir'])
    # ! SummaryWriter will be set in each train function run on each GPU to prevent overwriting on the same file.

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    # Load previous model with the same configuration as config if exists.
    if((last_epoch:=config['model'].get('last_epoch',None))): # there is an already trained model with this config
        # build model file path
        model_file_name = f"{model_basename}_{last_epoch}.pt"
        model_file = str(Path(model_dir) / model_file_name)

        # load previous model
        state = torch.load(model_file)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    # set crossEntropy loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1)



    print("Ready to call train function over multi-GPU using torch.multiprocessing ")
    # call train in torch.multiprocessing
    torch.multiprocessing.spawn(train, args=(world_size, initial_epoch, global_step, model, traindataloader,tokenizer_tgt, loss_fn, optimizer, config), nprocs=world_size)




if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # get config
    config = get_config("config.yml")

    # effective batch
    effective_batch = config["effective_batch"]
    print(f">> Base: approximate effective batch size: {effective_batch}")


    # Number of GPUs
    gpu_num = torch.cuda.device_count()
    world_size = gpu_num
    print(f">> Base: number of GPU recognized: {world_size}")

    # find the batch num to be run in each GPU
    each_gpu_batch = effective_batch // gpu_num
    print(f">> Base: Each GPU batch num: {each_gpu_batch}")

    # training_DS, valid_DS, training_DL, valid_DL, tokenizer_src, tokenizer_tgt = create_ds_dl(config, batch_size = config["batch_size"], num_workers= 2 )
    training_DS, valid_DS, training_DL, valid_DL, tokenizer_src, tokenizer_tgt = create_ds_dl(config, batch_size = each_gpu_batch, train_num_workers= 2)
    print(">> Base: Dataset and Tokenizers loaded: OPUS Book")

    # get the model on the device
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    print(f">> Base: Model loaded: {config['model']['model_basename']}")

    # call the main Multi_GPU powered function
    main(world_size, tokenizer_src, tokenizer_tgt, training_DL, config)

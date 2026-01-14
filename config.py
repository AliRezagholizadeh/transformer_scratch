import yaml
from pathlib import Path
from copy import deepcopy

def get_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.load(file, yaml.SafeLoader)

    config["model"]["lr"] = float(config["model"]["lr"])

    return config


def del_dict_value(dict_: dict, key: str) -> None:
    """Delete value from a dictionary.

    Parameters:
        key: can be a nested variable, like: model/preload

    """
    nested_values = key.split("/")
    print(f"nested_values: {nested_values}")

    a = dict_
    for i in range(len(nested_values) - 1):
        print(f"- nested_values:{nested_values}")

        if(isinstance(a, dict)):
            a = a.get(nested_values[0], None)

        del nested_values[0]

    if(isinstance(a, dict) and a.get(nested_values[0])):
        del a[nested_values[0]]

def prune_dict(dictn: dict, base_keys, except_keys: list, parent_keys: str = "") -> None:
    """A recursive function to prune a copy of the dictionary to keep base_keys and delete except_keys.

    Parameters:
        base_keys, except_keys: a list of strings to store keys or nested keys (ex. key1/n_key1).
        parent_keys: a string to keep track of the nested keys
    """
    # if(parent_keys == ""): # make deep copy at the first call of the prune_dict.
    #     dict_ = deepcopy(dictn)
    # else:
    #     dict_ = dictn
    dict_ = deepcopy(dictn)
    # iterate over the keys and call the function if nested dict found
    for key in dictn:
        if(isinstance(dict_[key], dict)): # nested key
            if((f"{parent_keys}{key}" in except_keys) or (f"{parent_keys}{key}" not in base_keys)):
                del dict_[key]
            else:
                dict_[key] = prune_dict(dict_[key], base_keys, except_keys, f"{parent_keys}{key}/")
        else:
            if ((f"{parent_keys}{key}" in except_keys) or (f"{parent_keys}{key}" not in base_keys)):
                print(f"'{parent_keys}{key}' is either in except ({except_keys}), or not in base_keys {base_keys} ---> So it will be deleted.")
                del dict_[key]


    return dict_


def find_dict_keys(dictn: dict, keys_found: list = [], parent_keys:str = ""):
    """
    A recursive function to find and store all keys and nested keys in keys_found.
    """
    for key in dictn:
        if(isinstance(dictn[key], dict)): # nested key
            keys_found.append(f"{parent_keys}{key}")
            keys_found = find_dict_keys(dictn[key], keys_found, f"{parent_keys}{key}/")
        else:
            keys_found.append(f"{parent_keys}{key}")

    return keys_found

def config_identical(base_config, model_config, except_keys):
    """
    Check if both configs are the same (considering base_config keys as pivot point, except some keys).

    Return:
        Boolean of whether two config is the same (with the mentioned considerations)
    """

    # find base keys given the base config
    base_keys = find_dict_keys(base_config)
    print(f"base_keys found: {base_keys}")
    model_config_pr = prune_dict(model_config, base_keys, except_keys)
    config_pr = prune_dict(base_config, base_keys, except_keys)

    print(f"model_config_pr: {model_config_pr}")
    print(f"config_pr: {config_pr}")

    return model_config_pr == config_pr



def find_model_conf_dir(config):
    """
    Find corresponding dir in the checkpoint directory to resume the model trained with the same config.
    If it does not exist in the checkpoint directory, it will copy the input config with model_dir and config_dir added.
    Parameters:
        config: This is the general config located in original project dir.
    Return:
    """
    # epoch = None
    except_keys = ["model/last_epoch", "model/model_dir", "config_dir", "experiment/name"]

    model_dir = None
    config_model = config.get("model", None)
    if(isinstance(config_model, dict)):
        model_dir = config_model.get("model_dir", None)

    if(not model_dir):  # config entered is the original one located in the project dir.
        # find corresponding checkpoint dir given main parameters in config.

        # model checkpoint base dir
        model_base_dir = Path('.')/config["model"]["checkpoint_dir_name"]
        # mkdir if required
        if(not model_base_dir.is_dir()):
            model_base_dir.mkdir(parents=True, exist_ok=True)

        # iterate over sub-directories
        model_config = None
        model_indx = 1
        if(modeldirs:=model_base_dir.iterdir()):
            for modeldir in modeldirs:
                print(f"check if this model dir ({str(modeldir)}) has the identical config.")
                # print("sub-dir: ", modeldir, f" - type: {type(modeldir)}")
                # check config
                if(modelfiles:= modeldir.iterdir()):
                    for file_ in modelfiles:
                        print(f"check this sub dir/file {file_}")
                        # check the config exists (considering the main parameters).
                        if(file_.is_file() and str(file_).split('/')[-1] == config["yml_config_file_name"]):
                            model_config = get_config(file_)
                            # print(f"Model config: {model_config}")
                            # print(f"Is it equal to the current run: {model_config == config}")
                            # check if the config found is the same as the original config (considering the main parameters).
                            if(config_identical(config, model_config, except_keys)):
                                model_dir = file_.parent
                                print(f"⚠️ The same config to the current one found. So we write checkpoints on this path.")

                                config = model_config
                                print(f"config updated by the model_config items. new config: {config}")
                                # assert model_dir and config_dir are correct
                                assert (m_dir:=config["model"].get("model_dir", None)) and m_dir == str(model_dir), f"model dir in the corresp. config is not the same: config[model][model_dir]:{m_dir}."
                                assert (c_dir:=config.get("config_dir", None)) and c_dir == str(model_dir), f"config dir in the corresp. config is not the same: config[config_dir]:{c_dir}. "


                                epoch = model_config["model"].get("last_epoch", None)
                                if(epoch):
                                    print(f"--> last epoch trained: {epoch}")
                                else:
                                    print(f"--> It would be a raw model. last epoch not found")

                                break
                            else:
                                print("-- The config here is not the same as the base config.")

                model_indx += 1


        # If a right model dir not found, create and copy the current config
        if(not model_dir):
            print("✅ New root made for storing the model. ")
            model_dir = model_base_dir / config["model"]["model_basename"] / f"hyperP_setting_{model_indx}"
            model_dir.mkdir(parents=True, exist_ok=True)

            # set experiemnt dirs/sub dirs
            exp_base_dir = Path('.') / config["experiment"]["name"]
            # mkdir if required
            if (not exp_base_dir.is_dir()):
                exp_base_dir.mkdir(parents=True, exist_ok=True)
            exp_dir = exp_base_dir / config["model"]["model_basename"] / f"hyperP_setting_{model_indx}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            # add model/model_dir and config_dir
            config["model"].update({"model_dir": str(model_dir)})
            config.update({"config_dir": str(model_dir)})

            # update experiment
            config["experiment"]["dir"] = exp_dir

            # store the updated config file in the checkpoint dir
            conf_file = str(model_dir/config["yml_config_file_name"])
            write_yaml_config(config, conf_file)
            # model_dir.write_text(model_dir/config["yml_config_file_name"], config, encoding="utf-8")
            print(f"✅ Updated config file stored in the path: {str(model_dir)}")
        else:
            print(f"---> model with the same config found: {model_dir}")

    else: # the config contains mdoel_dir, so it means the config is the one within the checkpoint dir, not the base config.
        pass


    # model_basename = config["model"]["model_basename"]
    # model_file_name = f"{model_basename}"+"_{0}.pt" if not epoch else f"{model_basename}_{epoch}.pt"
    # return str(Path('.') / model_dir / model_file_name)
    return config

def update_lepoch_config(config: dict, last_epoch: int):
    """
    Update last epoch of the config file located in the corresponding checkpoint.
    Parameters:
        config: config file. If it is the base config (not containing the model/model_dir), it will call
        last_epoch: int

    """
    assert config.get("config_dir",
                                None), f"config dir does not exist in the corresp. config. "

    config["model"].update({"last_epoch": last_epoch})
    file_path = f"{config['config_dir']}/{config['yml_config_file_name']}"
    write_yaml_config(config, file_path)

def write_yaml_config(config, file):
    with open(file, 'w') as conf_file:
        yaml.dump(config, conf_file, default_flow_style=False, sort_keys=False)





if __name__ == "__main__":
    config_file = "config.yml"
    config = get_config(config_file)
    # config["model"]["lr"] = float(config["model"]["lr"])
    print("config: ", config)

    # config_
    config_file = "config_.yml"
    config_ = get_config(config_file)
    # config["model"]["lr"] = float(config["model"]["lr"])
    print(f"config_: {config_} - type: {type(config_)}")

    print("identical? ", config_ == config)

    sample_model = get_model_file_path(config, 100)
    print("sample_model: ", sample_model)
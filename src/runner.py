from .char_datasets import CharWindowDataset
from .model import CharModel
from .char_datasets import CharWindowDataset

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
from tqdm import tqdm
import json
import os


class Runner:
    
    def __init__(self, configs_dir: str) -> None:
        
        self.optimizers = {
            'SGD': optim.SGD,
            'Adam': optim.Adam,
            'RMSprop': optim.RMSprop,
            'Adagrad': optim.Adagrad,
            'AdamW': optim.AdamW
        }
        self.loss_fns = {
            "mse": nn.MSELoss,
            "cross_entropy": nn.CrossEntropyLoss,
            "nll": nn.NLLLoss,
            "l1": nn.L1Loss,
            "bce": nn.BCELoss,
            "bce_with_logits": nn.BCEWithLogitsLoss
        }
        self.default_configs = {
            "model_name": "default_model",
            "data_dir": "data/default.txt",
            "args_model": {
                "input_size": 50,
                "embedding_dim": 2,
                "hidden_size": 8,
                "max_norm": 2,
                "num_layers": 1,
                "dense_size": 32,
                "dropout": 0,
            },
            "window_size": 100,
            "default_char": "~",
            "batch_size": 128,
            "num_workers": 0,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss_fn": "cross_entropy",
            "epoches": 0,
            "save_freq": 1,
            "verbose": 1
        }
        self.configs_dir = configs_dir
        self.configs = self.json_load_default(configs_dir, self.default_configs)
        if torch.cuda.is_available():
            print("Training on GPU")
            self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"
        self.data_dir = self.configs["data_dir"]
        self.model_name = self.configs["model_name"]
        self.verbose = self.configs["verbose"]
        self.model_dir = os.path.join("models", f"{self.model_name}.pth")
        self.char_map = self.configs.get("char_map", None)
        self.dataset = self.get_dataset()
        self.char_map = self.dataset.char_map
        self.configs["char_map"] = self.char_map
        self.configs["args_model"]["input_size"] = len(self.char_map)
        self.model = self.get_model()
    
    def train(self):
        configs = self.configs  

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f'Training data file does not exist at {self.data_dir}')
        
        dataloader = self.get_dataloader()
        
        model_dir = self.model_dir
        model = self.model
        optimizer = self.get_optimizer(configs["optimizer"], model.parameters(), configs["lr"])
        loss_fn = self.get_loss_fn(configs["loss_fn"])
        epoches = configs["epoches"]
        save_freq = configs["save_freq"]
        
        self.train_model(
            model=model, dataloader_train=dataloader, dataloader_test=None, optimizer=optimizer,
            loss_fn=loss_fn, epoches=epoches, save_freq = save_freq)
        
        model.save(model_dir)
        self.json_save(self.configs_dir, configs)
        print(f"Training completed. Model is saved to {model_dir}")
        
    def generate_text(self, model, prompt, length, random_state = None):
        if not prompt:
            raise ValueError("Model received an empty prompt.")
        if model is None:
            model = self.model
        model.eval()
        h, c = None, None
        self.vprint(f"Input:\n\t{prompt}\nGeneration:\n\t{prompt}", {"end":""}, cond=0)
        
        if random_state is not None:
            np.random.seed(random_state)
        
        dataset = self.dataset
        next_char = None
        for i in range(length):
            text_in = prompt if next_char is None else next_char
            encoded_text_in = dataset.encode(text_in)
            logits, h, c = model(encoded_text_in, h, c)
            probs = nn.functional.softmax(logits, dim=0).detach().cpu().numpy()            
            next_char = np.random.choice(dataset.char_map, p=probs)
            self.vprint(next_char, {"end":""}, cond=0)
    
    def get_checkpoint(self):
        model_dir = os.path.join("models", f"{self.model_name}_best.pth")
        return self.get_model(model_dir)
        
    
    def get_dataset(self):
        with open(self.data_dir, 'r', encoding='utf-8') as f:
            text = f.read()        
        configs = self.configs
        window_size = configs["window_size"]
        char_map = self.char_map
        max_map_size = configs["args_model"]["input_size"]
        default_char = configs["default_char"]
        return CharWindowDataset(
            text, window_size=window_size, max_map_size=max_map_size,
            char_map=char_map, default_char=default_char, device=self.device)
         
    
    def get_dataloader(self):
        dataset = self.dataset
        configs = self.configs 
        batch_size = configs["batch_size"]
        num_workers = configs["num_workers"]
        return data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    
    def get_model(self, model_dir = None):
        configs = self.configs
        if model_dir is None:
            model_dir = self.model_dir
        args_model = configs["args_model"]
        model = CharModel(device=self.device, **args_model)
        if os.path.exists(model_dir):
            model.load_data(model_dir)
        return model
    
    def get_optimizer(self, optimizer_name, model_parameters, lr=0.001):
        if optimizer_name in self.optimizers:
            return self.optimizers[optimizer_name](model_parameters, lr=lr)
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")

    def get_loss_fn(self, fn_name):
        if fn_name in self.loss_fns:
            return self.loss_fns[fn_name]()
        raise ValueError(f"Loss function '{fn_name}' is not supported.")
    
    
    def train_model(
        self, model, dataloader_train, dataloader_test, optimizer,
        loss_fn, epoches, save_freq):
        checkpoint_dir = os.path.join("models", f"{self.model_name}_best.pth")
        print("Start Training.")
        min_loss = np.inf
        for e in range(epoches):
            self.vprint(f"Epoches {e+1}/{epoches}:", cond = 1)
            cur_loss = self.compute_loss(model, dataloader_train, loss_fn, optimizer, mode="train")
            self.vprint(f"\tLoss (training): {cur_loss}", cond=1)
            if dataloader_test is not None:
                cur_loss = self.compute_loss(model, dataloader_test, loss_fn, optimizer, mode="eval")
                self.vprint(f"\tLoss (testing): {cur_loss}", cond=1)
                
            if e%save_freq == 0:
                if cur_loss < min_loss:
                    self.vprint("Auto Save:", cond=1)
                    model.save(checkpoint_dir)
                    min_loss = cur_loss
                    self.vprint(f"\tCheckpoint saved to {checkpoint_dir}", cond=1)
                

    def compute_loss(self, model, dataloader, loss_fn, optimizer=None, mode = "eval"):
        if mode == "train":
            model.train()
        else:
            model.eval()
            
        if self.verbose > 0:
            dl_iter = tqdm(dataloader)
        else:
            dl_iter = dataloader
        
        losses = torch.zeros(len(dataloader), dtype=torch.float32)
        for i, (X_batch, y_batch) in enumerate(dl_iter):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            loss = loss_fn(model.predict(X_batch), y_batch)
            if (mode == "train"):
                for param in model.parameters():
                    param.grad = None
                loss.backward()
                optimizer.step()
            losses[i] = loss.item()
        return torch.mean(losses)
    
    
    def vprint(self, text, print_args:dict = dict(), cond:int = 1):
        if self.verbose >= cond:
            print(text, **print_args)
        
    
    def json_load_default(self, json_dir: str, default: dict):
        if os.path.exists(json_dir):
            with open(json_dir, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return default
    
    def json_save(self, json_dir: str, contents: dict):
        os.makedirs(os.path.dirname(json_dir), exist_ok=True)
        with open(json_dir, 'w', encoding='utf-8') as f:
            json.dump(contents, f, ensure_ascii=False, indent=4)
        

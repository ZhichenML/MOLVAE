import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import GrammarVariationalAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm
import csv
from collections import defaultdict
import ast
import math

def load_model(model, save_dir):
    ckpt = torch.load(save_dir)
    model.load_state_dict(ckpt['model_state_dict'])
    print('Loaded from {}'.format(save_dir))
    return model
    

class ZincCharacterModel(object):
    """_summary_

    used in smiles to smiles generation (reconstruction)
    """    
    def __init__(self, model, latent_rep_size=56):
        self.MAX_LEN = 120
        self.vae = model
        self.charlist = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
                         '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
                         '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix
        # self.vae.load(self.charlist, weights_file, max_length=self.MAX_LEN, latent_rep_size=latent_rep_size)

    def encode(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in smiles]
        one_hot = np.zeros((len(indices), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        self.batch_size = len(one_hot)
        self.one_hot = torch.tensor(one_hot).to(device)
        mu, log_var = self.vae.encoder(self.one_hot)
        return mu, log_var

    def decode(self, mu, log_var):
        """ Sample from the character decoder """
        # assert z.ndim == 2
        out = self.vae.dec(self.one_hot, mu, log_var).cpu().detach().numpy()
        # noise = np.random.gumbel(size=out.shape)
        sampled_chars = np.argmax(np.log(out) , axis=-1) # + noise
        char_matrix = np.array(self.charlist)[np.array(sampled_chars, dtype=int)]
        return [''.join(ch).strip() for ch in char_matrix]

def test_generation():
    smiles = ["C[C@@H]1CN(C(=O)c2cc(Br)cn2C)CC[C@H]1[NH3+]",
            "CC[NH+](CC)[C@](C)(CC)[C@H](O)c1cscc1Br",
            "O=C(Nc1nc[nH]n1)c1cccnc1Nc1cccc(F)c1",
            "Cc1c(/C=N/c2cc(Br)ccn2)c(O)n2c(nc3ccccc32)c1C#N",
            "CSc1nncn1/N=C\c1cc(Cl)ccc1F"]

    # ? trained on GPU and loaded on CPU, not sure if would cause problem
    vae = GrammarVariationalAutoEncoder()
    char_weights = "checkpoints/GrammarVAE/2022-11-07-09-33model-263400.pt"
    vae = load_model(vae, char_weights).to(device)
    vae.eval()

    char_model = ZincCharacterModel(vae)

    # 4. encode and decode
    z2, log_var = char_model.encode(smiles)
    for mol in char_model.decode(z2, log_var):
        print(mol)
 

class property_prediction_net(nn.Module):
    def __init__(self, input_size=56, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size,1)
                             )

    def forward(self, x):
        
        return torch.squeeze(self.net(x))
        
    
    
    
class Dataset:
    """_summary_
    readin raw zinc250k smiles and converts to one-hot numpy array,
    then dump to disk waiting for training.
    """        
    def __init__(self):
        # raw dataset
        # self.train_prop_file = './data/zinc/train.logP-SA'
        # self.train_data_file = './data/zinc/train.txt'
        # self.val_prop_file = './data/zinc/opt.valid.logP-SA'
        # self.val_data_file = './data/zinc/valid.txt'
        # self.test_prop_file = './data/zinc/opt.test.logP-SA'
        # self.test_data_file = './data/zinc/test.txt'
        
        # self.train = 'data/zinc/train.pkl'
        # self.val = 'data/zinc/val.pkl'
        # self.test=  'data/zinc/test.pkl'
        
        
        
        # char setting
        self.MAX_LEN = 120
        self.charlist = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
                         '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
                         '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix
        

    def _load_raw_data(self, path):
        with open(path) as f:
            train_data = [line.strip("\r\n ").split()[0] for line in f]
        return train_data
        
    def load_label(self):
        import pdb; pdb.set_trace()
        train_y, val_y, test_y = np.loadtxt(self.train_prop_file), np.loadtxt(self.val_prop_file),  np.loadtxt(self.test_prop_file)
        return torch.tensor(train_y), torch.tensor(val_y), torch.tensor(test_y)
        
        # val_y = np.loadtxt(val_prop_file)
        # with open(val_data_file) as f:
        #     val_data = [line.strip("\r\n ").split()[0] for line in f]
            
        # test_y = np.loadtxt(test_prop_file)
        # with open(test_data_file) as f:
        #     test_data = [line.strip("\r\n ").split()[0] for line in f]
            
    def _process_raw_data(self, path):
        data = self._load_raw_data(path)
        one_hot = self._one_hot(data)
        
        return np.array(one_hot)
        
    def data_pipeline(self):
        train_one_hot= self._process_raw_data(self.train_data_file)
        self._save_processed_data(train_one_hot, './data/zinc/train.pkl')
        
        val_one_hot = self._process_raw_data(self.val_data_file)
        self._save_processed_data(val_one_hot, './data/zinc/val.pkl')
        
        test_one_hot = self._process_raw_data(self.test_data_file)
        self._save_processed_data(test_one_hot, './data/zinc/test.pkl')
         
        
    def _one_hot(self, smiles):
        # one-hot
        indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in smiles]
        one_hot = np.zeros((len(indices), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        return one_hot

    def _save_processed_data(self, data, path):
        with open(path, 'wb') as fout:
            pickle.dump(data, fout)
        print('Saved processed data')

    def load_data(self):
        with open(self.train, "rb") as fin:
            train = pickle.load(fin)
        with open(self.train, "rb") as fin:
            val = pickle.load(fin)
        with open(self.train, "rb") as fin:
            test = pickle.load(fin)
        return torch.tensor(train), torch.tensor(val), torch.tensor(test)
    


class Session():
    def __init__(self, model, vae, train_step_init=0, lr=1e-2, is_cuda=False):
        self.train_step = train_step_init
        self.model = model
        self.vae = vae
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=3, min_lr=0.0001)
        self.loss_fn = nn.MSELoss()
        # self.dashboard = Dashboard('Grammar-Variational-Autoencoder-experiment')

    def train(self, loader):
        # built-in method for the nn.module, sets a training flag.
        size = len(loader.dataset)
        
        self.model.train()
        _losses = []

       
        for batch_idx, data in enumerate(tqdm(loader)):
            
            smiles, logp, qed = data[0].to(device).to(torch.float32), data[1].to(device).to(torch.float32), data[2].to(device).to(torch.float32)
            batch_size = len(logp)
            # have to cast data to FloatTensor. DoubleTensor errors with Conv1D
            mu, log_var = self.vae.encoder(smiles)
            
            pred = self.model(mu)
            
            self.optimizer.zero_grad()
            loss = self.loss_fn(pred, logp)
            loss.backward()
            self.optimizer.step()
            self.train_step += 1
            
            _losses.append(loss.cpu().item())
            
          
            
            # self.dashboard.append('training_loss', 'line',
            #                       X=np.array([self.train_step]),
            #                       Y=loss_value / batch_size)
            
            # if batch_idx == 0:
            #     print('batch size', batch_size)
            # if batch_idx % 40 == 0:
            #     print('training loss: {:.4f}'.format(loss_value))# / batch_size))
        print('train num batch: ', len(loader))
        
        return np.mean(_losses)

    def test(self, loader):
        # nn.Module method, sets the training flag to False
        
        size = len(loader.dataset)
        num_batch = len(loader)
        test_loss = 0
        self.model.eval()
        for batch_idx, data in enumerate(loader):
            # data = Variable(data, volatile=True).to(device)
            
            smiles, logp, qed = data[0].to(device).to(torch.float32), data[1].to(device).to(torch.float32), data[2].to(device).to(torch.float32)
            mu, log_var = self.vae.encoder(smiles)
            
            pred = self.model(mu)
            loss = self.loss_fn(pred, logp)
            test_loss += loss.cpu().item()
            
            
        
        test_loss /= num_batch
        # print(' testset length', size)
        # print(' ====> Test set loss: {:.4f}'.format(test_loss))
        
        return test_loss
    
    def save_model_by_name(self, model):
        save_dir = os.path.join('checkpoints', model.name)
    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, t+'model-{:05d}.pt'.format(self.train_step))
        state = model.state_dict()
        torch.save({'model_state_dict': state, 'optimizer_state_dict': self.optimizer.state_dict()}, file_path)
        print('Saved to {}'.format(file_path))
        
    def load_model(self, save_dir):
        ckpt = torch.load(save_dir)
        
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_step = int(save_dir.strip().split('/')[-1].split('-')[-1][:-3])
        print('Loaded from {}'.format(save_dir))
        
    

def property_prediction(train, val, test):
    # 1. create property predictor
    net = property_prediction_net().to(device)
    
    # 2. load encoding model
    vae = GrammarVariationalAutoEncoder().to(device)
    char_weights = "checkpoints/GrammarVAE/2022-11-07-09-33model-263400.pt"
    vae = load_model(vae, char_weights)
    vae.eval()    
    
    # 3. load pre_processed pytorch.utils.data.Dataset
    batch_szie = 512
    train_loader, val_loader, test_loader = DataLoader(train, batch_szie), DataLoader(val, batch_szie), DataLoader(test, batch_szie)
    
    # 4. create training session
    epoches = 10
    sess = Session(net, vae)
    for t in range(epoches):
        train_loss = sess.train(train_loader)
        val_loss = sess.test(val_loader)
        sess.scheduler.step(val_loss)
        test_loss = sess.test(test_loader)
        print('==================Epoch {} complete. ==================\n'+
              '===> train loss: {}\n'+
              '===> val_loss:{}\n' + 
              '===> test_loss: {}\n'.format(t, train_loss, val_loss, test_loss))
    
    print('done')


class zinc_dataset(torch.utils.data.Dataset):
    def __init__(self, smiles, targets):
        
        self.smiles = smiles
        self.targets = targets
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return self.smiles[idx], self.targets['logP'][idx], self.targets['qed'][idx]
    
def load_csv(csv_file='./data/zinc/250k_rndm_zinc_drugs_clean_3.csv', smiles_field="smiles", target_fields=["logP", "qed"], verbose=0, **kwargs):
    """
    Load the dataset from a csv file.

    Parameters:
        csv_file (str): file name
        smiles_field (str, optional): name of SMILES column in the table.
            Use ``None`` if there is no SMILES column.
        target_fields (list of str, optional): name of target columns in the table.
            Default is all columns other than the SMILES column.
        verbose (int, optional): output verbose level
        **kwargs
    """
    
    def literal_eval(string):
        
        """
        Evaluate an expression into a Python literal structure.
        """
        try:
            return ast.literal_eval(string)
        except (ValueError, SyntaxError):
            return string
        
    if target_fields is not None:
        target_fields = set(target_fields)

    with open(csv_file, "r") as fin:
        
        reader = csv.reader(fin)
        fields = next(reader)
        smiles = []
        targets = defaultdict(list)
        for values in reader:
            if not any(values):
                continue
            if smiles_field is None:
                smiles.append("")
            for field, value in zip(fields, values):
                if field == smiles_field:
                    smiles.append(value.strip())
                elif target_fields is None or field in target_fields:
                    value = literal_eval(value)
                    if value == "":
                        value = math.nan
                    targets[field].append(value)
    return smiles, targets

def split_data():
    smiles, targets = load_csv()
   
    pipline = Dataset()
    smiles = pipline._one_hot(smiles)

    train_data = smiles[:199564]
    train_targets = dict()
    train_targets['logP'], train_targets['qed'] = targets['logP'][:199564], targets['qed'][:199564]
    train = zinc_dataset(train_data, train_targets) 
    
    val_data = smiles[199564:199564+24946]
    val_targets = dict()
    val_targets['logP'], val_targets['qed'] = targets['logP'][199564:199564+24946], targets['qed'][199564:199564+24946]
    val = zinc_dataset(val_data, val_targets)
    
    test_data = smiles[199564+24946:]
    test_targets = dict()
    test_targets['logP'], test_targets['qed'] = targets['logP'][199564+24946:], targets['qed'][199564+24946:]
    test = zinc_dataset(test_data, test_targets)
    
    with open('data/zinc/property/train.pkl', 'wb') as fout:
        pickle.dump(train,fout)
    with open('data/zinc/property/val.pkl', 'wb') as fout:
        pickle.dump(val, fout)
    with open('data/zinc/property/test.pkl', 'wb') as fout:
        pickle.dump(test, fout)
    
if __name__ == '__main__':
    # split_data() # only need done once
    with open('data/zinc/property/train.pkl', 'rb') as fin:
        train = pickle.load(fin)
    with open('data/zinc/property/val.pkl', 'rb') as fin:
        val = pickle.load(fin)
    with open('data/zinc/property/test.pkl', 'rb') as fin:
        test = pickle.load(fin)

    property_prediction(train, val, test)
        
        
    
    
    
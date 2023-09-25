import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

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

from pytorchtools import EarlyStopping

from utils import setup_seed, load_model
from torch.utils.tensorboard import SummaryWriter
import datetime 
t = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))




class property_prediction_net(nn.Module):
    
    def __init__(self, input_size=56, hidden_size=[1024,768,512,128,64]):
        super().__init__()
        self.name = 'property_prediction_net'
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size[0]),
                             nn.ELU(),
                             nn.Linear(hidden_size[0],hidden_size[1]),                             
                             nn.ELU(),
                             nn.Linear(hidden_size[1], hidden_size[2]),
                            #  nn.Dropout(0.5),
                             nn.ELU(),
                             nn.Linear(hidden_size[2], hidden_size[3]),
                             nn.Dropout(0.5),
                             nn.ELU(),
                             nn.Linear(hidden_size[3], hidden_size[4]),
                             nn.Dropout(0.5),
                             nn.ELU(),
                             nn.Linear(hidden_size[4],1)
                             )

    def forward(self, x):
        # ? prediction is otherwise higher rank than label
        return torch.squeeze(self.net(x))
        
class Dict_Embedding:
    """_summary_
    readin raw zinc250k smiles and converts to one-hot numpy array,
    then dump to disk waiting for training.
    """        
    def __init__(self):

        # char setting
        self.MAX_LEN = 120
        self.charlist = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
                         '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
                         '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix
            
    def _process_raw_data(self, path):
        data = self._load_raw_data(path)
        one_hot = self._one_hot(data)
        
        return np.array(one_hot)
    
    def _save_processed_data(self, data, path):
        with open(path, 'wb') as fout:
            pickle.dump(data, fout)
        print('Saved processed data')
        
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
    
    def load_data(self):
        """_summary_
            used for downstream output api
        Returns:
            _type_: _description_
        """        
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
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)#
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.6, patience=3, min_lr=3e-4)
        # self.scheduler =  optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98) # lr * gamma**epoch
        # self.scheduler = None
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        self.loss_fn = nn.MSELoss()
        self.log = SummaryWriter('./checkpoints/property_prediction_net/log/'+t)
        

    def train(self, loader):
        
        size = len(loader.dataset)
        self.model.train()
        self.vae.eval()
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
        self.vae.eval()
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
    
    def demo_result(self, test_loader):
        data = next(iter(test_loader))
        smiles, logp, qed = data[0].to(device).to(torch.float32), data[1].to(device).to(torch.float32), data[2].to(device).to(torch.float32)
        mu, log_var = self.vae.encoder(smiles)
        
        pred = self.model(mu)
        import pdb; pdb.set_trace()
        ret = pred - logp
        print('compare prediction and target: ', ret)
            
        
    
    def save_model_by_name(self):
        save_dir = os.path.join('checkpoints', self.model.name)
    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, t+'model-{:05d}.pt'.format(self.train_step))
        state = self.model.state_dict()
        torch.save({'model_state_dict': state, 'optimizer_state_dict': self.optimizer.state_dict()}, file_path)
        print('Saved to {}'.format(file_path))
        
    def load_model(self, save_dir):
        ckpt = torch.load(save_dir)
        
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_step = int(save_dir.strip().split('/')[-1].split('-')[-1][:-3])
        print('Loaded from {}'.format(save_dir))
    
    


class zinc_dataset(torch.utils.data.Dataset):
    """_summary_

    Args:
        np.array: one-hot embeddings of smiles
        python dict: the regressor's label
    """    
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
   
    pipline = Dict_Embedding()
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
    early_stop = EarlyStopping()
    epoches = 600
    sess = Session(net, vae)
    # sess.load_model('checkpoints/property_prediction_net/2022-11-17-10-04model-39000.pt')
    # sess.demo_result(test_loader)
    for t in range(epoches):
        train_loss = sess.train(train_loader)
        val_loss = sess.test(val_loader)
        if early_stop(val_loss).early_stop: print('Early stopping'); break
        if sess.scheduler: sess.scheduler.step(val_loss) #
        test_loss = sess.test(test_loader)
        # * tensorboarding here
        sess.log.add_scalar('loss/train_loss', train_loss, t)
        sess.log.add_scalar('loss/val_loss', val_loss, t)
        sess.log.add_scalar('loss/test_loss', test_loss, t)
        sess.log.add_scalar('loss/learning rate', sess.optimizer.state_dict()['param_groups'][0]['lr'], t) 
        print(('==================Epoch {} complete. ==================\n'+
              '===> train loss: {}\n'+
              '===> val_loss:{}\n' + 
              '===> test_loss: {}\n').format(t, train_loss, val_loss, test_loss))
    
    sess.save_model_by_name()
    print('done')

def get_embedding(train, val, test):
    # 2. load encoding model
    vae = GrammarVariationalAutoEncoder().to(device)
    char_weights = "checkpoints/GrammarVAE/2022-11-07-09-33model-219500.pt"
    vae = load_model(vae, char_weights)
    vae.eval()    
    
    # 3. load pre_processed pytorch.utils.data.Dataset
    batch_szie = 512
    train_loader, val_loader, test_loader = DataLoader(train, batch_szie), DataLoader(val, batch_szie), DataLoader(test, batch_szie)
    
    embeddings = []
    train_logp = []

    for data in tqdm(train_loader):
            
            smiles, logp, qed = data[0].to(device).to(torch.float32), data[1].to(device).to(torch.float32), data[2].to(device).to(torch.float32)
            batch_size = len(logp)
            # have to cast data to FloatTensor. DoubleTensor errors with Conv1D
            mu, log_var = vae.encoder(smiles)
            
            embeddings.extend(list(mu.cpu().detach().numpy()))
            train_logp.extend(list(logp.cpu().detach().numpy()))
    
    embeddings = np.array(embeddings)
    train_logp = np.array(train_logp)
    with open('./data/zinc/property/train_embeddings.pkl', 'wb') as fout:
        pickle.dump(embeddings, fout)
    with open('./data/zinc/property/train_logp.pkl', 'wb') as fout:
        pickle.dump(train_logp, fout)
 
def visual_embeddings():
    with open('./data/zinc/property/train_embeddings.pkl', 'rb') as fin:
        embeddings = pickle.load(fin)
    with open('./data/zinc/property/train_logp.pkl', 'rb') as fin:
        logp = pickle.load(fin)
    from sklearn import manifold
    import pylab
    
    tsne = manifold.TSNE(perplexity=20)
    Y = tsne.fit_transform(embeddings[:10000,:])
    pylab.scatter(Y[:, 0], Y[:, 1], 20, logp[:10000])
    pylab.show()
    pylab.savefig('./fig.jpg')

    
if __name__ == '__main__':
    # split_data() # only need done once
    with open('data/zinc/property/train.pkl', 'rb') as fin:
        train = pickle.load(fin)
    with open('data/zinc/property/val.pkl', 'rb') as fin:
        val = pickle.load(fin)
    with open('data/zinc/property/test.pkl', 'rb') as fin:
        test = pickle.load(fin)

    setup_seed(49)
    # get_embedding(train,val,test)
    # visual_embeddings()
    property_prediction(train, val, test)
        
        
    
    
    
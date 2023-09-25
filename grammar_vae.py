import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import numpy as np
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import GrammarVariationalAutoEncoder, VAELoss
import pickle

from utils import setup_seed
# from visdom_helper.visdom_helper import Dashboard
import os
os.environ['HDF5_USE_FILE_LOCKING']='False'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print('Using: ', device)
import datetime
t = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))



class Session():
    def __init__(self, model, train_step_init=0, lr=1e-2, is_cuda=False):
        self.train_step = train_step_init
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.6, patience=3, min_lr=0.0001)
        self.loss_fn = VAELoss()
        # self.dashboard = Dashboard('Grammar-Variational-Autoencoder-experiment')

    def train(self, loader, epoch_number):
        # built-in method for the nn.module, sets a training flag.
        self.model.train()
        train_correct = 0
        train_total = 0
        _losses = []
 
        _BCE, _KLD = [], []
        num_batch = 0
        for batch_idx, data in enumerate(tqdm(loader)):
            # have to cast data to FloatTensor. DoubleTensor errors with Conv1D
            data = Variable(data).to(device)
            # do not use CUDA atm
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
        
            
            loss, BCE, KLD = self.loss_fn(data, mu, log_var, recon_batch, batch_idx)
            # print('BCE: ' + str(BCE) +  "KLD: " + str(KLD))
            
            loss.backward()
            self.optimizer.step()
            self.train_step += 1
            
            _losses.append(loss.cpu().item())
            _BCE.append(BCE.cpu().item())
            _KLD.append(KLD.cpu().item())
            num_batch += 1
            loss_value = loss.cpu().data.numpy()
            batch_size = len(data)
            
            predict = torch.argmax(recon_batch, -1)
            labels = torch.argmax(data, -1)
            train_correct += (predict == labels).sum().cpu().item()
            train_total += labels.size()[0]*labels.size()[1]

            # self.dashboard.append('training_loss', 'line',
            #                       X=np.array([self.train_step]),
            #                       Y=loss_value / batch_size)
            
            # if batch_idx == 0:
            #     print('batch size', batch_size)
            # if batch_idx % 40 == 0:
            #     print('training loss: {:.4f}'.format(loss_value))# / batch_size))
        acc = train_correct / train_total
        print('train num batch: ', num_batch)
        
        return np.mean(_losses), acc, np.mean(_BCE), np.mean(_KLD)

    def test(self, loader):
        # nn.Module method, sets the training flag to False
        self.model.eval()
        train_correct = 0
        train_total = 0
        test_loss = 0
        _BCE, _KLD=0,0
        num_batch = 0
        for batch_idx, data in enumerate(loader):
            # data = Variable(data, volatile=True).to(device)
            data = data.to(device)
            # do not use CUDA atm
            recon_batch, mu, log_var = self.model(data)
            loss, BCE, KLD= self.loss_fn(data, mu, log_var, recon_batch)
            test_loss += loss.cpu().item()
            _BCE += BCE.cpu().item()
            _KLD += KLD.cpu().item()
            num_batch += 1
            
            
            predict = torch.argmax(recon_batch, -1)
            labels = torch.argmax(data, -1)
            train_correct += (predict == labels).sum().cpu().item()
            train_total += labels.size()[0]*labels.size()[1]
        
        
        test_loss /= num_batch
        print(' testset length', len(test_loader.dataset))
        print(' ====> Test set loss: {:.4f}'.format(test_loss))
        acc = train_correct / train_total
        return test_loss, acc, _BCE/num_batch, _KLD/num_batch
    
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
        


EPOCHS = 600
BATCH_SIZE = 512
# import h5py

setup_seed(49)

def kfold_loader(k, s, e=None):
    if not e:
        e = k
    with open("data/zinc_str_dataset.pkl", "rb") as fin:
        data = pickle.load(fin)
    result = np.concatenate([data[i::k] for i in range(s, e)])
    return torch.FloatTensor(result)
    # with h5py.File('data/zinc_str_dataset.h5', 'r') as h5f:
    #     result = np.concatenate([h5f['data'][i::k] for i in range(s, e)])
    #     return torch.FloatTensor(result)


train_loader = torch.utils.data \
    .DataLoader(kfold_loader(10, 1),
                batch_size=BATCH_SIZE, shuffle=False)
# todo: need to have separate training and validation set
test_loader = torch.utils \
    .data.DataLoader(kfold_loader(10, 0, 1),
                     batch_size=BATCH_SIZE, shuffle=False)


vae = GrammarVariationalAutoEncoder().to(device)

writer = SummaryWriter('./checkpoints/log/'+t)
sess = Session(vae, lr=1e-2)
# sess.load_model('./checkpoints/GrammarVAE/2022-11-06-17-19model-131700.pt')
# sess.train_step = 131700
for epoch in range(1, EPOCHS + 1):
    train_losses, train_acc, train_BCE, train_KLD=sess.train(train_loader, epoch)
    
    test_loss, test_acc, test_BCE, test_KLD = sess.test(test_loader)
    sess.scheduler.step(test_loss)
    print('epoch {} complete, train_loss： {}， test_loss: {}, ---train_BCE: {}, train_KLD: {}, acc: {}, test_acc: {}'.format(epoch, 
                                                                                            train_losses, test_loss, train_BCE, train_KLD, train_acc, test_acc))
    
    writer.add_scalar('train_loss/loss', train_losses, epoch)
    writer.add_scalar('train_loss/BCE', train_BCE, epoch)
    writer.add_scalar('train_loss/KLD', train_KLD, epoch)
    writer.add_scalar('test_loss/loss', test_loss, epoch)
    writer.add_scalar('test_loss/BCE', test_BCE, epoch)
    writer.add_scalar('test_loss/KLD', test_KLD, epoch)
    writer.add_scalar('train_acc', train_acc, epoch)
    writer.add_scalar('test_acc', test_acc, epoch)
    

sess.save_model_by_name(vae)

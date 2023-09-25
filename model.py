import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

charset = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[', '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/', '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
charset_length = len(charset)
Lantent_dim = 56
max_length = 120
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, input_size=56, hidden_n=56, output_feature_size=charset_length, max_seq_length=max_length):
        super(Decoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.hidden_n = hidden_n
        self.output_feature_size = output_feature_size
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc_input = nn.Linear(input_size, hidden_n)
        # we specify each layer manually, so that we can do teacher forcing on the last layer.
        # we also use no drop-out in this version.
        self.gru_1 = nn.GRU(input_size=input_size, hidden_size=501, batch_first=True)
        self.gru_2 = nn.GRU(input_size=501, hidden_size=501, batch_first=True)
        self.gru_3 = nn.GRU(input_size=501, hidden_size=501, batch_first=True)
        self.fc_out = nn.Linear(501, charset_length)

    def forward(self, encoded, hidden_1, hidden_2, hidden_3, beta=0.3, target_seq=None):
        _batch_size = encoded.size()[0]
        
        embedded = F.relu(self.fc_input(self.batch_norm(encoded))) \
            .view(_batch_size, 1, -1) \
            .repeat(1, self.max_seq_length, 1)
        # batch_size, seq_length, hidden_size; batch_size, hidden_size
        out_1, hidden_1 = self.gru_1(embedded, hidden_1)
        out_2, hidden_2 = self.gru_2(out_1, hidden_2)
        # NOTE: need to combine the input from previous layer with the expected output during training.
        if self.training and target_seq:
            out_2 = out_2 * (1 - beta) + target_seq * beta
        out_3, hidden_3 = self.gru_3(out_2, hidden_3)
        
        out = self.fc_out(out_3.contiguous().view(-1, 501)).view(_batch_size, self.max_seq_length, self.output_feature_size)
        
        # return F.softmax(out, dim=-1), hidden_1, hidden_2, hidden_3
        return F.relu(torch.sigmoid(out)), hidden_1, hidden_2, hidden_3

    def init_hidden(self, batch_size):
        # NOTE: assume only 1 layer no bi-direction
        h1 = Variable(torch.zeros(1, batch_size, 501), requires_grad=False).to(device)
        h2 = Variable(torch.zeros(1, batch_size, 501), requires_grad=False).to(device)
        h3 = Variable(torch.zeros(1, batch_size, 501), requires_grad=False).to(device)
        return h1, h2, h3


class Encoder(nn.Module):
    def __init__(self, k1=9, k2=9, k3=11, hidden_n=56):
        super(Encoder, self).__init__()
        # NOTE: GVAE implementation does not use max-pooling. Original DCNN implementation uses max-k pooling.
        self.conv_1 = nn.Conv1d(in_channels=35, out_channels=9, kernel_size=k1)#, groups=12)
        self.bn_1 = nn.BatchNorm1d(9)
        self.conv_2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=k2)#, groups=12)
        self.bn_2 = nn.BatchNorm1d(9)
        self.conv_3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=k3)#, groups=12)
        self.bn_3 = nn.BatchNorm1d(10)

        # todo: harded coded because I can LOL
        self.fc_0 = nn.Linear(940, 435)
        self.fc_mu = nn.Linear(435, hidden_n)
        self.fc_var = nn.Linear(435, hidden_n)

    def forward(self, x):
        
        batch_size = x.size()[0]
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x_ = x.view(batch_size, -1)
        h = F.relu(self.fc_0(x_))
        m = self.fc_mu(h)
        log_var = self.fc_var(h) 
        if (log_var>60).sum()>1:
            import pdb; pdb.set_trace()
        return m, log_var


# from visdom_helper.visdom_helper import Dashboard


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_loss.size_average = False
        # self.dashboard = Dashboard('Variational-Autoencoder-experiment')

    # question: how is the loss function using the mu and variance?
    def forward(self, x, mu, log_var, recon_x, b=None):
        """gives the batch normalized Variational Error."""

        batch_size = x.size()[0]
        BCE = self.bce_loss(recon_x, x) * max_length

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.mean(KLD_element).mul_(-0.5)
        # if b == 37:
        #     import pdb; pdb.set_trace()
        if abs(KLD)>1000:
            print('KLD: ', KLD)
            print(KLD_element)
            import time; time.sleep(1)
        
        
        
        return (BCE + KLD) , BCE.detach(), KLD.detach()


class GrammarVariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(GrammarVariationalAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.name = 'GrammarVAE'

    def forward(self, x):
        batch_size = x.size()[0]
        mu, log_var = self.encoder(x)
        # todo multiple sampling
        z = self.reparameterize(mu, log_var)
        h1, h2, h3 = self.decoder.init_hidden(batch_size)
        output, h1, h2, h3 = self.decoder(z, h1, h2, h3)
        
        return output, mu, log_var

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        print('mu: ' + str(mu) + 'log_var: ' + str(log_var))
        import time; time.sleep(0.1)
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_(mean=0, std=1)).to(device)
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)

    def enc(self, x):
        batch_size = x.size()[0]
        mu, log_var = self.encoder(x)
        return mu, log_var
        
    def dec(self, x, mu, log_var):
        batch_size = x.size()[0]
        z = self.reparameterize(mu, log_var)
        h1, h2, h3 = self.decoder.init_hidden(batch_size)
        output, h1, h2, h3 = self.decoder(z, h1, h2, h3)
        return output
        
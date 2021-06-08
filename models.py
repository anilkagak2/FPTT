
import os
import shutil
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
            
def ptb_w_save_checkpoint(model, optimizer, path, finetune=False):
    print('Save path = ', os.path.join(path, 'model.pt'))
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer, os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer, os.path.join(path, 'optimizer.pt'))

def model_save(fn, model, criterion, optimizer):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)
    return model, criterion, optimizer

def save_checkpoint(state, is_best, prefix, filename='_checkpoint.pth.tar'):
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_model_best.pth.tar')

def ptb_count_parameters(model):
    return sum(p.numel() for p in model.rnn.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.network[0].parameters() if p.requires_grad)

class DecoupledRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_timesteps, P=10):
        super(DecoupledRNN, self).__init__()
        
        print('DecoupledRNN ', P)
        wdrop = 0.2 #0.5
        self.use_dropout = True
        
        self.P = P
        self.step = n_timesteps // self.P
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.n_timesteps = n_timesteps
        
        self.rnn_name = 'LSTM'
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        
    def forward(self, inputs, h=None):
        T = inputs.size()[0]
        
        outputs = []
        hiddens = []
        next_tokens = []
        
        for i in range(self.P):            
            rnn = self.rnn
            
            start = i*self.step
            end = (i+1)*self.step
            
            reach_end_sentence = False
            if (end >= T) or (i==self.P-1): 
                end = T
                reach_end_sentence = True

            x = inputs[ start : end ]         
                        
            o, h = rnn( x, h )

            outputs.append(o)
            hiddens.append(h)
            if start+1 < T:
                next_tokens.append( inputs[start+1] )
                
            if reach_end_sentence:
                break
                
            if self.training:
                if p==(i+1):
                    break
        
        output = torch.cat( outputs, dim=0 )
        final_state = h
        
        return output, final_state, hiddens, next_tokens


class SeqModel(nn.Module):
    def __init__(self, ninp, nhid, nout, dropout=0.0, dropouti=0.0, dropouth=0.0, wdrop=0.0,
                 temporalwdrop=False, wnorm=True, n_timesteps=784, nfc=256, parts=10):

        super(SeqModel, self).__init__()
        self.nout = nout    # Should be the number of classes
        self.nhid = nhid
        self.dropout = dropout
        self.dropouti = dropouti

        self.rnn_name = 'LSTM'

        #self.network = nn.LSTM(input_size=ninp, hidden_size=nhid)
        self.network = DecoupledRNN(input_size=ninp, hidden_size=nhid, n_timesteps=n_timesteps, P=parts)
        
        # If weight normalization is used, we apply the weight dropout to its "direction", instead of "scale"
        self.linear1 = nn.Linear(nhid, nfc)
        self.linear2 = nn.Linear(nfc, nout)
        
        self.network = nn.ModuleList([self.network])
        self.l2_loss = nn.MSELoss()

    def forward(self, inputs, hidden):
        inputs = inputs.permute(2, 0, 1)  

        output, hidden, hiddens, next_tokens = self.network[0](inputs, hidden)
        
        outputs = []
        for l, h in enumerate(hiddens):
            out = ((l+1)/self.network[0].P) * F.dropout(self.linear2(self.linear1(h[0])), self.dropout)
            out = out.squeeze(dim=0)
            outputs.append( F.log_softmax(out, dim=1) )

        recon_loss = torch.zeros(1, device=inputs.device)
        return outputs, hidden, recon_loss

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_name == 'B_TARNN':
            return weight.new(1, bsz, self.nhid).zero_()
        else:
            return (weight.new(1, bsz, self.nhid).zero_(),
                    weight.new(1, bsz, self.nhid).zero_())


class PTBModel_300(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(PTBModel_300, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

from ptb_char_utils import LockedDropout, WeightDrop, embedded_dropout

class PTBWordModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nhidlast, nlayers, 
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, 
                 tie_weights=False, ldropout=0.5, n_experts=10, n_timesteps=70):
        super(PTBWordModel, self).__init__()
        self.use_dropout = True
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnn_name = 'LSTM'
        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast, 1, dropout=0) for l in range(nlayers)]       
 
        #self.rnn_name = 'B_TARNN'
        #self.rnns = [B_TARNN( ninp if l == 0 else nhid, 
        #                         nhid if l != nlayers - 1 else nhidlast, 
        #                         10*n_timesteps, ) for l in range(nlayers)]
        
        if wdrop:
            #self.rnns = [WeightDrop(rnn, ['weight_hh'], dropout=wdrop if self.use_dropout else 0) for rnn in self.rnns]
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.prior = nn.Linear(nhidlast, n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(nhidlast, n_experts*ninp), nn.Tanh())
        self.decoder = nn.Linear(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ldropout = ldropout
        self.dropoutl = ldropout
        self.n_experts = n_experts
        self.ntoken = ntoken

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('param size: {}'.format(size))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, return_prob=False):
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if (self.training and self.use_dropout) else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti if self.use_dropout else 0)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth if self.use_dropout else 0)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout if self.use_dropout else 0)
        outputs.append(output)

        latent = self.latent(output)
        latent = self.lockdrop(latent, self.dropoutl if self.use_dropout else 0)
        logit = self.decoder(latent.view(-1, self.ninp))

        prior_logit = self.prior(output).contiguous().view(-1, self.n_experts)
        prior = nn.functional.softmax(prior_logit, -1)

        prob = nn.functional.softmax(logit.view(-1, self.ntoken), -1).view(-1, self.n_experts, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(prob.add_(1e-8))
            model_output = log_prob

        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_name == 'B_TARNN':
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_())
                for l in range(self.nlayers)]
        else:
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()),
                     Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
                    for l in range(self.nlayers)]

class PTBCharModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(PTBCharModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
            
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]


class AddTaskModel(nn.Module):
    def __init__(self, rec_net, hidden_size):
        super(AddTaskModel, self).__init__()
        self.nhid = hidden_size
        self.rnn = rec_net
        self.lin = nn.Linear(hidden_size, 1) 
        self.loss_func = nn.MSELoss()

        nn.init.xavier_normal_(self.lin.weight)

    def forward(self, x, y, h):        
        loss = 0
        output, hidden = self.rnn.forward(x, h)
        out = self.lin(hidden[0])           
        out = out.squeeze()
        y  = y.squeeze()
        loss += self.loss_func(out, y.t())
        return loss, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, self.nhid),
                weight.new_zeros(1, bsz, self.nhid))



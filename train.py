
import os
import sys
import argparse
import math
import shutil
import time
import logging
from io import open

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

from ptb_utils import get_train_val_test_data, get_batch, repackage_hidden
from ptb_char_utils import ptb_char_get_train_val_test_data, ptb_char_get_batch
from ptb_char_utils import ptb_word_get_train_val_test_data #, ptb_char_get_batch
from ptb_char_utils import LockedDropout, WeightDrop, embedded_dropout, SplitCrossEntropyLoss
from utils import get_xt
from models import *
from datasets import data_generator, adding_problem_generator

def get_stats_named_params( model ):
    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = param.detach().clone(), 0.0*param.detach().clone(), 0.0*param.detach().clone()
        named_params[name] = (param, sm, lm, dm)
    return named_params

def pre_pre_optimizer_updates( named_params, args ):
    if not args.debias: return
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param_data = param.data.clone()
        param.data.copy_( sm.data )
        sm.data.copy_( param_data )
        del param_data

def pre_optimizer_updates( named_params, args ):
    if not args.debias: return
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        lm.data.copy_( param.grad.detach() )
        param_data = param.data.clone()
        param.data.copy_( sm.data )
        sm.data.copy_( param_data )
        del param_data

def post_optimizer_updates( named_params, args, epoch ):
    alpha = args.alpha
    beta = args.beta
    rho = args.rho
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        if args.debias:
            beta = (1. / (1. + epoch))
            sm.data.mul_( (1.0-beta) )
            sm.data.add_( beta * param )

            rho = (1. / (1. + epoch))
            dm.data.mul_( (1.-rho) )
            dm.data.add_( rho * lm )
        else:
            lm.data.add_( -alpha * (param - sm) )
            sm.data.mul_( (1.0-beta) )
            sm.data.add_( beta * param - (beta/alpha) * lm )

def get_regularizer_named_params( named_params, args, _lambda=1.0 ):
    alpha = args.alpha
    rho = args.rho
    regularization = torch.zeros( [], device=args.device )
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho-1.) * torch.sum( param * lm )
        if args.debias:
            regularization += (1.-rho) * torch.sum( param * dm )
        else:
            regularization += _lambda * 0.5 * alpha * torch.sum( torch.square(param - sm) )
    return regularization 

def reset_named_params(named_params, args):
    if args.debias: return
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)
        if args.dataset not in ['PTB-300', 'PTB-Char']:
            lm.data.mul_(0.0)
            dm.data.mul_(0.0)

def add_task_train_online( net, optimizer, args, named_params ):
    batch_size = args.batch_size
    n_steps = args.epochs
    c_length = args.bptt

    losses = []
    
    PARTS = args.parts #10
    step = c_length // PARTS
    logger.info('step = ' + str(step))
    
    alpha = 0.05 #0.2
    alpha1 = 0.005 #001
    alpha2 = 0.01
    
    for i in range(n_steps):        
        s_t = time.time()
        x,y = adding_problem_generator(batch_size, seq_len=c_length, number_of_ones=2)        
        x = x.cuda()
        y = y.cuda()
        data = x.transpose(0, 1)
        y = y.transpose(0, 1)
        
        net.train()
        xdata = data.clone()
        inputs = xdata
        
        T = c_length
        
        for p in range(PARTS-1):
            x, start, end = get_xt(p, step, T, inputs)
            xtp, _, _ = get_xt(p+1, step, T, inputs)
            
            if p==0:
                h = net.init_hidden(batch_size)               
            else:
                #_, h = net.rnn( inputs[:end], h )
                h = (h[0].detach(), h[1].detach())

            optimizer.zero_grad()
            loss, h = net.forward(x, y, h) 
            loss = (p+1/PARTS) *  loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

        '''
        optimizer.zero_grad()
        h = get_initial_hidden_state(net, batch_size, hidden_size)               
        x = data
        loss, _ = net.forward(x, y, h)
        loss_act = loss
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        '''
        
        
        ### Evaluate
        net.eval()
        x,y = adding_problem_generator(batch_size, seq_len=c_length, number_of_ones=2)        
        x = x.cuda()
        y = y.cuda()
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)

        h = net.init_hidden(batch_size)               
        loss, _ = net.forward(x, y, h)
        loss_act = loss
        losses.append(loss_act.item())

        if i%args.log_interval == 0:
            logger.info('Update {}, Time for Update: {} , Average Loss: {}'
                  .format(i +1, time.time()- s_t, loss_act.item() ))
    
    logger.info("Average loss: " + str(np.mean(np.array(losses))) )
    logger.info('Losses : ' + str( losses ))
    return losses


def ptb_evaluate(data_source, model, args, criterion, batch_size):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(batch_size)
    denom = len(data_source) - 1
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):

            if args.dataset in [ 'PTB-Char', 'PTB-Word' ]:
                data, targets = ptb_char_get_batch(data_source, i, args, evaluation=True)
                denom = len(data_source)
            else:
                data, targets = get_batch(data_source, i, args)
            
            targets = targets.contiguous().view(-1)
            if args.dataset == 'PTB-Word':
                log_prob, hidden = model(data, hidden)
                loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data
                total_loss += loss * len(data)
            else:
                output, hidden = model(data, hidden)

                if args.dataset == 'PTB-Char':
                    total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
                else:
                    total_loss += len(data) * criterion(output, targets).item()

            hidden = repackage_hidden(hidden)
    return total_loss / denom #(len(data_source) - 1)


def ptb_train( epoch, args, train_data, model, named_params, logger, criterion, params ):
    alpha = args.alpha
    beta = args.beta

    if args.dataset == 'PTB-Word':
        assert( args.batch_size == args.small_batch_size )

    model.train()
    loss = torch.zeros( [], device=args.device )
    total_loss = 0.
    total_mini_loss = 0.
    total_regularization_loss = 0.
    start_time = time.time()

    #if args.dataset == 'PTB-Word':
    #    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    #else:
    hidden = model.init_hidden(args.batch_size)

    batch, i = 0, 0
    #for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    while i < train_data.size(0) - 1 - 1:
        if args.dataset in ['PTB-Char', 'PTB-Word']:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            if args.dataset == 'PTB-Word':
                seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            model.train()
            data, targets = ptb_char_get_batch(train_data, i, args, seq_len=seq_len)
        else:
            seq_len = bptt = args.bptt
            data, targets = get_batch(train_data, i, args)

        xdata, xtargets = data.clone(), targets.clone()
        
        T = xdata.size()[0]
        PARTS = args.parts
        STEP = T // PARTS

        if (PARTS * STEP < T):
            PARTS += 1
        start, end, s_id = 0, STEP, 0

        if args.dataset in ['PTB-Char', 'PTB-Word']:
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / ( args.bptt * 2 * PARTS)
        
        xhidden = repackage_hidden(hidden)
        while (start < end) and (end < T):
            cur_data, cur_targets = xdata[start:end, :], xtargets[start:end, :].contiguous().view(-1)
            #if cur_targets.size(0) == 0:
            #    print('Error.. 0 length input')
            #    exit(1)

            old_xhidden = xhidden
            if args.debias:
                pre_pre_optimizer_updates( named_params, args )
                optimizer.zero_grad()
                xhidden = repackage_hidden(xhidden)
                output, xhidden = model(cur_data, xhidden)
                loss = criterion(output, cur_targets)
                loss.backward()
                pre_optimizer_updates( named_params, args )
            xhidden = old_xhidden

            for k in range(args.K):
                #xhidden = old_xhidden
                optimizer.zero_grad()
            
                xhidden = repackage_hidden(xhidden)
                if args.dataset in ['PTB-Char', 'PTB-Word']:
                    if args.dataset == 'PTB-Char':
                        output, xhidden, rnn_hs, dropped_rnn_hs = model(cur_data, xhidden, return_h=True)
                        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, cur_targets)
                    elif args.dataset == 'PTB-Word':
                        log_prob, xhidden, rnn_hs, dropped_rnn_hs = model(cur_data, xhidden, return_h=True)
                        raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)  #criterion(model.decoder.weight, model.decoder.bias, output, cur_targets)

                    loss = raw_loss
                    # Activiation Regularization
                    if args.ptb_alpha: loss = loss + sum(args.ptb_alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                    # Temporal Activation Regularization (slowness)
                    if args.ptb_beta: loss = loss + sum(args.ptb_beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

                else:
                    output, xhidden = model(cur_data, xhidden)
                    loss = criterion(output, cur_targets)
            
                total_mini_loss += loss.item()
                _lambda = 2.0
                #if args.dataset == 'PTB-Char': _lambda = 10.0
                regularizer = get_regularizer_named_params( named_params, args, _lambda=_lambda )       
                loss += regularizer
                total_regularization_loss += regularizer.item()
                #print('-- total_mini_loss = ', total_mini_loss, ' -- loss=', loss.item())
            
                loss.backward()
                if args.dataset in ['PTB-Char', 'PTB-Word'] or args.debias:
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    torch.nn.utils.clip_grad_norm_(params, args.clip)
                optimizer.step() 
            
            post_optimizer_updates( named_params, args, epoch ) 

            s_id += 1
            start = end
            end = start + STEP #start + args['small_batch_size']
            if end >= T: end = T-1

        targets = targets.contiguous().view(-1)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        if args.dataset == 'PTB-Char':
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
            loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
            optimizer.param_groups[0]['lr'] = lr2
        elif args.dataset == 'PTB-Word':
            log_prob, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)  #criterion(model.decoder.weight, model.decoder.bias, output, targets)
            optimizer.param_groups[0]['lr'] = lr2
        else:
            output, hidden = model(data, hidden)
            loss = criterion(output, targets)

        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f} | L(w) {:5.2f} | R(w) {:5.2f} '.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2),
                total_mini_loss, total_regularization_loss))
            total_loss = 0
            total_mini_loss = 0.
            total_regularization_loss = 0.
            start_time = time.time()

        batch += 1
        i += seq_len

def test(model, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]

        hidden = model.init_hidden(data.size(0))
        
        outputs, hidden, recon_loss = model(data, hidden)        
        output = outputs[-1]
        test_loss += F.nll_loss(output, target, size_average=False).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
    sys.stdout.flush()
    return test_loss, 100. * correct / len(test_loader.dataset)


def train(epoch, args, train_loader, permute, n_classes, model, named_params, logger):
    global steps
    global estimate_class_distribution

    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta

    PARTS = args.parts
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_oracle_loss = 0
    model.train()
    
    T = seq_length
    #entropy = EntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]
        
        B = target.size()[0]
        step = model.network[0].step
        xdata = data.clone()
        pdata = data.clone()
        
        inputs = xdata.permute(2, 0, 1) 
        T = inputs.size()[0]
 
        Delta = torch.zeros(B, dtype=xdata.dtype, device=xdata.device)
        
        _PARTS = PARTS
        if (PARTS * step < T):
            _PARTS += 1
        for p in range(_PARTS):
            x, start, end = get_xt(p, step, T, inputs)
            
            if p==0:
                h = model.init_hidden(xdata.size(0))
            else:
                #_, h = model.network[0].rnn( inputs[:end], h )
                h = (h[0].detach(), h[1].detach())
            
            if p<PARTS-1:
                if epoch < 20:
                    if args.per_ex_stats:
                        oracle_prob = estimatedDistribution[batch_idx*batch_size:(batch_idx+1)*batch_size, p]
                    else:
                        oracle_prob = 0*estimate_class_distribution[target, p] + (1.0/n_classes)
                else:
                    oracle_prob = estimate_class_distribution[target, p]
            else:
                oracle_prob = F.one_hot(target).float() 
            
            o, h = model.network[0].rnn( x, h )
            out = F.dropout(model.linear2(model.linear1( (h[0]) )), model.dropout)
            out = out.squeeze(dim=0)
            prob_out = F.softmax(out, dim=1)
            output = F.log_softmax(out, dim=1) 

            if p<PARTS-1:
                with torch.no_grad():
                    filled_class = [0]*n_classes
                    n_filled = 0
                    for j in range(B):
                        if n_filled==n_classes: break

                        y = target[j].item()
                        if filled_class[y] == 0 and (torch.argmax(prob_out[j]) != target[j]):
                            filled_class[y] = 1
                            estimate_class_distribution[y, p] = prob_out[j].detach()
                            n_filled += 1

            optimizer.zero_grad()
            
            clf_loss = (p+1)/(_PARTS)*F.nll_loss(output, target)
            oracle_loss = (1 - (p+1)/(_PARTS)) * 1.0 *torch.mean( -oracle_prob * output )
                
            regularizer = get_regularizer_named_params( named_params, args, _lambda=1.0 )       
            loss = clf_loss + oracle_loss + regularizer 
            #loss.backward(retain_graph=True)
            
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                
            '''
            for name in named_params:
                #print('name : ', name)
                param, sm, lm = named_params[name]
                sm.data.copy_(param.data)
                if param.grad is not None:
                    #print('copy grad = ', name)
                    lm.data.copy_(param.grad.data)'''
                
            optimizer.step()
            post_optimizer_updates( named_params, args )
        
            train_loss += loss.item()
            total_clf_loss += clf_loss.item()
            total_regularizaton_loss += regularizer #.item()
            total_oracle_loss += oracle_loss.item()
        
        '''
        hidden = model.init_hidden(data.size(0))

        optimizer.zero_grad()
        outputs, hidden, recon_loss = model(data, hidden, PARTS)
        clf_loss = F.nll_loss(outputs[-1], target)
        
        recon_loss = args.recon * recon_loss
        loss = clf_loss + recon_loss
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        '''
        
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\tLoss: {:.6f}\tOracle: {:.6f}\tClf: {:.6f}\tReg: {:.6f}\tSteps: {}'.format(
                   epoch, batch_idx * batch_size, len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), lr, train_loss / args.log_interval, 
                   total_oracle_loss / args.log_interval, 
                   total_clf_loss / args.log_interval, total_regularizaton_loss / args.log_interval, steps))
            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_oracle_loss = 0

            sys.stdout.flush()





parser = argparse.ArgumentParser(description='Sequential Decision Making..')

parser.add_argument('--alpha', type=float, default=0.1, help='Alpha')
parser.add_argument('--beta', type=float, default=0.5, help='Beta')
parser.add_argument('--rho', type=float, default=0.0, help='Rho')
parser.add_argument('--debias', action='store_true', help='FedDyn debias algorithm')
parser.add_argument('--K', type=int, default=1, help='Number of iterations for debias algorithm')

parser.add_argument('--ptb_alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--ptb_beta', type=float, default=0,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')

parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1, #2,
                    help='number of layers')
parser.add_argument('--bptt', type=int, default=300, #35,
                    help='sequence length')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')

parser.add_argument('--n_experts', type=int, default=15,
                    help='PTB-Word n_experts')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=620,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--clip', type=float, default=1.0, #0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 200)')
parser.add_argument('--parts', type=int, default=10,
                    help='Parts to split the sequential input into (default: 10)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--small_batch_size', type=int, default=-1, metavar='N',
                    help='batch size')
parser.add_argument('--max_seq_len_delta', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='batch size')

parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='output locked dropout (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='input locked dropout (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=0.29,
                    help='input locked dropout (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.1,
                    help='dropout applied to weights (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.2,
                    help='dropout applied to hidden layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wnorm', action='store_false',
                    help='use weight normalization (default: True)')
parser.add_argument('--temporalwdrop', action='store_false',
                    help='only drop the temporal weights (default: True)')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use')
parser.add_argument('--when', nargs='+', type=int, default=[50, 75, 90],
                    help='When to decay the learning rate')
parser.add_argument('--load', type=str, default='',
                    help='path to load the model')
parser.add_argument('--save', type=str, default='./models/',
                    help='path to load the model')

parser.add_argument('--per_ex_stats', action='store_true',
                    help='Use per example stats to compute the KL loss (default: False)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted dataset (default: False)')
parser.add_argument('--dataset', type=str, default='CIFAR-10',
                    help='dataset to use')
parser.add_argument('--dataroot', type=str, 
                    default='/home/anilkag/code/compact-vision-nets/PDE-Feature-Generator/data/',
                    help='root location of the dataset')
args = parser.parse_args()

if args.dataset == 'PTB-Word':
    if args.nhidlast < 0:
        args.nhidlast = args.emsize
    if args.dropoutl < 0:
        args.dropoutl = args.dropouth
    if args.small_batch_size < 0:
        args.small_batch_size = args.batch_size

args.cuda = True

exp_name = args.dataset + '-nhid-' + str(args.nhid) + '-parts-' + str(args.parts) + '-optim-' + args.optim
exp_name += '-B-' + str(args.batch_size) + '-E-' + str(args.epochs) + '-K-' + str(args.K)
if args.permute:
    exp_name += '-perm-' + str(args.permute)
if args.per_ex_stats:
    exp_name += '-per-ex-stats-'
if args.debias:
    exp_name += '-debias-'

prefix = args.save + exp_name

logger = logging.getLogger('trainer')

file_log_handler = logging.FileHandler( './logs/logfile-' + exp_name + '.log')
logger.addHandler(file_log_handler)

stderr_log_handler = logging.StreamHandler()
logger.addHandler(stderr_log_handler)

# nice output format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(formatter)
stderr_log_handler.setFormatter(formatter)

logger.setLevel( 'DEBUG' )

logger.info('Args: {}'.format(args))
logger.info('Exp_name = ' + exp_name)
logger.info('Prefix = ' + prefix)




torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed(args.seed)



steps = 0
if args.dataset in ['CIFAR-10', 'MNIST-10']:
    train_loader, test_loader, seq_length, input_channels, n_classes = data_generator(args.dataset, 
                                                                     batch_size=args.batch_size,
                                                                     dataroot=args.dataroot, 
                                                                     shuffle=(not args.per_ex_stats))
    permute = torch.Tensor(np.random.permutation(seq_length).astype(np.float64)).long()   # Use only if args.permute is True

    estimate_class_distribution = torch.zeros(n_classes, args.parts, n_classes, dtype=torch.float)
    estimatedDistribution = None
    if args.per_ex_stats:
        estimatedDistribution = torch.zeros(len(train_loader)*args.batch_size, args.parts, n_classes, dtype=torch.float)

elif args.dataset == 'PTB-300':
    corpus, train_data, val_data, test_data, ntokens = get_train_val_test_data(args, device)
elif args.dataset == 'PTB-Char':
    corpus, train_data, val_data, test_data, ntokens = ptb_char_get_train_val_test_data(args, device, logger)
elif args.dataset == 'PTB-Word':
    corpus, train_data, val_data, test_data, ntokens = ptb_word_get_train_val_test_data(args, device, logger)
elif args.dataset == 'Add-Task':
    logger.info('No explicit loader..')
    input_size = 2
else:
    logger.info('Unknown dataset.. customize the routines to include the train/test loop.')
    exit(1)

optimizer = None
if len(args.load) > 0:
    logger.info("Loaded model\n")
    model = torch.load(args.load)
elif args.dataset == 'PTB-300':
    model = PTBModel_300( args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied )
    criterion = nn.NLLLoss()
    params = list(model.parameters())
    total_params = ptb_count_parameters(model)
elif args.dataset == 'PTB-Word':
    model = PTBWordModel(args.model, ntokens, args.emsize, args.nhid, args.nhidlast, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, 
             args.tied, args.dropoutl, args.n_experts)
    params = list(model.parameters())
    total_params = sum(x.data.nelement() for x in model.parameters())
    criterion = nn.CrossEntropyLoss()
elif args.dataset == 'PTB-Char':
    model = PTBCharModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)

    criterion = None
    if args.resume:
        logger.info('Resuming model ...')
        model, criterion, optimizer = model_load(args.resume)
        optimizer.param_groups[0]['lr'] = args.lr
        model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
        if args.wdrop:
            for rnn in model.rnns:
                if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
                elif rnn.zoneout > 0: rnn.zoneout = args.wdrop

    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]

        logger.info('Using' + str(splits))
        criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
    if args.cuda:
        criterion = criterion.cuda()
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())

elif args.dataset == 'Add-Task':
    rnn = nn.LSTM( input_size, args.nhid )
    model = AddTaskModel( rnn, args.nhid )
    total_params = ptb_count_parameters(model)
else:
    model = SeqModel(ninp=input_channels,
                     nhid=args.nhid,
                     nout=n_classes,
                     dropout=args.dropout,
                     dropouti=args.dropouti,
                     dropouth=args.dropouth,
                     wdrop=args.wdrop,
                     temporalwdrop=args.temporalwdrop,
                     wnorm=args.wnorm,
                     n_timesteps=seq_length, 
                     parts=args.parts)

    total_params = count_parameters(model)
    if args.cuda:
        permute = permute.cuda()

if args.cuda:
    model.cuda()

lr = args.lr
if optimizer is None:
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.wdecay)
    if args.optim == 'SGD':
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.wdecay)

    if args.dataset in ['PTB-Char', 'PTB-Word']:
        optimizer = getattr(optim, args.optim)(params, lr=lr, weight_decay=args.wdecay)

logger.info('Optimizer = ' + str(optimizer) )
logger.info('Model total parameters: {}'.format(total_params))

all_test_losses = []
epochs = args.epochs #100
best_acc1 = 0.0
best_val_loss = None
first_update = False
named_params = get_stats_named_params( model )
for epoch in range(1, epochs + 1):
    start = time.time()
    
    if args.dataset in ['CIFAR-10', 'MNIST-10']:
        if args.per_ex_stats and epoch%5 == 1 :
            first_update = update_prob_estimates( model, args, train_loader, permute, estimatedDistribution, estimate_class_distribution, first_update )

        train(epoch, args, train_loader, permute, n_classes, model, named_params, logger)   
        #train_oracle(epoch)

        reset_named_params(named_params, args)

        test_loss, acc1 = test( model, test_loader, logger )
    
        logger.info('time taken = ' + str(time.time() - start) )
        if epoch in args.when:
            # Scheduled learning rate decay
            lr /= 10.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
            
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                #'oracle_state_dict': oracle.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                #'oracle_optimizer' : oracle_optim.state_dict(),
            }, is_best, prefix=prefix)
 
        all_test_losses.append(test_loss)

    elif args.dataset == 'Add-Task':
       all_test_losses = add_task_train_online( model, optimizer, args, named_params )
       break
    elif args.dataset in ['PTB-300', 'PTB-Char', 'PTB-Word']:
        save_filename = args.save + exp_name + 'model.pt'

        ptb_train(epoch, args, train_data, model, named_params, logger, criterion, params)   
        reset_named_params(named_params, args)

        if args.dataset == 'PTB-Word':
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    if 'ax' in optimizer.state[prm]:
                        prm.data = optimizer.state[prm]['ax'].clone()

        val_loss = ptb_evaluate(val_data, model, args, criterion, args.eval_batch_size)
        all_test_losses.append(val_loss)

        if args.dataset == 'PTB-Word':
            if 't0' in optimizer.param_groups[0]:
                for prm in model.parameters():
                    if prm in tmp.keys():
                        prm.data = tmp[prm].clone()

        logger.info('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(epoch, (time.time() - start),
                                           val_loss, math.exp(val_loss), val_loss / math.log(2)))
        logger.info('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            if args.dataset in ['PTB-Char', 'PTB-Word']:
                model_save(save_filename, model, criterion, optimizer)
            else:
                with open(save_filename, 'wb') as f:
                    torch.save(model, f)
            best_val_loss = val_loss
        elif args.dataset == 'PTB-300' and (not args.debias):
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            #lr /= 10.0

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if args.dataset == 'PTB-300' and args.debias:
            if epoch in args.when:
                lr /= 4.0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        if args.dataset == 'PTB-Char':
            if epoch in args.when:
                logger.info('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

        if args.dataset == 'PTB-Word':
            if args.optim == 'SGD' and 't0' not in optimizer.param_groups[0] and (len(all_test_losses)>args.nonmono and val_loss > min(all_test_losses[:-args.nonmono])):
                logger.info('Switching!')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

        if epoch == epochs:
            if args.dataset in ['PTB-Char', 'PTB-Word']:
                model, criterion, optimizer = model_load(save_filename)
            else:
                # Load the best saved model.
                with open(save_filename, 'rb') as f:
                    model = torch.load(f)

            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            # Currently, only rnn model supports flatten_parameters function.
            if args.dataset == 'PTB-300':
                if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
                    model.rnn.flatten_parameters()

            # Run on test data.
            bsize = args.eval_batch_size
            if args.dataset in ['PTB-Char', 'PTB-Word']: bsize = 1
            test_loss = ptb_evaluate(test_data, model, args, criterion, bsize)
            logger.info('=' * 89)
            logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(test_loss, math.exp(test_loss), test_loss / math.log(2)))
            logger.info('=' * 89)

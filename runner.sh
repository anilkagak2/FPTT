
CUDA='0'

# MNIST-Permute, MNIST, CIFAR-10
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset MNIST-10 --permute
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset MNIST-10
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset CIFAR-10

# PTB-300 word level language modelling
# 103.29 (fixed the learning rate bug) 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset PTB-300 --alpha 0.5 --beta 0.5 --dataroot '/home/anilkag/code/rnn_results/aditya/PTB/data/' --clip 0.25 --lr 2.0 --optim 'SGD' --nhid 256 --log-interval 10 --epochs 100 --dropout 0.2 --parts 6 --K 1
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset PTB-300 --alpha 0.5 --beta 0.5 --dataroot '/home/anilkag/code/rnn_results/aditya/PTB/data/' --clip 0.25 --lr 2.0 --optim 'SGD' --nhid 128 --log-interval 10 --epochs 100 --dropout 0.2 --parts 6 --K 1

# Add-Task dataset
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset Add-Task --alpha 0.5 --beta 0.5 --clip 1.0 --lr 0.001 --optim 'Adam' --nhid 128 --log-interval 200 --epochs 10000 --parts 10 --bptt 1000
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset Add-Task --alpha 0.5 --beta 0.5 --clip 1.0 --lr 0.001 --optim 'Adam' --nhid 128 --log-interval 200 --epochs 10000 --parts 10 --bptt 750
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset Add-Task --alpha 0.5 --beta 0.5 --clip 1.0 --lr 0.001 --optim 'Adam' --nhid 128 --log-interval 200 --epochs 10000 --parts 10 --bptt 500
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset Add-Task --alpha 0.5 --beta 0.5 --clip 1.0 --lr 0.001 --optim 'Adam' --nhid 128 --log-interval 200 --epochs 10000 --parts 10 --bptt 200

# PTB Char level language modelling (bptt 150 sequence length)
dataroot='/home/anilkag/code/rnn_results/aditya/PTB-c/' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset PTB-Char --alpha 0.2 --beta 0.5 --dataroot $dataroot --emsize 200 --nhid 1000  --nlayers 3 --lr 2e-3 --optim 'Adam' --clip 0.25 --epochs 500 --batch_size 128 --bptt 150 --dropout 0.1 --dropouth 0.25 --dropouti 0.1 --dropoute 0 --wdrop 0.2 --nonmono 5 --log-interval 200  --wdecay 1.2e-6  --when 300 400  --tied --parts 3  

# PTB Word level language modelling (bptt 70 sequence length). Need to run the dynamic equal script after training this model.
dataroot='/home/anilkag/code/rnn_results/aditya/PTB/data/' 
CUDA_VISIBLE_DEVICES=$CUDA python train.py --dataset PTB-Word --alpha 0.2 --beta 0.7 --dataroot $dataroot --emsize 280 --nhid 960 --nhidlast 960   --nlayers 3 --lr 20 --optim 'SGD' --clip 0.2 --epochs 1000 --batch_size 6 --bptt 70 --dropout 0.4 --dropouth 0.225 --dropouti 0.4 --dropoute 0.1 --dropoutl 0.29   --wdrop 0.2 --seed 28 --nonmono 5 --log-interval 200  --wdecay 1.2e-6 --ptb_alpha 2.0 --ptb_beta 1.0 --n_experts 15 --small_batch_size -1 --max_seq_len_delta 40  --when  -1  --tied --parts 3 


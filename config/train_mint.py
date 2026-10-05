
import time

device = 'cuda'

out_dir = 'output/mint'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 25
log_interval = 25 # don't print too too often
seed = 42

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'mint'
wandb_run_name = 'run' + str(time.time())

batch_size = 128
block_size = 64
data_fraction = 1.0
select = "left"

n_layer = 12
n_head = 12
n_embd = 120
dropout = 0.1
weight_decay = 2e-1
vocab_size = 1476 + 2 # len(vocab.csv) + 2 [padding and no event?]

learning_rate = 6e-4 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 1000 # not super necessary potentially
ignore_tokens = [0] # padding
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5

# we also disabled lifestyle_augmentations
# TODO, ablations with rounding = 60

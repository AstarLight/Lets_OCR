import alphabets

raw_folder = ''
train_data = './data/train_lmdb'
test_data = './data/test_lmdb'
random_sample = True
random_seed = 1111
using_cuda = True
keep_ratio = False
gpu_id = '5'
model_dir = './w160_bs64_model'
data_worker = 5
batch_size = 64
img_height = 32
img_width = 160
alphabet = alphabets.alphabet
epoch = 20
display_interval = 20
save_interval = 4000
test_interval = 2000
test_disp = 20
test_batch_num = 32
lr = 0.0001
beta1 = 0.5
infer_img_w = 160

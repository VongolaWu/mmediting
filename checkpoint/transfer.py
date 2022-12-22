import torch
from tqdm import tqdm

ckpt = torch.load(
    'checkpoint/model_best.pth.tar', map_location=torch.device('cpu'))
print(ckpt.keys())
params = ckpt['state_dict']
# params = ckpt
keys = params.keys()
print(list(keys)[:20])
for key in tqdm(list(keys)):
    new_key = 'generator.' + key[13:]
    # new_ema_key = 'generator_ema.'+key
    params[new_key] = params.pop(key)
    # params[new_ema_key] = params[new_key]

path = 'checkpoint/mmp-rnn.pth'
torch.save(params, path)
print(list(params.keys())[:20])

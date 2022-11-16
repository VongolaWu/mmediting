import torch
from tqdm import tqdm

ckpt = torch.load(
    '../AirNet/ckpt/Derain.pth', map_location=torch.device('cpu'))
print(ckpt.keys())
# params = ckpt['params']
params = ckpt
keys = params.keys()
print(list(keys)[:20])
for key in tqdm(list(keys)):
    new_key = 'generator.' + key
    # new_ema_key = 'generator_ema.'+key
    params[new_key] = params.pop(key)
    # params[new_ema_key] = params[new_key]

path = 'checkpoint/AirNet_derain.pth'
torch.save(params, path)
print(list(params.keys())[:20])

import torch

model = torch.load('./exps/adet_checkpoint0011.pth')
model_new = {}
for key, value in model['model'].items():
    if 'class_embed' not in key:
        model_new[key] = value
model_new = {'model': model_new}
torch.save(model_new, './exps/adet_checkpoint0011_pop.pth')

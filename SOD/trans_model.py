import importlib
from skimage import io
import os
import torch
import skimage
from skimage.transform import resize
import numpy as np

from model.utils.simplesum_octconv import simplesum
from train import set_model

path = "csnet_model.pt"
model,_ = set_model()
model.cuda(0)

this_checkpoint = "checkpoint_epoch20.pth.tar"
if os.path.isfile(this_checkpoint):
    print("=> loading checkpoint '{}'".format(this_checkpoint))
    checkpoint = torch.load(this_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(
        this_checkpoint, checkpoint['epoch']))

    example = skimage.img_as_float(io.imread("example.jpg"))
    img = resize(example, (224, 224),
                 mode='reflect',
                 anti_aliasing=False)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    example = torch.Tensor(img).unsqueeze_(dim=0).cuda()
    model = model.eval()
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)

else:
    print(this_checkpoint, "Not found.")

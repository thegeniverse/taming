from typing import *

import torch
import PIL
import PIL.Image
import numpy as np
import torchvision

from taming.modeling_utils import load_model


def generation(img: PIL.Image, ):
    # model_name = "sber"
    model_name = "coco"
    device = "cuda"

    vqgan_model = load_model(model_name=model_name, ).to(device)

    img = img.convert('RGB')
    img_tensor = torch.tensor(np.asarray(img))
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = (img_tensor / 255.) * 2 - 1

    if len(img_tensor.shape) <= 3:
        img_tensor = img_tensor[None, ::]

    img_tensor = img_tensor.float().to(device)

    img_tensor = torch.nn.functional.interpolate(
        img_tensor,
        (int((img_tensor.shape[2] / 16) * 16),
         int((img_tensor.shape[3] / 16) * 16)),
        mode="bilinear",
    )

    z, _, [_, _, _indices] = vqgan_model.encode(img_tensor)

    z = z.to(device)

    z = vqgan_model.post_quant_conv(z)
    # z = torch.cat(
    #     [z, z],
    #     axis=1,
    # )
    img_rec = vqgan_model.decoder(z)
    img_rec = (img_rec.clip(-1, 1) + 1) / 2
    torchvision.transforms.ToPILImage()(img_rec[0]).save(f"{model_name}.png")


def test():
    try:
        model_name = "sber"
        model = load_model(model_name=model_name, )
        print("OK 👌")

    except Exception as e:
        print("ERROR! 😥")
        print(e)


if __name__ == "__main__":
    # test()

    img = PIL.Image.open("./img.jpg")
    img = img.resize((200, 125))
    generation(img)

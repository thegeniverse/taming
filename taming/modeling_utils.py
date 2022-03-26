import os
import logging

import torch
import omegaconf
import yaml
import requests

from taming.models.vqgan import VQModel, GumbelVQ
from taming.models.cond_transformer import Net2NetTransformer

logging.basicConfig(format='%(message)s', level=logging.INFO)

VQGAN_CKPT_DICT = {
    "imagenet-16384":
    r"https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
    "openimages-8192":
    r"https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
    "sber": r"https://www.dropbox.com/s/9tzfjsuf78xg4g9/sber.ckpt?dl=1",
    "wikiart_16384":
    r"http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt",
    "coco": r'https://dl.nmkd.de/ai/clip/coco/coco.ckpt',
    "faceshq":
    r'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt',
    "wikiart_1024":
    r'https://github.com/Eleiber/VQGAN-Mirrors/releases/download/0.0.1/wikiart_1024.ckpt',
    "wikiart_7mil":
    r'http://batbot.tv/ai/models/VQGAN/WikiArt_augmented_Steps_7mil_finetuned_1mil.ckpt',
    "sflckr":
    r'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1',
    "lstage": r'http://batbot.tv/ai/models/VQGAN/coco_first_stage.ckpt',
}
VQGAN_CONFIG_DICT = {
    "imagenet-16384":
    r"https://raw.githubusercontent.com/thegeniverse/taming/master/configs/imagenet-16384.yaml",
    "openimages-8192":
    r"https://raw.githubusercontent.com/thegeniverse/taming/master/configs/openimages-8192.yaml",
    "sber":
    r"https://raw.githubusercontent.com/thegeniverse/taming/master/configs/sber.yaml",
    "wikiart_16384":
    r"http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml",
    "coco": r'https://dl.nmkd.de/ai/clip/coco/coco.yaml',
    "faceshq":
    r'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT',
    "wikiart_1024":
    r'https://github.com/Eleiber/VQGAN-Mirrors/releases/download/0.0.1/wikiart_1024.yaml',
    "wikiart_7mil":
    r'http://batbot.tv/ai/models/VQGAN/WikiArt_augmented_Steps_7mil_finetuned_1mil.yaml',
    "sflckr":
    r'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1',
    "lstage": r'http://batbot.tv/ai/models/VQGAN/coco_first_stage.yaml',
}


def load_vqgan(
    config: omegaconf.dictconfig.DictConfig,
    ckpt_path: str = None,
) -> VQModel:
    """
    Load a VQGAN model from a config file and a ckpt path 
    where a VQGAN model is saved.

    Args:
        config ([type]): VQGAN model config.
        ckpt_path ([type], optional): path of a saved model. 
            Defaults to None.

    Returns:
        VQModel: 
            loaded VQGAN model.
    """
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(ckpt_path)

    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(ckpt_path)
        model = parent_model.first_stage_model

    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = GumbelVQ(**config.model.params)
        print(config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(ckpt_path)

    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss

    return model

    # if "GumbelVQ" in config.model.target:
    #     model = GumbelVQ(**config.model.params)

    # else:
    #     model = VQModel(**config.model.params)

    # if ckpt_path is not None:
    #     state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    #     _missing, _unexpected = model.load_state_dict(state_dict, strict=False)

    # return model.eval()


def load_config(
    config_path: str,
    display=False,
) -> omegaconf.dictconfig.DictConfig:
    """
    Loads a VQGAN configuration file from a path or URL.

    Args:
        config_path (str): local path or URL of the config file.
        display (bool, optional): if `True` the configuration is 
            printed. Defaults to False.

    Returns:
        omegaconf.dictconfig.DictConfig: configuration dictionary.
    """
    config = omegaconf.OmegaConf.load(config_path)

    if display:
        logging.info(yaml.dump(omegaconf.OmegaConf.to_container(config)))

    return config


def download_model(
    model_name: str = "imagenet-16384",
    force_download: bool = False,
):
    modeling_dir = os.path.dirname(os.path.abspath(__file__))
    modeling_cache_dir = os.path.join(modeling_dir, ".modeling_cache")
    os.makedirs(modeling_cache_dir, exist_ok=True)

    modeling_config_path = os.path.join(modeling_cache_dir,
                                        f'{model_name}.yaml')
    if not os.path.exists(modeling_config_path) or force_download:
        modeling_config_url = VQGAN_CONFIG_DICT[model_name]

        logging.info(
            f"Downloading `{model_name}.yaml` from {modeling_config_url}")
        response = requests.get(modeling_config_url, allow_redirects=True)

        assert response.ok, "Error downloading config!"

        with open(modeling_config_path, "wb") as yaml_file:
            yaml_file.write(response.content)

    modeling_ckpt_path = os.path.join(modeling_cache_dir, f'{model_name}.ckpt')
    if not os.path.exists(modeling_ckpt_path) or force_download:
        modeling_ckpt_url = VQGAN_CKPT_DICT[model_name]

        logging.info(
            f"Downloading pre-trained weights for VQ-GAN from {modeling_ckpt_url}"
        )
        session = requests.session()
        if "ffhq" in modeling_ckpt_url:
            session.headers.update({
                'referer':
                "https://app.koofr.net/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fcheckpoints",
            })

        response = session.get(
            modeling_ckpt_url,
            allow_redirects=True,
        )

        assert response.ok, "Error downloading pre-trained weights!"

        with open(modeling_ckpt_path, "wb") as ckpt_file:
            ckpt_file.write(response.content)

    return modeling_config_path, modeling_ckpt_path


def load_model(model_name: str = "imagenet-16384", ):
    logging.info(f"Loading {model_name}")

    modeling_config_path, modeling_ckpt_path = download_model(
        model_name,
        force_download=False,
    )

    vqgan_config = load_config(
        config_path=modeling_config_path,
        display=False,
    )
    vqgan_model = load_vqgan(
        vqgan_config,
        ckpt_path=modeling_ckpt_path,
    )

    return vqgan_model
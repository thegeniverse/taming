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
}
VQGAN_CONFIG_DICT = {
    "imagenet-16384":
    r"https://raw.githubusercontent.com/thegeniverse/taming/master/configs/imagenet-16384.yaml",
    "openimages-8192":
    r"https://raw.githubusercontent.com/thegeniverse/taming/master/configs/openimages-8192.yaml",
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
    if "GumbelVQ" in config.model.target:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        _missing, _unexpected = model.load_state_dict(state_dict, strict=False)

    return model.eval()


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
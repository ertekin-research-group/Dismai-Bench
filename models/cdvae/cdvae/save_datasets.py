from pathlib import Path

import hydra
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from cdvae.common.utils import PROJECT_ROOT


def run(cfg: DictConfig) -> None:
    """
    Saves datasets and scalers for future loading

    :param cfg: run configuration, defined by Hydra in /conf
    """

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    
    # Save datasets
    hydra.utils.log.info("Saving datasets")
    datamodule.setup()
    torch.save(datamodule.train_dataset, Path(cfg.data.root_path) / 'train.pt')
    torch.save(datamodule.val_datasets, Path(cfg.data.root_path) / 'val.pt')
    torch.save(datamodule.test_datasets, Path(cfg.data.root_path) / 'test.pt')
    torch.save(datamodule.lattice_scaler, Path(cfg.data.root_path) / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, Path(cfg.data.root_path) / 'prop_scaler.pt')


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="", config_name="config")
def main(cfg: DictConfig):
    lightning_data_module = hydra.utils.instantiate(cfg.lightning_data_module)
    lightning_module = hydra.utils.instantiate(cfg.lightning_module)

    # create a trainer from the hydra config file
    trainer = hydra.utils.instantiate(cfg.trainer)

    for task in cfg.tasks:
        trainer_method = getattr(trainer, task)
        trainer_method(lightning_module, lightning_data_module)


if __name__ == "__main__":
    main()

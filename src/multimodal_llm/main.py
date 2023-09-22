import hydra
from omegaconf import DictConfig
from src.multimodal_llm.data.scicap import SciCapDataModule
from src.multimodal_llm.model.model import GPT, LLaMAMLP
from src.multimodal_llm.model.config import Config

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    lightning_data_module = hydra.utils.instantiate(cfg.lightning_data_module)
    lightning_module = hydra.utils.instantiate(cfg.lightning_module)

    config = Config.from_name("tiny_LLaMA_1b")
    model = GPT(config=config)
    lightning_module.model = model

    # create a trainer from the hydra config file
    trainer = hydra.utils.instantiate(cfg.trainer)

    for task in cfg.tasks:
        trainer_method = getattr(trainer, task)
        trainer_method(lightning_module, lightning_data_module)


if __name__ == "__main__":
    main()

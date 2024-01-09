from fileinput import filename
from configuration import config
from train import GModelTrainer
import logging
def main():
    format = "%(asctime)s:     %(message)s"
    logging.basicConfig(filename="training.log",
                        filemode="w",
        format=format, level=logging.DEBUG,
                        datefmt="%H:%M:%S")
    
    if config["data_config"]["evaluate"]:
        raise NotImplementedError
    else:
        model = GModelTrainer(config, logging)
        model.train()
    logging.info("Finish")
if __name__ == "__main__":
    main()
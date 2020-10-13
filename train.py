from utils.config_reader import read_config
from trainer import STDNTrainer
import config

model, optimizer, dataloaders, loss, scheduler, flags = read_config(config.flags)
epoch = flags.param_config.epoch

trainer = STDNTrainer(flags, loss, model, optimizer, scheduler)
trainer.train(dataloaders, n_epochs=epoch)

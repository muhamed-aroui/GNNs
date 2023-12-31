import torch
import time
from data import get_data
from pprint import pformat
import torch.nn as nn
import torch.optim as optim
from utils import AverageMeter
import torch.nn.functional as F


class GModelTrainer:
    def __init__(self, config, logger):
        self.logger = logger
        self.gpu_count = torch.cuda.device_count()
        self.since = time.time()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"DEVICE : {self.device}")
        self.config = config
        self.epochs = self.config["training_config"]["epoch"]
        # self.writer = SummaryWriter()
        self.best_acc = dict(train=0.0, validation=0.0, test=0.0)
        self._init_dataloaders()
        self._init_loss()
        self._init_optim()
        self._init_models()

    def _init_dataloaders(self):
        self.dataloaders, self.classes, self.class_to_idx, node_features = get_data(self.config, self.logger)
        self.config["data_config"]["node_features"] = node_features
        self.config["data_config"]["num_classes"] = len(self.class_to_idx)
        self.num_batches = len(iter(self.dataloaders["train"]))
        self.logger.info(pformat(self.config))

    def _init_loss(self):
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1) # binary classification

    def _init_optim(self):
        optim_params = [{'params': self.model.parameters(),
                            'lr': self.config['optim_config']["lr"],
                            'momentum': 0.9,
                            'weight_decay': 0.0005}]
        if self.config['optim_config']["optim"] == "sgd":
            self.optimizer = optim.SGD(optim_params)
        elif self.config['optim_config']["optim"] == "adam":
            self.optimizer = optim.Adam(optim_params)
        elif self.config['optim_config']["optim"] == "adamw":
            self.optimizer = optim.AdamW(optim_params)

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.1)
    
    def _init_models(self):
        self.model = get_model(self.config, self.logger)
        if self.gpu_count > 1:
            self.model = nn.DataParallel(
                self.model).to(self.device)
            self.logger.info("Using "+ str(self.gpu_count) + " GPUs: ")
        else:
            self.model = self.model.to(self.device)
            if self.gpu_count ==1:
                self.logger.info("Using 1 GPU")
            else:
                self.logger.info("Using CPU")
        
        self.logger.info(pformat(self.model))

    def _log_tensorboard(self, epoch, loss, top1, phase):
        _phase = phase.upper()
        _epoch = f"EPOCH[{epoch+1}/{self.epochs}]"
        _loss = f"LOSS : {loss}"
        _top1 = f"TOP1 : {round(top1, 4)}"
        self.logger.info(f"{_phase:20s}{_epoch:20}{_loss:20s}{_top1:20s}")
    
    def _log_training(self, epoch, batch_idx, loss):
        _epoch = f"EPOCH[{epoch+1}/{self.epochs}]"
        _iter = f"ITER[{batch_idx+1}/{self.num_batches}]"
        _loss = f"TRAIN_LOSS : {loss}"
        self.logger.info(f"{_epoch:15s}{_iter:30s}{_loss:20s}")
    
    def _pass(self, data, phase="val"):
        out = self.model(data.x, data.edge_index, data.batch)
        loss_value = self.criterion(out, data.y)
        if phase =="train":
            loss_value.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return out, loss_value.detach().cpu().item()

    def train(self):
        # breakpoint()
        self.model.train()
        avg_loss = AverageMeter()
        avg_top1 = AverageMeter()
        avg_top5 = AverageMeter()
        early_stopping_counter = 0
        for epoch in range(self.config["training_config"]["epoch"]):
            for batch_idx, input in enumerate(self.dataloaders["train"]):
                # self.logger.info(pformat(input.x))
                input = input.to(self.device) 
                out, loss_value = self._pass(input, phase="train")
                avg_loss.update(loss_value, input.num_graphs)
                top1 = self.cls_accuracy(output=out.detach().cpu().data, target=input.y.detach().cpu().data)
                avg_top1.update(top1, input.num_graphs)
                if batch_idx % self.config["training_config"]["log_iter"] == 0:
                    self._log_training(epoch, batch_idx, loss_value)
                
            self.logger.info("")
            self._log_tensorboard(epoch=epoch, loss= avg_loss.value, top1 = avg_top1.value, phase="train")
            val_acc, test_acc = self.evaluate(epoch)
            self.logger.info("")
            if val_acc > self.best_acc['validation']:
                self.best_acc['validation'] = val_acc
                self.best_acc['test'] = test_acc
                self.best_acc['train'] = avg_top1.value
                self.best_acc['epoch'] = epoch
                self.save_model(epoch=epoch, val_acc=val_acc, test_acc=test_acc)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter > self.config["validation_config"]["early_stopping_epoch"]:
                self.logger.info(f"NO IMPROVEMENT IN THE LAST {early_stopping_counter} EPOCHS: EARLY STOPPING AT EPOCH {epoch}")
                break
            
        self.logger.info(f"AT EPOCH {self.best_acc['epoch']}/{self.epochs}: BEST VAL ACCURACY: {self.best_acc['validation']}      TEST ACC : {self.best_acc['test']}     TRAIN ACC : {self.best_acc['train']} ")
        time_taken = time.time() - self.since
        hours_taken = time_taken // 3600
        minutes_taken = (time_taken % 3600) // 60
        seconds_taken = round((time_taken % 3600) % 60, 2)
        self.logger.info(f"Training took {hours_taken}h : {minutes_taken}m : {seconds_taken}s")

    @torch.no_grad() 
    def evaluate(self, epoch):
        self.model.eval()
        acc = dict(test= 0.0 , validation=0.0)
        if epoch >= self.config["validation_config"]["test_accuracy_log_epoch"]:
            phase_list = ["validation", "test"]
        else:
            phase_list = ["validation"]
        for phase in phase_list:
            avg_loss = AverageMeter()
            avg_top1 = AverageMeter()
            avg_top5 = AverageMeter()
            for _, input in enumerate(self.dataloaders[phase]):
                input = input.to(self.device) 
                out, loss_value = self._pass(input)
                avg_loss.update(loss_value, input.num_graphs)
                top1 = self.cls_accuracy(output=out.data, target=input.y.data)
                avg_top1.update(top1, input.num_graphs)
            acc[phase] = avg_top1.value
            self._log_tensorboard(epoch=epoch, loss= avg_loss.value, top1 = avg_top1.value, phase=phase)
        self.model.train()
        return acc["validation"], acc["test"]

    def save_model(self, epoch, val_acc, test_acc):
        self.logger.info("Saving a new model")
        weights = self.model.state_dict() if self.gpu_count <=1 else self.model.module.state_dict()
        model_states = {'epoch': epoch,
                        'state_dict': weights,
                        'best_val_acc': val_acc,
                        'test_acc': test_acc,
                        'class_to_idx': self.class_to_idx}
        torch.save(model_states, "./saved_model.pth")

    @staticmethod
    def cls_accuracy(output, target):
        pred = output.round()
        correct = (pred == target).float().sum()
        accuracy = correct / target.size(0)
        return accuracy.item()
    
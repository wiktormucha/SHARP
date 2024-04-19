import numpy as np
import torch
from tqdm import tqdm
import wandb
from utils.testing import batch_epe_calculation
import torchmetrics as metrics
import torch.nn as nn
import torch
import os
import cv2 as cv2
from utils.testing import batch_epe_calculation
import random
import json
import torchvision.transforms.v2 as v2

DEBUG = True
# DEBUG = False


def save_best_model(model, run_name, new_value: float, best_value, save_on_type: str):
    """
    Saves best model
    Args:
        val_loss (float): Current validation loss
        epoch (int): Current epoch
    """

    # Check if folder named wandb.run.name exists, if not, create the folder
    if not os.path.exists(f'checkpoints/{run_name}'):
        os.makedirs(f'checkpoints/{run_name}')

    best_value_ret = best_value
    # When higher value:
    if save_on_type == 'greater_than':

        if new_value >= best_value:
            best_value_ret = new_value
            print("Saving best model..")
            torch.save(model.state_dict(),
                       f'checkpoints/{run_name}/checkpoint_best.pth')

    else:
        if new_value <= best_value:
            best_value_ret = new_value
            print("Saving best model..")
            torch.save(model.state_dict(),
                       f'checkpoints/{run_name}/checkpoint_best.pth')

    return best_value_ret


class TrainerAR:
    """
    Class for training the model
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, training_config, wandb_logger: wandb = None, scheduler: torch.optim = None) -> None:
        """
        Initialisation
        Args:
            model (torch.nn.Module): Input modle used for training
            criterion (torch.nn.Module): Loss function
            optimizer (torch.optim): Optimiser
            config (dict): Config dictionary (needed max epochs and device)
            scheduler (torch.optim, optional): Learning rate scheduler. Defaults to None.
            grad_clip (int, optional): Gradient clipping. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": [], "test": []}
        self.epochs = training_config.max_epochs
        self.device = training_config.device
        self.scheduler = scheduler
        self.early_stopping_epochs = training_config.early_stopping
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.best_val_loss = 100000
        self.best_acc_val = 0
        self.wandb_logger = wandb_logger
        self.acc_val = []
        self.acc_test = []
        self.emplying_objs = training_config.emplying_objs
        self.pth_to_save_chkpt = training_config.checkpoint_pth
        self.softmax = nn.Softmax(dim=1)

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader) -> torch.nn.Module:
        """
        Training loop
        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
        Returns:
            torch.nn.Module: Trained model
        """
        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}, Val Acc={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                    self.acc_val[-1],
                )
            )

            # reducing LR if no improvement in training
            if self.scheduler is not None:
                self.scheduler.step()

            self.__save_best_model(acc_val=np.round(
                self.acc_val[-1], 10), epoch=epoch, pth_to_save=self.pth_to_save_chkpt)

            self.__save_best_model_val(val_loss=np.round(
                self.loss["val"][-1], 10), epoch=epoch, pth_to_save=self.pth_to_save_chkpt)

            # early stopping if no progress
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(
                    np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]),
                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0

            self.wandb_logger.log(
                {"acc": self.acc_val[-1],
                 "train_loss": np.round(
                    self.loss["train"][-1], 10),
                    "val_loss": np.round(self.loss["val"][-1], 10),
                    "best_acc": self.best_acc_val,
                 })

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        return self.model

    def _epoch_train(self, dataloader: torch.utils.data.DataLoader):
        """
        Training step in epoch
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.train()
        running_loss = []

        for i, data in enumerate(tqdm(dataloader, 0)):

            inputs = data['positions'].to(
                self.device).type(torch.cuda.FloatTensor)

            labels = data['action_label'].to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()

            self.optimizer.step()

            running_loss.append(loss.item()*labels.shape[0])

        epoch_loss = sum(running_loss)/len(dataloader.dataset)
        self.loss["train"].append(epoch_loss)

    def test_model(self, test_dataloader: torch.utils.data.DataLoader):
        self.__evaluation(dataloader=test_dataloader,
                          loss_list=self.loss["test"], acc_list=self.acc_test)
        print('Acc: ', self.acc_test[-1], ' Loss: ', self.loss["test"][-1])

    def __evaluation(self, dataloader: torch.utils.data.DataLoader, loss_list, acc_list):

        self.model.eval()
        running_loss = []
        Accuracy = metrics.Accuracy(
            task="multiclass", num_classes=36).to(self.device)
        acc_lst = []

        with torch.no_grad():

            for i, data in enumerate(dataloader, 0):

                inputs = data['positions'].to(
                    self.device).type(torch.cuda.FloatTensor)

                labels = data['action_label'].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item()*labels.shape[0])
                prob = self.softmax(outputs)
                acc = Accuracy(prob, labels)

                acc_lst.append(acc*labels.shape[0])
            epoch_loss = sum(running_loss)/len(dataloader.dataset)
            loss_list.append(epoch_loss)
            accuracy = sum(acc_lst)/len(dataloader.dataset)
            acc_list.append(accuracy.cpu().numpy())

    def test_h2o(self, dataloader: torch.utils.data.DataLoader):

        self.model.eval()

        h2o_test_results = {
            "modality": "hand+obj",
        }
        with torch.no_grad():

            for i, data in enumerate(dataloader, 0):

                inputs = data['positions'].to(
                    self.device).type(torch.cuda.FloatTensor)

                outputs = self.model(inputs)
                prob = self.softmax(outputs)
                # In prob of shape 64, 36 pick the highest value in each row

                indxs = torch.argmax(prob, dim=1)

                for key, value in zip(data['action_id'], indxs):
                    h2o_test_results[int(key)] = int(value) + 1

            print(f'Len of dict: {len(h2o_test_results)}')
            with open("action_labels.json", "w") as fp:
                json.dump(h2o_test_results, fp)

            # if answear.zip exists, remove it
            if os.path.exists('answear.zip'):
                os.system('rm answear.zip')
            os.system('zip answear.zip action_labels.json')
            return h2o_test_results

    def _epoch_eval(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluation step in epoch
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.__evaluation(dataloader=dataloader,
                          loss_list=self.loss["val"], acc_list=self.acc_val)

    def __save_best_model(self, acc_val: float, epoch: int, pth_to_save: str):
        """
        Saves best model
        Args:
            val_loss (float): Current validation loss
            epoch (int): Current epoch
        """

        # Check if folder named wandb.run.name exists, if not, create the folder
        if not os.path.exists(f'checkpoints/{self.wandb_logger.name}'):
            os.makedirs(f'checkpoints/{self.wandb_logger.name}')

        if acc_val >= self.best_acc_val:
            self.best_acc_val = acc_val
            print("Saving best model..")
            torch.save(self.model.state_dict(),
                       f'checkpoints/{self.wandb_logger.name}/checkpoint_best.pth')

    def __save_best_model_val(self, val_loss: float, epoch: int, pth_to_save: str):
        """
        Saves best model
        Args:
            val_loss (float): Current validation loss
            epoch (int): Current epoch
        """

        # Check if folder named wandb.run.name exists, if not, create the folder
        if not os.path.exists(f'checkpoints/{self.wandb_logger.name}'):
            os.makedirs(f'checkpoints/{self.wandb_logger.name}')

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print("Saving best model..")
            torch.save(self.model.state_dict(),
                       f'checkpoints/{self.wandb_logger.name}/checkpoint_best_val_loss.pth')


class Trainer3D:
    """
    Class for training the model
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, config: dict, wandb_logger: wandb, scheduler: torch.optim = None) -> None:
        """
        Initialisation

        Args:
            model (torch.nn.Module): Input modle used for training
            criterion (torch.nn.Module): Loss function
            optimizer (torch.optim): Optimiser
            config (dict): Config dictionary (needed max epochs and device)
            scheduler (torch.optim, optional): Learning rate scheduler. Defaults to None.
            grad_clip (int, optional): Gradient clipping. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": []}
        self.loss_depth = {"train": [], "val": []}
        self.epe = {"train": [], "val": []}
        self.acc = {"train": [], "val": []}
        self.epe_3d_l = {"train": [], "val": []}
        self.epe_3d_r = {"train": [], "val": []}
        self.epe_3d = {"train": [], "val": []}
        self.epochs = config.max_epochs
        self.device = config.device
        self.scheduler = scheduler
        self.early_stopping_epochs = config.early_stopping
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.best_val_loss = 100000
        self.wandb_logger = wandb_logger
        self.best_epe = 10000000
        self.best_epe3d = 10000000
        self.loss_hands = {"train": [], "val": []}
        self.loss_flags = {"train": [], "val": []}

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        if wandb_logger:
            self.run_name = wandb_logger.name
        else:
            self.run_name = f'debug_{random.randint(0,100000)}'

    def test(self, dataloader):
        self.model.eval()
        self._epoch_eval(dataloader, test=True)

        print(f'Test loss: {self.loss["val"][-1]}')
        print(f'Test EPE: {self.epe["val"][-1]}')
        print(f'Test EPE 3D: {self.epe_3d["val"][-1]}')
        print(f'Test EPE 3D left: {self.epe_3d_l["val"][-1]}')
        print(f'Test EPE 3D right: {self.epe_3d_r["val"][-1]}')
        print(f'Test depth loss: {self.loss_depth["val"][-1]}')
        print(f'Test hand acc: {self.acc["val"][-1]}')

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader) -> torch.nn.Module:
        """
        Training loop

        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader
        Returns:
            torch.nn.Module: Trained model
        """

        for epoch in range(self.epochs):

            wandb_dict = {}

            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)

            self.best_epe3d = save_best_model(self.model, self.run_name,
                                              self.epe_3d["val"][-1], self.best_epe3d, save_on_type='less_than')

            wandb_dict["train_loss"] = self.loss["train"][-1]
            wandb_dict["val_loss"] = self.loss["val"][-1]
            wandb_dict["val_epe"] = self.epe["val"][-1]
            wandb_dict["val_hand_acc"] = self.acc["val"][-1]
            wandb_dict["val_depth_loss"] = self.loss_depth["val"][-1]
            wandb_dict["train_depth_loss"] = self.loss_depth["train"][-1]
            wandb_dict["epe_3d_left [mm]"] = self.epe_3d_l["val"][-1]
            wandb_dict["epe_3d_right [mm]"] = self.epe_3d_r["val"][-1]
            wandb_dict["epe_3d [mm]"] = self.epe_3d["val"][-1]
            wandb_dict["epe_3d_best"] = self.best_epe3d

            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}, Val EPE={}, Val Acc: {}".format(
                    epoch + 1,
                    self.epochs,
                    wandb_dict["train_loss"],
                    wandb_dict["val_loss"],
                    wandb_dict["val_epe"],
                    wandb_dict["val_hand_acc"]
                )
            )

            # reducing LR if no improvement in training
            if self.scheduler is not None:
                self.scheduler.step()

            # early stopping if no progress
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(
                    np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]),
                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0

            if self.wandb_logger:
                self.wandb_logger.log(wandb_dict)

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        torch.save(self.model.state_dict(),
                   f'checkpoints/{self.run_name}/final.pth')
        return self.model

    def _epoch_train(self, dataloader: torch.utils.data.DataLoader):
        """
        Training step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """
        self.model.train()
        running_loss = []
        depth_loss = []

        for i, data in enumerate(tqdm(dataloader, 0)):

            inputs = data["img"].to(self.device).type(torch.cuda.FloatTensor)

            heatmaps = data["heatmaps"].to(
                self.device).type(torch.cuda.FloatTensor)

            targets = {
                'heatmaps': heatmaps,
                'handness': torch.stack((data["left_hand_flag"], data["right_hand_flag"]), dim=0).to(self.device),
                'pose_3d_gt': data["kpts_3d_cam"].to(self.device).type(torch.cuda.FloatTensor),
                'kpts2d_img': data["kpts_2d_img"].to(self.device).type(torch.cuda.FloatTensor),
                'kpts2d_norm': data["keypoints"].to(self.device).type(torch.cuda.FloatTensor),
            }

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                results = self.model(inputs)

                loss, loss_depth = self.criterion(
                    results, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            depth_loss.append(loss_depth.detach() * inputs.shape[0])

            running_loss.append(loss.detach() * inputs.shape[0])

            if DEBUG and i == 10:
                break

        epoch_loss = sum(running_loss) / len(dataloader.dataset)
        self.loss["train"].append(epoch_loss)
        self.loss_depth["train"].append(
            sum(depth_loss)/len(dataloader.dataset))

    def _epoch_eval(self, dataloader: torch.utils.data.DataLoader, test=False):
        """
        Evaluation step in epoch

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader
        """

        self.model.eval()
        running_loss = []
        acc_lst = []
        depth_loss = []

        Accuracy = metrics.Accuracy(
            task="multiclass", num_classes=2).to(self.device)

        activation = nn.Sigmoid()

        epe_3d_left = []
        epe_3d_right = []
        correct_left = 0
        correct_right = 0
        epe_2d_left = []
        epe_2d_right = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, 0)):

                inputs = data["img"].to(self.device).type(
                    torch.cuda.FloatTensor)

                heatmaps = data["heatmaps"].to(
                    self.device).type(torch.cuda.FloatTensor)

                labels_left_flag = data["left_hand_flag"].to(self.device)
                labels_right_flag = data["right_hand_flag"].to(self.device)

                targets = {
                    'heatmaps': heatmaps,
                    'handness': torch.stack((data["left_hand_flag"], data["right_hand_flag"]), dim=0).to(self.device),
                    'pose_3d_gt': data["kpts_3d_cam"].to(self.device).type(
                        torch.cuda.FloatTensor),
                    'kpts2d_img': data["kpts_2d_img"].to(self.device).type(
                        torch.cuda.FloatTensor),
                    'kpts2d_norm': data["keypoints"].to(self.device).type(torch.cuda.FloatTensor),
                }

                results = self.model(inputs)

                loss, loss_depth = self.criterion(results, targets)
                depth_loss.append(loss_depth.item())

                pred_kpts2d_img = results['kpts2d_img'].cpu().detach().numpy()
                gt_kpts2d_img = data["kpts_2d_img"].cpu().detach().numpy()

                acc_left = Accuracy(
                    results['left_handness'], labels_left_flag)

                acc_right = Accuracy(
                    results['right_handness'], labels_right_flag)

                acc_lst.append(0.5*(acc_left+acc_right) * inputs.shape[0])

                # If hand was detected and it is in the image:
                left_batch_mask = torch.argmax(
                    activation(results['left_handness']), dim=1)
                right_batch_mask = torch.argmax(
                    activation(results['right_handness']), dim=1)

                valid_pose_index_left = torch.logical_and(
                    left_batch_mask, labels_left_flag).int().cpu().detach().numpy()
                valid_pose_index_right = torch.logical_and(
                    right_batch_mask, labels_right_flag).int().cpu().detach().numpy()

                # Calculate EPE for 3D img for each hand
                pred_kpts3d_img = results['kpts_3d_cam'].cpu().detach().numpy()
                gt_kpts3d_img = data["kpts_3d_cam"].cpu().detach().numpy()

                running_loss.append(loss.item())

                if test == True:

                    epe_2d_l = batch_epe_calculation(
                        pred_kpts2d_img[:, :21, :], gt_kpts2d_img[:, :21, :], batch_mask=valid_pose_index_left) * np.sum(valid_pose_index_left)
                    epe_2d_r = batch_epe_calculation(
                        pred_kpts2d_img[:, 21:, :], gt_kpts2d_img[:, 21:, :], batch_mask=valid_pose_index_right) * np.sum(valid_pose_index_right)

                    epe_3d_left.append(batch_epe_calculation(
                        pred_kpts3d_img[:, :21, :], gt_kpts3d_img[:, :21, :], batch_mask=valid_pose_index_left)*np.sum(valid_pose_index_left))

                    epe_3d_right.append(batch_epe_calculation(
                        pred_kpts3d_img[:, 21:, :], gt_kpts3d_img[:, 21:, :], batch_mask=valid_pose_index_right)*np.sum(valid_pose_index_right))

                    correct_left += np.sum(valid_pose_index_left)
                    correct_right += np.sum(valid_pose_index_right)

                elif test == False:

                    batch_mask_left = np.ones((inputs.shape[0]))
                    batch_mask_right = np.ones((inputs.shape[0]))
                    correct_left += int(inputs.shape[0])
                    correct_right += int(inputs.shape[0])

                    epe_2d_l = batch_epe_calculation(
                        pred_kpts2d_img[:, :21, :], gt_kpts2d_img[:, :21, :], batch_mask=batch_mask_left) * inputs.shape[0]
                    epe_2d_r = batch_epe_calculation(
                        pred_kpts2d_img[:, 21:, :], gt_kpts2d_img[:, 21:, :], batch_mask=batch_mask_right) * inputs.shape[0]

                    epe_3d_left.append(batch_epe_calculation(
                        pred_kpts3d_img[:, :21, :], gt_kpts3d_img[:, :21, :], batch_mask=batch_mask_left) * inputs.shape[0])

                    epe_3d_right.append(batch_epe_calculation(
                        pred_kpts3d_img[:, 21:, :], gt_kpts3d_img[:, 21:, :], batch_mask=batch_mask_right) * inputs.shape[0])

                epe_2d_left.append(epe_2d_l)
                epe_2d_right.append(epe_2d_r)

                if DEBUG and i == 10:
                    break

            all_samples = correct_left + correct_right
            epoch_loss = sum(running_loss)/all_samples
            acc = sum(acc_lst)/len(dataloader.dataset)
            epe_3d_left = sum(epe_3d_left)/correct_left
            epe_3d_right = sum(epe_3d_right)/correct_right
            epe_2d_left = sum(epe_2d_left)/correct_left
            epe_2d_right = sum(epe_2d_right)/correct_right

            self.epe["val"].append((epe_2d_left+epe_2d_right)/2)
            self.epe_3d["val"].append(
                np.mean((epe_3d_left + epe_3d_right)/2))
            self.loss["val"].append(epoch_loss)
            self.acc["val"].append(acc)
            self.loss_depth["val"].append(np.mean(depth_loss))
            self.epe_3d_l["val"].append(epe_3d_left)
            self.epe_3d_r["val"].append(epe_3d_right)

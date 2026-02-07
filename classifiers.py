import os
import json

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

import adabound

from architectures import cnn_model
from deep_training import ModelTraining

from utils.enums import Tasks
from utils.augmentation import between_class, mixup_data, mixup_criterion
from utils.utils import create_results_folder
from utils.metrics import accuracy_mixup, accuracy_score, f1_score


class TaskClassifier(ModelTraining):
    """General Task Classifier (replaced specific naming with general)"""

    def __init__(
        self,
        images_dir,
        csv_file,
        fold,
        num_classes,
        model_task,
        batch_size=24,
        epochs=80,
        model="resnet50",
        pretrained=True,
        optimizer="sgd",
        lr=0.0001,
        weight_decay=5e-4,
        data_augmentation="standard",
        results_path="results",
        experiment_name="experiment",
    ):
        # Dataset parameters
        self.images_dir = images_dir
        self.csv_file = csv_file
        self.fold = fold
        self.num_classes = num_classes
        self.model_task = model_task

        # Training parameters
        self.batch_size = batch_size
        self.epochs = epochs

        # Model parameters
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = model
        self.pretrained = pretrained
        self.data_augmentation = data_augmentation

        # Results parameters
        self.experiment_name = experiment_name
        self.results_folder = create_results_folder(results_path, experiment_name)

    def train(self, loader, model, criterion, optimizer, data_augmentation=None):
        # Tell PyTorch that we are training the model
        model.train()

        metrics = {"loss": 0.0, "acc": 0.0}
        correct = 0
        total = 0

        # Print the shape of the loader
        pbar = tqdm(loader)
        for images, labels in pbar:
            # Loading images on GPU
            if torch.cuda.is_available():
                images, labels = (images.cuda(), labels.cuda(),)

            # Clear gradients parameters
            model.zero_grad()

            # Pass images through the network
            outputs = model(images)  # , outputs_features

            loss = criterion(outputs, labels)

            # Getting gradients
            loss.backward()

            # Clipping gradient
            clip_grad_norm_(model.parameters(), 5)

            # Updating parameters
            optimizer.step()

            # Compute metrics
            ## Loss
            avg_loss = torch.mean(loss)
            metrics["loss"] += avg_loss.data.cpu() * len(images)

            ## Accuracy
            pred = torch.max(outputs.data, 1)[1]
            y_pred = pred.cpu().int()

            correct += accuracy_score(labels.cpu().int(), y_pred) * len(images)

            total += labels.size(0)
            metrics["acc"] = 100.0 * float(correct) / total

            # Update progress bar
            pbar.set_description("[ACC: %.2f]" % metrics["acc"])

        metrics["loss"] = float(metrics["loss"] / len(loader.dataset))

        return metrics

    def validation(self, loader, model, criterion):
        # Tell PyTorch that we are evaluating the model
        model.eval()

        metrics = {"loss": 0.0, "acc": 0.0, "fs": 0.0}
        correct_acc = 0
        correct_fs = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(loader)
            for images, labels in pbar:
                # Loading images on GPU
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()

                # Pass images through the network
                outputs = model(images)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Compute metrics
                ## Loss
                avg_loss = torch.mean(loss)
                metrics["loss"] += avg_loss.data.cpu() * len(images)

                ## Accuracy
                pred = torch.max(outputs.data, 1)[1]
                y_pred = pred.cpu().int()
                y_true = labels.cpu().int()

                correct_acc += accuracy_score(y_true, y_pred) * len(images)
                correct_fs += f1_score(y_true, y_pred, average="macro") * len(images)

                total += labels.size(0)

                metrics["acc"] = 100.0 * float(correct_acc) / total
                metrics["fs"] = 100.0 * float(correct_fs) / total

                # Update progress bar
                pbar.set_description("[ACC: %.2f]" % metrics["acc"])

        metrics["loss"] = float(metrics["loss"] / len(loader.dataset))

        return metrics

    def print_info(self, **kwargs):
        data_type = kwargs.get("data_type")
        metrics = kwargs.get("metrics")
        epoch = kwargs.get("epoch")
        epochs = kwargs.get("epochs")

        if "fs" in metrics:
            print(
                "[Epoch:%3d/%3d][%s][LOSS: %4.2f][Acc: %5.2f][F1: %5.2f]"
                % (
                    epoch + 1,
                    epochs,
                    data_type,
                    metrics["loss"],
                    metrics["acc"],
                    metrics["fs"],
                )
            )
        else:
            print(
                "[Epoch:%3d/%3d][%s][LOSS: %4.2f][Acc: %5.2f]"
                % (
                    epoch + 1,
                    epochs,
                    data_type,
                    metrics["loss"],
                    metrics["acc"],
                )
            )

    def run_training(self, data_loader):
        print("run training: " + self.experiment_name)

        # Dataset
        train_loader, val_loader, test_loader = data_loader(
            self.images_dir,
            self.batch_size,
        )

        # Model
        model = cnn_model(self.model, self.pretrained, self.num_classes)

        # Criterion
        criterion_train = nn.CrossEntropyLoss()
        criterion_val = nn.CrossEntropyLoss()

        # Optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "AdaBound":
            optimizer = adabound.AdaBound(model.parameters(), lr=self.lr, final_lr=0.1, weight_decay=self.weight_decay)
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Optimizer not implemented.")

        record = {}
        record["model"] = self.model
        record["batch_size"] = self.batch_size
        record["lr"] = self.lr
        record["weight_decay"] = self.weight_decay
        record["optimizer"] = self.optimizer
        record["pretrained"] = self.pretrained
        record["data_augmentation"] = self.data_augmentation
        record["epochs"] = self.epochs
        record["train_loss"] = []
        record["val_loss"] = []
        record["train_acc"] = []
        record["val_acc"] = []

        best_fs = 0.0

        for epoch in range(self.epochs):
            # Training
            train_metrics = self.train(
                train_loader, model, criterion_train, optimizer, self.data_augmentation
            )
            self.print_info(
                data_type="TRAIN", metrics=train_metrics, epoch=epoch, epochs=self.epochs
            )

            # Validation
            val_metrics = self.validation(val_loader, model, criterion_val)
            self.print_info(data_type="VAL", metrics=val_metrics, epoch=epoch, epochs=self.epochs)

            # Adjust learning rate
            optimizer = self.adjust_learning_rate(optimizer, self.optimizer, epoch, self.epochs)

            # Recording metrics
            record["train_loss"].append(train_metrics["loss"])
            record["train_acc"].append(train_metrics["acc"])

            record["val_loss"].append(val_metrics["loss"])
            record["val_acc"].append(val_metrics["acc"])

            # Record best model (only based on validation set, no test set involved)
            curr_fs = val_metrics["fs"]
            if (curr_fs > best_fs) and epoch >= 3:
                best_fs = curr_fs
                # Save best model based on validation F1 score
                torch.save(
                    model.state_dict(),
                    os.path.join(self.results_folder, "net_weights.pth"),
                )
                print(f"Best model saved at epoch {epoch+1} (val F1: {best_fs:.2f})")
                torch.save(
                    model.state_dict(),
                    os.path.join(self.results_folder, f"{epoch}_%.2f_net_weights.pth" % val_metrics["acc"]),
                )

            # Saving log (generalized filename)
            with open(os.path.join(self.results_folder, "training_log.json"), "w") as fp:
                json.dump(record, fp, indent=4, sort_keys=True)

        # After all training epochs, evaluate on test set (only once)
        print("\nTraining completed. Evaluating on test set...")
        test_metrics = self.validation(test_loader, model, criterion_val)
        self.print_info(data_type="***TEST***", metrics=test_metrics, epoch=self.epochs-1, epochs=self.epochs)
        # Add test metrics to record
        record["test_loss"] = test_metrics["loss"]
        record["test_acc"] = test_metrics["acc"]
        record["test_f1"] = test_metrics["fs"]
        # Save final record with test metrics
        with open(os.path.join(self.results_folder, "training_log.json"), "w") as fp:
            json.dump(record, fp, indent=4, sort_keys=True)

    def run_test(self, data_loader):
        # Dataset
        _, _, test_loader = data_loader(
            self.images_dir,
            self.batch_size,
        )

        # Loading model
        weights_path = os.path.join(self.results_folder, "net_weights.pth")
        model = cnn_model(self.model, self.pretrained, self.num_classes, weights_path)

        # Tell PyTorch that we are evaluating the model
        model.eval()

        y_pred = np.empty(0)
        y_true = np.empty(0)

        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                # Loading images on GPU
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()

                # Pass images through the network
                outputs = model(images)

                # Compute metrics
                pred = torch.max(outputs.data, 1)[1]
                y_pred = np.concatenate((y_pred, pred.data.cpu().numpy()))
                y_true = np.concatenate((y_true, labels.data.cpu().numpy()))

        return y_true, y_pred

    def get_n_params(self):
        weights_path = os.path.join(self.results_folder, "net_weights.pth")
        # Generalized: replaced hard-coded (5,5) with self.num_classes
        model = cnn_model(self.model, self.pretrained, self.num_classes, weights_path)
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
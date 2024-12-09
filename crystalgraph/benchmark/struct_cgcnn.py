from crystalgraph.benchmark.cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
import csv
import os
import warnings
from random import sample

# from model.utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from crystalgraph.benchmark.cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
from crystalgraph.benchmark.cgcnn.utils import Normalizer, AverageMeter, mae
from crystalgraph.benchmark.cgcnn.model import CrystalGraphConvNet
from crystalgraph.utils import FilePath

# warnings.simplefilter("ignore")
# warnings.warn("deprecated", UserWarning)
# warnings.warn("deprecated", FutureWarning)

import os.path
from typing import Optional

import numpy as np

from .base import TrainingParams, BenchmarkModel, BenchmarkResults


class StructCgcnnParams(TrainingParams):
    """ Parameters for regression CGCNN """

    loss: str = "mse"
    """ loss function used in trainning """

    batch_size: int = 64
    """ batch size """

    epochs: int = 100
    """ number of epochs """

    fine_tune_from: str = 'scratch'
    """ fine tune start point """

    eval_every_n_epochs: int = 1
    log_every_n_steps: int = 50

    lr: float = 1e-2
    """ learning rate """

    momentum: float = 0.9
    """ learning momentum """

    weight_decay: float = 1e-6
    """ weight decay """

    atom_fea_len: int = 64
    """ Number of hidden atom features in the convolutional layers """

    h_fea_len: int = 512
    """  Number of hidden features after pooling """

    n_conv: int = 3
    """ Number of convolutional layers """

    n_h: int = 1
    """ Number of hidden layers after pooling """

    max_num_nbr: int = 12
    """ The maximum number of neighbors while constructing the crystal graph """

    radius: int = 8
    """ The cutoff radius for searching neighbors """

    dmin: int = 0
    """ The minimum distance for constructing GaussianDistance """

    step: float = 0.2
    """ The step size for constructing GaussianDistance """

    device: str = 'cuda'
    """ Training device """


class StructCgcnnResults(BenchmarkResults):
    loss_val: list[float]

    loss_train: list[float]


class StructCgcnn(BenchmarkModel):
    """ a CGCNN regression model """

    benchmark_model_name: str = 'STRUCT-CGCNN'

    benchmark_model_params: StructCgcnnParams

    benchmark_model_results: Optional[StructCgcnnResults] = None

    structure_file_folder_path: str

    structure_file_extension: str

    def load_structure_xy(self, X_train, X_test, y_train, y_test, X_ind_train, X_ind_test):

        id_prop_data_train = []
        y_train_list = y_train.tolist()
        y_test_list = y_test.tolist()
        for i in range(len(X_train)):
            id_prop_data_train.append(
                (X_ind_train[i], y_train_list[i])
            )

        id_prop_data_test = []
        for i in range(len(X_test)):
            id_prop_data_test.append(
                (X_ind_test[i], y_test_list[i])
            )

        id_prop_data_all = id_prop_data_train + id_prop_data_test

        cif_data_train = CIFData(
            root_dir=self.structure_file_folder_path,
            id_prop_data=id_prop_data_train,
            file_extension=self.structure_file_extension,
            max_num_nbr=self.benchmark_model_params.max_num_nbr,
            radius=self.benchmark_model_params.radius,
            dmin=self.benchmark_model_params.dmin,
            step=self.benchmark_model_params.step,
            random_seed=42
        )

        cif_data_test = CIFData(
            root_dir=self.structure_file_folder_path,
            id_prop_data=id_prop_data_test,
            file_extension=self.structure_file_extension,
            max_num_nbr=self.benchmark_model_params.max_num_nbr,
            radius=self.benchmark_model_params.radius,
            dmin=self.benchmark_model_params.dmin,
            step=self.benchmark_model_params.step,
            random_seed=43
        )

        cif_data_all = CIFData(
            root_dir=self.structure_file_folder_path,
            id_prop_data=id_prop_data_all,
            file_extension=self.structure_file_extension,
            max_num_nbr=self.benchmark_model_params.max_num_nbr,
            radius=self.benchmark_model_params.radius,
            dmin=self.benchmark_model_params.dmin,
            step=self.benchmark_model_params.step,
            random_seed=43
        )
        return cif_data_train, cif_data_test, cif_data_all

    def get_data_loaders(self, cif_data_train, cif_data_test):
        collate_fn = collate_pool
        train_loader, valid_loader = get_train_val_test_loader(
            batch_size=self.benchmark_model_params.batch_size,
            train_size=None,
            test_size=None,
            val_size=None,
            train_ratio=0.9,
            val_ratio=0.1,
            test_ratio=0.0,
            dataset=cif_data_train,
            collate_fn=collate_fn,
            pin_memory=self.benchmark_model_params.device != 'cpu',
            return_test=False,
        )
        test_loader, _ = get_train_val_test_loader(
            batch_size=self.benchmark_model_params.batch_size,
            train_size=None,
            test_size=None,
            val_size=None,
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
            dataset=cif_data_test,
            collate_fn=collate_fn,
            pin_memory=self.benchmark_model_params.device != 'cpu',
            return_test=False,
        )
        return train_loader, valid_loader, test_loader

    def get_normalizer(self, cif_data):
        n_dps = len(cif_data)
        if n_dps < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [cif_data[i] for i in range(len(cif_data))]
        else:
            sample_data_list = [cif_data[i] for i in sample(range(len(cif_data)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)
        return normalizer

    @staticmethod
    def load_pre_trained_weights(model_folder: FilePath, model, device):
        pretrained = "model_pretrained.pth"
        pretrained_path = os.path.join(model_folder, pretrained)
        if not os.path.isfile(pretrained_path):
            logger.warning(f"pretrained not found at: {pretrained_path}")
            return model
        load_state = torch.load(pretrained_path, map_location=device)
        model_state = model.state_dict()
        pytorch_total_params = sum(p.numel() for p in model_state.parameters if p.requires_grad)
        logger.info("pretrained parameters: {}".format(pytorch_total_params))
        for name, param in load_state.items():
            if name not in model_state:
                logger.warning(f'NOT loaded: {name}')
                continue
            else:
                logger.warning(f"loaded: {name}")
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            model_state[name].copy_(param)
        logger.info("Loaded pre-trained model with success.")
        return model

    def train_model(self, train_loader, valid_loader, cif_data_train, writer, normalizer, criterion):

        structures, _, _ = cif_data_train[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(
            orig_atom_fea_len, nbr_fea_len, atom_fea_len=self.benchmark_model_params.atom_fea_len,
            n_conv=self.benchmark_model_params.n_conv,
            h_fea_len=self.benchmark_model_params.h_fea_len,
            n_h=self.benchmark_model_params.n_h,
            classification=False,
        )

        model = self.load_pre_trained_weights(self.work_dir, model, self.benchmark_model_params.device)
        model = model.to(self.benchmark_model_params.device)

        layer_list = []
        for name, param in model.named_parameters():
            if 'fc_out' in name:
                logger.info("new layer: {}".format(name))
                layer_list.append(name)
        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        lr_multiplier = 0.2
        if 'scratch' in self.benchmark_model_params.fine_tune_from:
            lr_multiplier = 1
        optimizer = optim.Adam(
            [
                {'params': base_params, 'lr': self.benchmark_model_params.lr * lr_multiplier}, {'params': params}
            ],
            self.benchmark_model_params.lr,
            weight_decay=self.benchmark_model_params.weight_decay
        )

        model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
        # json_dump(self.model_dump(),  os.path.join(self.work_dir, "model_pre_train.json"))

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_mae = np.inf
        best_valid_roc_auc = 0
        loss_train, loss_val = [], []

        device = self.benchmark_model_params.device

        for epoch_counter in range(self.benchmark_model_params.epochs):
            for bn, (input, target, _) in enumerate(train_loader):
                train_loss_batch = []
                if self.benchmark_model_params.device != 'cpu':
                    input_var = (Variable(input[0].to(device, non_blocking=True)),
                                 Variable(input[1].to(device, non_blocking=True)),
                                 input[2].to(device, non_blocking=True),
                                 [crys_idx.to(device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])

                target_normed = normalizer.norm(target)

                if self.benchmark_model_params.device != 'cpu':
                    target_var = Variable(target_normed.to(device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output, _ = model(*input_var)

                # print(output.shape, target_var.shape)
                loss = criterion(output, target_var)

                if bn % self.benchmark_model_params.log_every_n_steps == 0:
                    writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    # self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    logger.info(f'Epoch: {epoch_counter+1}, Batch: {bn}, Loss: {loss.item()}')

                train_loss_batch.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

            loss_train.append(np.mean(train_loss_batch))

            # validate the model if requested
            if epoch_counter % self.benchmark_model_params.eval_every_n_epochs == 0:
                valid_loss, valid_mae = self.validate_model(model, valid_loader, epoch_counter, normalizer, criterion)
                loss_val.append(valid_loss)
                if valid_mae < best_valid_mae:
                    # save the model weights
                    best_valid_mae = valid_mae
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

        self.benchmark_model_results.loss_train = loss_train
        self.benchmark_model_results.loss_val = loss_val

    def validate_model(self, model: CrystalGraphConvNet, valid_loader, epoch_counter, normalizer, criterion):
        losses = AverageMeter()
        mae_errors = AverageMeter()

        device = self.benchmark_model_params.device
        with torch.no_grad():
            model.eval()
            for bn, (input, target, _) in enumerate(valid_loader):
                if self.benchmark_model_params.device != 'cpu':
                    input_var = (Variable(input[0].to(device, non_blocking=True)),
                                 Variable(input[1].to(device, non_blocking=True)),
                                 input[2].to(device, non_blocking=True),
                                 [crys_idx.to(device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])

                target_normed = normalizer.norm(target)

                if self.benchmark_model_params.device != 'cpu':
                    target_var = Variable(target_normed.to(device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output, _ = model(*input_var)

                loss = criterion(output, target_var)

                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

                logger.info('Epoch [{0}] Validate: [{1}/{2}], Loss {loss.val:.4f} ({loss.avg:.4f}), MAE ' \
                            '{mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch_counter + 1, bn + 1, len(valid_loader), loss=losses, mae_errors=mae_errors
                ))

        model.train()

        logger.info('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg

    def test_model(self, model, test_loader,  writer, normalizer, criterion):
        # test steps
        device = self.benchmark_model_params.device
        model_path = os.path.join(writer.log_dir, 'checkpoints', 'model.pth')
        logger.info(f"Testing model path: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Loaded trained model with success.")

        losses = AverageMeter()
        mae_errors = AverageMeter()

        test_targets = []
        test_preds = []
        test_cif_ids = []

        with torch.no_grad():
            model.eval()
            for bn, (input, target, batch_cif_ids) in enumerate(test_loader):
                if device != 'cpu':
                    input_var = (Variable(input[0].to(device, non_blocking=True)),
                                 Variable(input[1].to(device, non_blocking=True)),
                                 input[2].to(device, non_blocking=True),
                                 [crys_idx.to(device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])

                target_normed = normalizer.norm(target)

                if device != "cpu":
                    target_var = Variable(target_normed.to(device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output, _ = model(*input_var)

                loss = criterion(output, target_var)

                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

            logger.info('Test: [{0}/{1}], '
                        'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                        'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                bn, len(test_loader), loss=losses,
                mae_errors=mae_errors))

        with open(os.path.join(writer.log_dir, 'test_results.csv'), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))

        model.train()
        logger.info('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg


    def train_and_eval(self):
        if self.benchmark_model_params.loss == 'mse':
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        writer = SummaryWriter(log_dir=self.work_dir)
        X, y, X_train, X_test, y_train, y_test, X_ind_train, X_ind_test = self.load_lqg_xy()
        logger.info(f"xy loaded train/test: {len(X_train)}/{len(X_test)}")
        cif_data_train, cif_data_test, cif_data_all = self.load_structure_xy(X_train, X_test, y_train, y_test, X_ind_train, X_ind_test)
        train_loader, valid_loader, test_loader = self.get_data_loaders(cif_data_train, cif_data_test,)
        normalizer = self.get_normalizer(cif_data_all)  # TODO shouldn't we use `cif_data_train`?
        self.train_model(train_loader, valid_loader, cif_data_train, writer, normalizer, criterion)
        self.test_model(train_loader, test_loader, writer, normalizer, criterion)
        writer.close()

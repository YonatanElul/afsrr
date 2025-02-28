import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datetime import datetime
from afsrr.utils.loggers import Logger
from afsrr.utils.optim import Optimizer
from afsrr.utils.trainers import Trainer
from afsrr.utils.losses import AFSRRLoss
from afsrr.data.datasets import RPeaksDataset
from afsrr.models.afsrr import JointRegressionModel
from afsrr import EXPERIMENTS_LOGS_DIR, PROCESSED_DATA_DIR
from afsrr.data.data_utils import get_train_val_test_split

import torch
import numpy as np


if __name__ == '__main__':
    # Set seed
    seed = 8783
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define the log dir
    observable_dim = 64
    n_encoder_layers = 3
    l0_units = 64
    trajectory_length = 2
    prediction_horizon = 1
    peaks_per_sample = 64
    peaks_step = 16
    fit_model = True
    nsr_only = False
    record_length_to_use = None
    date = str(datetime.today()).split()[0]
    description = f"{observable_dim}K_" \
                  f"{n_encoder_layers}E_" \
                  f"{l0_units}L0_" \
                  f"{peaks_per_sample}PPS_" \
                  f"{peaks_step}PS_" \
                  f"{trajectory_length}TT_" \
                  f"{prediction_horizon}H"
    experiment_name = f"Train_JointRegressionModel_" \
                      f"{description}_" \
                      f"{'NSR_Only_' if nsr_only else ''}" \
                      f"{f'{record_length_to_use}RL_' if nsr_only else ''}" \
                      f"{date}"
    logs_dir = os.path.join(EXPERIMENTS_LOGS_DIR, experiment_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Define the Datasets & Data loaders
    data_dir = PROCESSED_DATA_DIR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    num_workers = 4
    train_records, _, _ = get_train_val_test_split()
    train_ds = RPeaksDataset(
        mode='Train',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        peaks_per_sample=peaks_per_sample,
        peaks_step=peaks_step,
        nsr_only=nsr_only,
        record_length_to_use=record_length_to_use,
        merge_modes=True,
        records_paths=train_records,
    )
    val_ds = RPeaksDataset(
        mode='Val',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        records_paths=train_records,
        peaks_per_sample=peaks_per_sample,
        peaks_step=peaks_step,
        nsr_only=False,
    )
    test_ds = RPeaksDataset(
        mode='Test',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        records_paths=train_records,
        peaks_per_sample=peaks_per_sample,
        peaks_step=peaks_step,
        nsr_only=False,
    )
    if nsr_only:
        invalid_files = train_ds.invalid_files + val_ds.invalid_files + test_ds.invalid_files
        invalid_files = list(np.unique(invalid_files).tolist())

        train_ds = RPeaksDataset(
            mode='Train',
            temporal_horizon=trajectory_length,
            prediction_horizon=prediction_horizon,
            records_paths=train_records,
            peaks_per_sample=peaks_per_sample,
            peaks_step=peaks_step,
            nsr_only=False,
            invalid_inds=invalid_files,
            merge_modes=True,
        )
        val_ds = RPeaksDataset(
            mode='Val',
            temporal_horizon=trajectory_length,
            prediction_horizon=prediction_horizon,
            records_paths=train_records,
            peaks_per_sample=peaks_per_sample,
            peaks_step=peaks_step,
            nsr_only=False,
            invalid_inds=invalid_files,
        )
        test_ds = RPeaksDataset(
            mode='Test',
            temporal_horizon=trajectory_length,
            prediction_horizon=prediction_horizon,
            records_paths=train_records,
            peaks_per_sample=peaks_per_sample,
            peaks_step=peaks_step,
            nsr_only=False,
            invalid_inds=invalid_files,
        )

    pin_memory = True
    drop_last = False
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # Define the model
    states_dim = peaks_per_sample
    m_dynamics = train_ds.n_patients
    units_factor = 2
    activation = 'leakyrelu'
    final_activation = None
    dropout = 0.2
    bias = True
    skip_connections = True
    use_revin = True
    dynamics_model_params = {
        'm_dynamics': m_dynamics,
        'observable_dim': observable_dim,
        'states_dim': states_dim,
        'n_encoder_layers': n_encoder_layers,
        'l0_units': l0_units,
        'units_factor': units_factor,
        'activation': activation,
        'final_activation': final_activation,
        'dropout': dropout,
        'bias': bias,
        'k_forward_prediction': prediction_horizon,
        'skip_connections': skip_connections,
        'use_revin': use_revin,
    }
    model_params = {
        'dynamics_model_params': dynamics_model_params,
        'n_classes': 1,
        'n_classification_layers': 3,
        'classifier_activation': 'leakyrelu',
        'classifier_units': 64,
    }
    model = JointRegressionModel(
        **model_params
    )
    model.to(device)

    # Define the optimizer
    lr = 0.001
    weight_decay = 0.0001
    optimizer_hparams = {
        'lr': lr,
        'weight_decay': weight_decay,
    }
    optimizers = [
        torch.optim.AdamW(
            params=model.parameters(),
            **optimizer_hparams,
        ),
    ]
    scheduler_hparams = {
        'mode': 'min',
        'factor': 0.5,
        'patience': 5,
        'threshold': 1e-4,
        'threshold_mode': 'rel',
        'cooldown': 0,
        'min_lr': 1e-6,
        'eps': 1e-8,
        'verbose': True,
    }
    schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizers[0],
            **scheduler_hparams
        ),
    ]
    num_epochs = 100
    optimizer = Optimizer(
        optimizers=optimizers,
        schedulers=schedulers,
    )

    # Define the loss & evaluation functions
    alpha = 0.1
    beta = 1
    loss_fn = AFSRRLoss(alpha=alpha, beta=beta)
    evaluation_metric = AFSRRLoss(alpha=alpha, beta=beta)

    # Define the logger
    logger = Logger(
        log_dir=EXPERIMENTS_LOGS_DIR,
        experiment_name=experiment_name,
        max_elements=2,
    )

    # Define the trainer
    checkpoints = True
    early_stopping = None
    checkpoints_mode = 'min'
    clip_grad_value = None
    filter_nan_grads = False
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        device=device,
        logger=logger,
        clip_grad_value=clip_grad_value,
        filter_nan_grads=filter_nan_grads,
    )

    # Write Scenario Specs
    specs = {
        'Data Specs': '',
        "seed": seed,
        'trajectory_length': trajectory_length,
        'prediction_horizon': prediction_horizon,
        'peaks_per_sample': peaks_per_sample,
        'peaks_step': peaks_step,
        'DataLoader Specs': '',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'Model Specs': '',
        'Model': type(model).__name__,
    }
    specs.update(model_params)
    loss_params = {
        'Loss Specs': '',
        'loss_fn': f"{loss_fn}",
        'eval_fn': f"{evaluation_metric}",
        'Trainer Specs': '',
        'num_epochs': num_epochs,
        'checkpoints': checkpoints,
        'early_stopping': early_stopping,
        'checkpoints_mode': checkpoints_mode,
        'clip_grad_value': clip_grad_value,
        'filter_nan_grads': filter_nan_grads,
        'Optimizer Specs': '',
        'optimizer': type(optimizers[0]).__name__,
    }
    specs.update(loss_params)
    specs.update(optimizer_hparams)
    specs['LR Scheduler Specs'] = ''
    specs['agnostic_scheduler'] = type(schedulers[0]).__name__
    specs.update(scheduler_hparams)

    specs_file = os.path.join(logs_dir, 'data_specs.txt')
    with open(specs_file, 'w') as f:
        for k, v in specs.items():
            f.write(f"{k}: {str(v)}\n")

    print("Fitting the model")
    if fit_model:
        trainer.fit(
            dl_train=train_dl,
            dl_val=val_dl,
            num_epochs=num_epochs,
            checkpoints=checkpoints,
            checkpoints_mode=checkpoints_mode,
            early_stopping=early_stopping,
        )

    # Define the test-set
    print("Evaluating over the test set")
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    model = JointRegressionModel(
        **model_params
    )
    model_ckpt_path = f"{logs_dir}/BestModel.PyTorchModule"  # loading best model
    model_ckp = torch.load(model_ckpt_path)
    model.load_state_dict(model_ckp['model'])
    model.to(device)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        device=device,
        logger=logger,
    )

    # Evaluate
    trainer.evaluate(
        dl_test=test_dl,
        ignore_cap=True,
    )

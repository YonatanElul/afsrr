import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datetime import datetime
from afsrr.utils.loggers import Logger
from afsrr.utils.optim import Optimizer
from afsrr.utils.trainers import Trainer
from afsrr.utils.losses import ModuleLoss
from afsrr.data.datasets import RPeaksDataset
from afsrr.models.afsrr import MIDST, JointRegressionModel
from afsrr import EXPERIMENTS_LOGS_DIR, PROCESSED_DATA_DIR
from afsrr.data.data_utils import get_train_val_test_split

import glob
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
    nsr_only = True
    fixed = True
    n_hours = 0.5
    min_per_hour = 60
    beats_per_min = 85
    record_length_to_use = int(n_hours * min_per_hour * beats_per_min)
    fit_model = True
    nsr_from_start = False
    nsr_from_end = False
    nsr_from_middle = True
    nsr_loc = 'Start' if nsr_from_start else ('Middle' if nsr_from_middle else 'End')
    date = str(datetime.today()).split()[0]
    description = f"{observable_dim}K_" \
                  f"{n_encoder_layers}E_" \
                  f"{l0_units}L0_" \
                  f"{peaks_per_sample}PPS_" \
                  f"{peaks_step}PS_" \
                  f"{trajectory_length}TT_" \
                  f"{prediction_horizon}H" \
                  f"{f'_NSR_Only_{record_length_to_use}_Beats_{nsr_loc}' if nsr_only else ''}" \

    experiment_name = f"Test_JointRegressionModel_{'Fixed_' if fixed else ''}Test_{description}_{date}"
    logs_dir = os.path.join(EXPERIMENTS_LOGS_DIR, experiment_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Define the Datasets & Data loaders
    data_dir = PROCESSED_DATA_DIR
    test_dir = os.path.join(data_dir, 'Test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    num_workers = 4
    _, _, test_records = get_train_val_test_split
    train_ds = RPeaksDataset(
        mode='Train',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        dir_path=test_dir,
        peaks_per_sample=peaks_per_sample,
        record_length_to_use=record_length_to_use,
        peaks_step=peaks_step,
        nsr_only=nsr_only,
        merge_modes=True,
        nsr_from_start=nsr_from_start,
        nsr_from_end=nsr_from_end,
        nsr_from_middle=nsr_from_middle,
        records_paths=test_records,
    )
    val_ds = RPeaksDataset(
        mode='Val',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        dir_path=test_dir,
        peaks_per_sample=peaks_per_sample,
        record_length_to_use=record_length_to_use,
        peaks_step=peaks_step,
        nsr_only=nsr_only,
        merge_modes=True,
        nsr_from_start=nsr_from_start,
        nsr_from_end=nsr_from_end,
        nsr_from_middle=nsr_from_middle,
        records_paths=test_records,
    )
    test_ds = RPeaksDataset(
        mode='Test',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        dir_path=test_dir,
        peaks_per_sample=peaks_per_sample,
        record_length_to_use=record_length_to_use,
        peaks_step=peaks_step,
        nsr_only=nsr_only,
        merge_modes=True,
        nsr_from_start=nsr_from_start,
        nsr_from_end=nsr_from_end,
        nsr_from_middle=nsr_from_middle,
        records_paths=test_records,
    )

    invalid_files = train_ds.invalid_files + val_ds.invalid_files + test_ds.invalid_files
    invalid_files = list(np.unique(invalid_files).tolist())

    train_ds = RPeaksDataset(
        mode='Train',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        dir_path=test_dir,
        peaks_per_sample=peaks_per_sample,
        record_length_to_use=record_length_to_use,
        peaks_step=peaks_step,
        nsr_only=nsr_only,
        invalid_inds=invalid_files,
        nsr_from_start=nsr_from_start,
        nsr_from_end=nsr_from_end,
        nsr_from_middle=nsr_from_middle,
        records_paths=test_records,
    )
    val_ds = RPeaksDataset(
        mode='Val',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        dir_path=test_dir,
        peaks_per_sample=peaks_per_sample,
        record_length_to_use=record_length_to_use,
        peaks_step=peaks_step,
        nsr_only=nsr_only,
        invalid_inds=invalid_files,
        nsr_from_start=nsr_from_start,
        nsr_from_end=nsr_from_end,
        nsr_from_middle=nsr_from_middle,
        records_paths=test_records,
    )
    test_ds = RPeaksDataset(
        mode='Test',
        temporal_horizon=trajectory_length,
        prediction_horizon=prediction_horizon,
        dir_path=test_dir,
        peaks_per_sample=peaks_per_sample,
        record_length_to_use=record_length_to_use,
        peaks_step=peaks_step,
        nsr_only=nsr_only,
        invalid_inds=invalid_files,
        nsr_from_start=nsr_from_start,
        nsr_from_end=nsr_from_end,
        nsr_from_middle=nsr_from_middle,
        records_paths=test_records,
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
    test_dl = torch.utils.data.DataLoader(
        test_ds,
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
    non_stationary_norm = False
    model_params = {
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
        'non_stationary_norm': non_stationary_norm,
    }
    model = MIDST(
        **model_params
    )
    model.to(device)

    if fixed:
        # Load the trained model
        trained_model_ckpt_path = glob.glob(
            os.path.join(
                EXPERIMENTS_LOGS_DIR,
                f"Train_JointRegressionModel_*_{date}",
                'BestModel.PyTorchModule',
            )
        )[0]
        trained_ckpt = torch.load(trained_model_ckpt_path)['model']
        trained_model_params = {
            'dynamics_model_params': model_params.copy(),
            'n_classes': 1,
            'n_classification_layers': 3,
            'classifier_activation': 'leakyrelu',
            'classifier_units': 64,
        }
        m = len([m for m in trained_ckpt if 'S_per_t_per_m' in m])
        trained_model_params['midst_params']['m_dynamics'] = m
        trained_model = JointRegressionModel(
            **trained_model_params
        )
        trained_model.load_state_dict(trained_ckpt)
        trained_model.to(device)

        # Copy the existing systems and freeze their parameters
        model.encoder = trained_model.midst.encoder
        model.encoder.requires_grad_(False)
        model.decoder = trained_model.midst.decoder
        model.decoder.requires_grad_(False)
        model._U_per_t = trained_model.midst._U_per_t
        model._U_per_t.requires_grad_(False)
        model._V_per_t = trained_model.midst._V_per_t
        model._V_per_t.requires_grad_(False)

        del trained_ckpt, trained_model

    # Define the optimizer
    lr = 0.001
    weight_decay = 0.0001
    optimizer_hparams = {
        'lr': lr,
        'weight_decay': weight_decay,
    }
    optimizers = [
        torch.optim.AdamW(
            params=model.koopman_params(),
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
    loss_fn = ModuleLoss(model=torch.nn.MSELoss())
    evaluation_metric = ModuleLoss(model=torch.nn.L1Loss())

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
    model = MIDST(
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

import time
import torch
import numpy as np
from tqdm import tqdm
import torchvision

from spnflow.torch.callbacks import EarlyStopping

class RunningAverageMetric:
    """Running (batched) average metric"""
    def __init__(self, batch_size):
        """
        Initialize a running average metric object.

        :param batch_size: The batch size.
        """
        self.batch_size = batch_size
        self.metric_accumulator = 0.0
        self.n_metrics = 0

    def __call__(self, x):
        """
        Accumulate a metric.

        :param x: The metric value.
        """
        self.metric_accumulator += x
        self.n_metrics += 1

    def average(self):
        """
        Get the metric average.

        :return: The metric average.
        """
        return self.metric_accumulator / (self.n_metrics * self.batch_size)


def torch_train(
        model,
        data_train,
        data_val,
        lr=1e-3,
        batch_size=100,
        epochs=1000,
        patience=30,
        optimizer_class=torch.optim.Adam,
        weight_decay=0.0,
        n_workers=4,
        device=None,
        verbose=True,
        writer=None,
        checkpoint_name=None,
        continue_checkpoint=None,
        gpu_id=0,
        data_parallel=False,
):
    """
    Train a Torch model.

    :param model: The model to train.
    :param data_train: The train dataset.
    :param data_val: The validation dataset.
    :param setting: The train setting. It can be either 'generative' or 'discriminative'.
    :param lr: The learning rate to use.
    :param batch_size: The batch size for both train and validation.
    :param epochs: The number of epochs.
    :param patience: The epochs patience for early stopping.
    :param optimizer_class: The optimizer class to use.
    :param weight_decay: L2 regularization factor.
    :param n_workers: The number of workers for data loading.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :param verbose: Whether to enable verbose mode.
    :param data_parallel: Whether model is trained using DataParallel.
    :return: The train history.
    """
    # Get the device to use
    if device is None:
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('Train using device: ' + str(device))

    # Setup the data loaders
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    # Instantiate the optimizer
    model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Load model and optimizer if continue checkpoint provided
    if continue_checkpoint is not None:
        cp = torch.load(continue_checkpoint)
        model.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])

    return torch_train_generative(model, train_loader, val_loader, optimizer, epochs, patience, device, verbose, writer, checkpoint_name, data_parallel)


def torch_train_generative(model, train_loader, val_loader, optimizer, epochs, patience, device, verbose, writer, checkpoint_name, data_parallel):
    """
    Train a Torch model in generative setting.

    :param model: The model.
    :param train_loader: The train data loader.
    :param val_loader: The validation data loader.
    :param optimizer: The optimize to use.
    :param epochs: The number of epochs.
    :param patience: The epochs patience for early stopping.
    :param device: The device to use for training.
    :param verbose: Whether to enable verbose mode.
    :return: The train history.
    """
    # Instantiate the train history
    history = {
        'train': [], 'validation': []
    }

    # Instantiate the early stopping callback
    early_stopping = EarlyStopping(patience=patience)

    # Move the model to device
    if not data_parallel:
        model.to(device)

    log_inputs = True
    for epoch in range(epochs):
        start_time = time.time()

        # Initialize the tqdm train data loader, if verbose is specified
        if verbose:
            tk_train = tqdm(
                train_loader, leave=False, bar_format='{l_bar}{bar:32}{r_bar}',
                desc='Train Epoch %d/%d' % (epoch + 1, epochs)
            )
        else:
            tk_train = train_loader

        # Make sure the model is set to train mode
        model.train()

        # Training phase
        running_train_loss = RunningAverageMetric(train_loader.batch_size)
        running_train_ll = RunningAverageMetric(train_loader.batch_size)
        for inputs, targets in tk_train:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            elbo, ll = model(inputs)
            loss = elbo.mean(dim=0)
            running_train_loss(loss)
            running_train_ll(ll.mean(dim=0))
            loss.backward()
            optimizer.step()

            # Log the inputs in tensorboard
            if log_inputs:
                if len(inputs.shape) > 3:
                    writer.add_image('input_images', torchvision.utils.make_grid(inputs), epoch)
                log_inputs = False

        # Initialize the tqdm validation data loader, if verbose is specified
        if verbose:
            tk_val = tqdm(
                val_loader, leave=False, bar_format='{l_bar}{bar:32}{r_bar}',
                desc='Validation Epoch %d/%d' % (epoch + 1, epochs)
            )
        else:
            tk_val = val_loader

        # Make sure the model is set to evaluation mode
        model.eval()

        # Validation phase
        running_val_loss = RunningAverageMetric(val_loader.batch_size)
        running_val_ll = RunningAverageMetric(val_loader.batch_size)
        with torch.no_grad():
            for inputs, targets in tk_val:
                inputs = inputs.to(device)
                elbo, ll = model(inputs)
                loss = elbo.mean(dim=0)
                running_val_loss(loss)
                running_val_ll(ll.mean(dim=0))

        # Get the average train and validation losses and print it
        end_time = time.time()
        train_loss = running_train_loss.average()
        val_loss = running_val_loss.average()
        train_ll = running_train_ll.average()
        val_ll = running_val_ll.average()
        print('Epoch %d/%d - train_loss: %.4f, train_ll: %.4f, validation_loss: %.4f [%ds], validation_ll: %.4f' %
              (epoch + 1, epochs, train_loss, train_ll, val_loss, val_ll, end_time - start_time))

        # Log the loss in tensorboard
        writer.add_scalar('running_train_loss', train_loss, epoch)
        writer.add_scalar('running_val_loss', val_loss, epoch)

        # Append losses to history data
        history['train'].append(train_loss)
        history['validation'].append(val_loss)

        # Check if training should stop according to early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print('Early Stopping... Best Loss: %.4f' % early_stopping.best_loss)
            break

        # Save the model every epoch
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, checkpoint_name)

    return history


def torch_test(
        model,
        data_test,
        setting,
        batch_size=100,
        n_workers=4,
        device=None
):
    """
    Test a Torch model.

    :param model: The model to test.
    :param data_test: The test dataset.
    :param setting: The test setting. It can be either 'generative' or 'discriminative'.
    :param batch_size: The batch size for testing.
    :param n_workers: The number of workers for data loading.
    :param device: The device used for training. If it's None 'cuda' will be used, if available.
    :return: The mean log-likelihood and two standard deviations if setting='generative'.
             The negative log-likelihood and accuracy if setting='discriminative'.
    """
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Setup the data loader
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    return torch_test_generative(model, test_loader, device)


def torch_test_generative(model, test_loader, device):
    """
    Test a Torch model in generative setting.

    :param model: The model to test.
    :param test_loader: The test data loader.
    :param device: The device used for testing.
    :return: The mean log-likelihood and two standard deviations.
    """
    # Make sure the model is set to evaluation mode
    model.eval()

    test_ll = np.array([])
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            elbo, ll = model(inputs).cpu().numpy().flatten()
            test_ll = np.hstack((test_ll, ll.mean(dim=0)))
    mu_ll = np.mean(test_ll)
    sigma_ll = 2.0 * np.std(test_ll) / np.sqrt(len(test_ll))

    return mu_ll, sigma_ll
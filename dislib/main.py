import numpy as np
from scipy.stats import pearsonr as corr

import torch
from torch import nn
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, TensorDataset

import sys
import os
from tqdm import tqdm
import argparse
import shutil

# This builds on https://github.com/facebookresearch/disentangling-correlated-factors
sys.path.append('PATH/TO/disentangling-correlated-factors')
from datasets.utils import get_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


SEED = 42
DATA_DIR = 'PATH/TO/data'
SAVE_PATH = 'MY/results'


DATASETS = [
    'dsprites', 
    'shapes3d', 
    'mpi3d', 
    'mpi3d_real', 
    'cars3d', 
    'smallnorb'
]

def get_data(args):
    print('Loading:', args.dataset)
    if args.dataset == 'dsprites':
        data = get_dataset(args.dataset)(
            subset=1, root=os.path.join(DATA_DIR, args.dataset),
            factors_to_use=['shape', 'scale', 'orientation', 'posX', 'posY']
        )
    else:
        data = get_dataset(args.dataset)(subset=1, root=os.path.join(DATA_DIR, args.dataset))

    nc = num_channels[args.dataset]
    cat = categorical[args.dataset][0]
    cat_ind = [j for j, item in enumerate(data.lat_names) if item == cat][0]
    if args.debug:
        print('Target', cat, cat_ind)

    # split train, val, test
    val_ratio = 0.1
    test_ratio = 0.1
    N = data.imgs.shape[0]
    if args.debug:
        print('Total Data', N)
    num_val = int(N * val_ratio)
    num_test = int(N * test_ratio)
    num_train = N - num_val - num_test

    np.random.seed(args.seed)
    indices = np.random.choice(N, N, replace=False)
    val_ind = indices[:num_val]
    test_ind = indices[num_val:num_val + num_test]
    train_ind = indices[num_val + num_test:]
    if args.debug:
        print(
            'num_val', len(val_ind),
            'num_test', len(test_ind),
            'num_train', len(train_ind),
            'sum', len(val_ind) + len(test_ind) + len(train_ind)
        )
    
    images = data.imgs.copy()
    if nc == 1:
        images = images.reshape(-1, 1, 64, 64)
    else:
        if images.shape[-1] == 3:
            images = np.transpose(images, (0, 3, 1, 2))
    images = images / 255.
    if args.debug:
        print('images', images.shape, images.dtype)
    
    targets = data.lat_values.copy()
    for i in range(targets.shape[1]):
        tmp = np.unique(targets[:, i])
        targets[:, i] /= tmp[1]
    targets = targets.astype(int)
    if args.debug:
        for i in range(targets.shape[1]):
            print(i, np.unique(targets[:, i]))
        print('targets', targets.shape, targets.dtype)
    
    train_data = TensorDataset(
        torch.tensor(images[train_ind]),
        torch.tensor(targets[train_ind])
    )
    val_data = TensorDataset(
        torch.tensor(images[val_ind]),
        torch.tensor(targets[val_ind])
    )
    test_data = TensorDataset(
        torch.tensor(images[test_ind]),
        torch.tensor(targets[test_ind])
    )

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if args.dataset == 'cars3d':
        # missing classes in cars3d
        out_size = 183
    else:
        out_size = len(np.unique(targets[:, cat_ind]))

    return (train_dataloader, val_dataloader, test_dataloader,
            data, out_size, nc, cat_ind)

# sort latents
categorical = {
    'dsprites': ['shape'],
    'shapes3d': ['objType'],
    'mpi3d': ['objShape'],
    'mpi3d_real': ['objShape'],
    'cars3d': ['object_type'],
    'smallnorb': ['category', 'instance']
}
continuous = {
    'dsprites': ['scale', 'posX', 'posY'],
    'shapes3d': ['objSize', 'objAzimuth'],
    'mpi3d': ['posX', 'posY'],
    'mpi3d_real': ['posX', 'posY'],
    'cars3d': ['elevation'],
    'smallnorb': ['elevation']
}
manifold = {
    'dsprites': ['orientation'],
    'shapes3d': ['floorCol', 'wallCol', 'objCol'],
    'mpi3d': [],
    'mpi3d_real': [],
    'cars3d': ['azimuth'],
    'smallnorb': ['rotation']
}
other = {
    'dsprites': [],
    'shapes3d': [],
    'mpi3d': ['objCol', 'objSize', 'cameraHeight', 'backCol'],
    'mpi3d_real': ['objCol', 'objSize', 'cameraHeight', 'backCol'],
    'cars3d': [],
    'smallnorb': ['lighting']
}

num_channels = {'dsprites': 1, 'shapes3d': 3, 'mpi3d': 3, 
                'mpi3d_real': 3, 'cars3d': 3, 'smallnorb': 1}

def get_model(model, nc, out_size, seed):
    torch.manual_seed(seed)
    if model == 'image':
        net = nn.Sequential(
            nn.Flatten(1),
        ).to(device)
    elif model == 'linear':
        net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(nc * 64 * 64, 512),
        ).to(device)
    elif model == 'mlp':
        net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(nc * 64 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        ).to(device)
    elif model == 'cnn':
        net = resnet18()
        net.conv1 = nn.Conv2d(nc, 64, 7, 2, 3, bias=False)
        net.fc = nn.Identity()
        net = net.to(device)
    else:
        raise ValueError("Model %s undefined." % model)
    return net

def LinReg(X, Y):
    beta = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return X @ beta

def train(args, dataset, log_file):
    with open(log_file, 'a') as file:
        print('\n\nTraining:', file=file)
    (train_dataloader, val_dataloader, test_dataloader,
    data, out_size, nc, cat_ind) = dataset
    net = get_model(args.model, nc, out_size, args.seed)
    readout = nn.Linear(512, out_size).to(device)

    optimizer = torch.optim.Adam(
        list(net.parameters())
        + list(readout.parameters()),
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(args.num_epochs):
        net.train()
        run_loss = []
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for x, y in progress_bar:
            x = x.to(torch.float32).to(device)
            y = y.to(torch.long)
            z = net(x)
            y_ = readout(z)
            if args.debug:
                print('y', y.shape, y.dtype, y.min(), y.max(), y)
                print('y_', y_.shape, y_.dtype, y_.min(), y_.max(), y_)
            loss = criterion(y_, y[:, cat_ind].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(run_loss))
            if args.debug:
                break

        net.eval()
        Y, Z, Y_ = [], [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(val_dataloader):
                x = x.to(torch.float32).to(device)
                z = net(x)
                y_ = readout(z)
                Y.append(y.to(torch.long).detach().cpu().numpy())
                Y_.append(y_.argmax(1).detach().cpu().numpy())
                Z.append(z.detach().cpu().numpy())
                if args.debug:
                    break
        Y = np.concatenate(Y)
        Y_ = np.concatenate(Y_)
        Z = np.concatenate(Z)
        val_acc = np.mean(Y[:, cat_ind] == Y_)
        print('val_acc', val_acc)
        Y_ = LinReg(Z, Y)
        with open(log_file, 'a') as file:
            for i in range(Y.shape[1]):
                print('Coordinate', i, data.lat_names[i], corr(Y[:, i], Y_[:, i])[0], file=file)
        with open(log_file, 'a') as file:
            print(
                'Epoch', epoch, 
                'Loss', np.mean(run_loss),
                'val_acc', val_acc,
                file=file
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), os.path.join(args.log_dir, 'model.pth'))
            torch.save(readout.state_dict(), os.path.join(args.log_dir, 'readout.pth'))
        if args.debug:
            break
        if best_val_acc > 0.999:
            with open(log_file, 'a') as file:
                print('early stopping', file=file)
            break


def evaluate(args, dataset, log_file):
    with open(log_file, 'a') as file:
        print('\n\nEvaluating:', file=file)
    (train_dataloader, val_dataloader, test_dataloader,
    data, out_size, nc, cat_ind) = dataset
    net = get_model(args.model, nc, out_size, args.seed)

    # prepare model
    if args.model == 'image':
        pass
    else:
        net.load_state_dict(torch.load(os.path.join(args.log_dir, 'model.pth')))
        net.eval()

    x_train, y_train, x_val, y_val = [], [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(torch.float32).to(device)
            y_train.append(y.to(torch.long).detach().cpu().numpy())
            x_train.append(net(x).detach().cpu().numpy())
            if args.debug:
                break
        for i, (x, y) in enumerate(val_dataloader):
            x = x.to(torch.float32).to(device)
            y_val.append(y.to(torch.long).detach().cpu().numpy())
            x_val.append(net(x).detach().cpu().numpy())
            if args.debug:
                break
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    if args.debug:
        with open(log_file, 'a') as file:
            print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, file=file)

    # decode all coordinates
    tmp = np.linalg.pinv(x_train.T @ x_train) @ x_train.T
    for i in range(y_train.shape[1]):
        y = y_train[:, i].copy() * 1.
        y -= np.mean(y)
        y /= np.std(y)
        beta = tmp @ y
        y_train_ = x_train @ beta
        y_val_ = x_val @ beta
        with open(log_file, 'a') as file:
            print(
                'Coordinate', i, data.lat_names[i],
                '\ntrain', corr(y_train[:, i], y_train_), 
                '\nval', corr(y_val[:, i], y_val_), 
                file=file
            )
    

def setup_logging(args):
    # Ensure the log directory exists and is empty
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)  # Remove the directory and all its contents
    os.makedirs(args.log_dir)  # Recreate the directory
    log_file = os.path.join(args.log_dir, 'log.txt')
    with open(log_file, 'a') as file:
        print(args, file=file)
    return log_file


class Args:
    def __init__(self):
        self.dataset = 'dsprites'
        self.seed = SEED
        self.batch_size = int(2 ** 12)
        self.model = 'cnn'
        self.lr = 1e-3
        self.num_epochs = 100
        self.log_dir = SAVE_PATH
        self.debug = False

    def __str__(self):
        return 'Args:\n' + '\n'.join(f'{attr}: {value}' for attr, value in 
                                     self.__dict__.items() if not attr.startswith('__'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script LinRep")
    parser.add_argument('--setting', type=int, default=0, help='train setting see below')
    setting = parser.parse_args().setting

    settings = []
    for rep in range(10):
        for dataset in DATASETS[:]:
            for model in [
                'image',
                'linear',
                'mlp',
                'cnn'
                ]:
                settings.append([rep, dataset, model])
    assert setting >= 0 and setting < len(settings)
    rep, dataset, model = settings[setting]
    print('Running setting:', 'rep:', rep, 'dataset:', dataset, 'model:', model)

    
    args = Args()
    args.seed = SEED + rep
    args.dataset = dataset
    args.model = model
    args.log_dir = os.path.join(SAVE_PATH, '%s_model_%s_rep_%s' % (
        dataset, model, rep)
    )

    log_file = setup_logging(args)
    dataset = get_data(args)
    if model != 'image':
        train(args, dataset, log_file)
    evaluate(args, dataset, log_file)

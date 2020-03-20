from model import *
from dataset import *

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

##
class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.norm = args.norm

        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.name_data = args.name_data
        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.gpu_ids = args.gpu_ids

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG, optimG, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': netG.state_dict(),
                    'optimG': optimG.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG, optimG=[], epoch=[], mode='train'):

        if not os.path.exists(dir_chck) or not os.listdir(dir_chck):
            epoch = 0
            if mode == 'train':
                return netG, optimG, epoch
            elif mode == 'test':
                return netG, epoch

        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            optimG.load_state_dict(dict_net['optimG'])

            return netG, optimG, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])

            return netG, epoch

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        size_data = (self.ny_in, self.nx_in, self.nch_in)
        size_window = (5, 5)

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')
        dir_data_val = os.path.join(self.dir_data, name_data, 'val')

        dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')
        dir_log_val = os.path.join(self.dir_log, self.scope, name_data, 'val')

        dir_result_train = os.path.join(self.dir_result, self.scope, name_data, 'train')
        dir_result_val = os.path.join(self.dir_result, self.scope, name_data, 'val')
        if not os.path.exists(os.path.join(dir_result_train, 'images')):
            os.makedirs(os.path.join(dir_result_train, 'images'))
        if not os.path.exists(os.path.join(dir_result_val, 'images')):
            os.makedirs(os.path.join(dir_result_val, 'images'))

        transform_train = transforms.Compose([Normalize(mean=0.5, std=0.5), RandomFlip(), RandomCrop((self.ny_load, self.nx_load)), ToTensor()])
        transform_val = transforms.Compose([Normalize(mean=0.5, std=0.5), RandomFlip(), RandomCrop((self.ny_load, self.nx_load)), ToTensor()])

        transform_inv = transforms.Compose([ToNumpy(), Denormalize(mean=0.5, std=0.5)])

        dataset_train = Dataset(dir_data_train, data_type=self.data_type, transform=transform_train, sgm=25, ratio=0.9, size_data=size_data, size_window=size_window)
        dataset_val = Dataset(dir_data_val, data_type=self.data_type, transform=transform_val, sgm=25, ratio=0.9, size_data=size_data, size_window=size_window)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

        num_train = len(dataset_train)
        num_val = len(dataset_val)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))
        num_batch_val = int((num_val / batch_size) + ((num_val % batch_size) != 0))

        if nch_out == 1:
            cmap = 'gray'
        else:
            cmap = None

        ## setup network
        # netG = UNet(nch_in, nch_out, nch_ker, norm)
        netG = ResNet(nch_in, nch_out, nch_ker, norm)

        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        fn_REG = nn.L1Loss().to(device)  # Regression loss: L1
        # fn_REG = nn.MSELoss().to(device)     # Regression loss: L2

        paramsG = netG.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG, optimG, st_epoch = self.load(dir_chck, netG, optimG, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)
        writer_val = SummaryWriter(log_dir=dir_log_val)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()

            loss_G_train = []

            for batch, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

                label = data['label'].to(device)
                input = data['input'].to(device)
                mask = data['mask'].to(device)

                # forward netG
                output = netG(input)

                # backward netG
                optimG.zero_grad()

                loss_G = fn_REG(output * (1 - mask), label * (1 - mask))

                loss_G.backward()
                optimG.step()

                # get losses
                loss_G_train += [loss_G.item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
                      % (epoch, batch, num_batch_train, np.mean(loss_G_train)))

                if should(num_freq_disp):
                    ## show output
                    input = transform_inv(input)
                    label = transform_inv(label)
                    output = transform_inv(output)

                    input = np.clip(input, 0, 1)
                    label = np.clip(label, 0, 1)
                    output = np.clip(output, 0, 1)
                    dif = np.clip(abs(label - input), 0, 1)

                    writer_train.add_images('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                    writer_train.add_images('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                    writer_train.add_images('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

                    for j in range(label.shape[0]):
                        # name = num_train * (epoch - 1) + num_batch_train * (batch - 1) + j
                        name = num_batch_train * (batch - 1) + j
                        fileset = {'name': name,
                                   'input': "%04d-input.png" % name,
                                   'output': "%04d-output.png" % name,
                                   'label': "%04d-label.png" % name,
                                   'dif': "%04d-dif.png" % name}

                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['input']), input[j, :, :, :].squeeze(), cmap=cmap)
                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['output']), output[j, :, :, :].squeeze(), cmap=cmap)
                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['label']), label[j, :, :, :].squeeze(), cmap=cmap)
                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['dif']), dif[j, :, :, :].squeeze(), cmap=cmap)

                        append_index(dir_result_train, fileset)

            writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)

            ## validation phase
            with torch.no_grad():
                netG.eval()

                loss_G_val = []

                for batch, data in enumerate(loader_val, 1):
                    def should(freq):
                        return freq > 0 and (batch % freq == 0 or batch == num_batch_val)

                    # input = data['input'].to(device)
                    input = data['label'].to(device)
                    label = data['label'].to(device)
                    mask = data['mask'].to(device)

                    # forward netG
                    output = netG(input)

                    loss_G = fn_REG(output * (1 - mask), label * (1 - mask))

                    loss_G_val += [loss_G.item()]

                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
                          % (epoch, batch, num_batch_val, np.mean(loss_G_val)))

                    if should(num_freq_disp):
                        ## show output
                        input = transform_inv(input)
                        label = transform_inv(label)
                        output = transform_inv(output)

                        input = np.clip(input, 0, 1)
                        label = np.clip(label, 0, 1)
                        output = np.clip(output, 0, 1)
                        dif = np.clip(abs(label - input), 0, 1)

                        writer_val.add_images('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                        writer_val.add_images('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                        writer_val.add_images('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

                        for j in range(label.shape[0]):
                            # name = num_train * (epoch - 1) + num_batch_train * (batch - 1) + j
                            name = num_batch_train * (batch - 1) + j
                            fileset = {'name': name,
                                       'input': "%04d-input.png" % name,
                                       'output': "%04d-output.png" % name,
                                       'label': "%04d-label.png" % name,
                                       'dif': "%04d-dif.png" % name, }

                            plt.imsave(os.path.join(dir_result_val, 'images', fileset['input']), input[j, :, :, :].squeeze(), cmap=cmap)
                            plt.imsave(os.path.join(dir_result_val, 'images', fileset['output']), output[j, :, :, :].squeeze(), cmap=cmap)
                            plt.imsave(os.path.join(dir_result_val, 'images', fileset['label']), label[j, :, :, :].squeeze(), cmap=cmap)
                            plt.imsave(os.path.join(dir_result_val, 'images', fileset['dif']), dif[j, :, :, :].squeeze(), cmap=cmap)

                            append_index(dir_result_val, fileset)

                writer_val.add_scalar('loss_G', np.mean(loss_G_val), epoch)

            # update schduler
            # schedG.step()
            # schedD.step()

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, netG, optimG, epoch)

        writer_train.close()
        writer_val.close()

    def test(self):
        mode = self.mode

        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        size_data = (self.ny_in, self.nx_in, self.nch_in)
        size_window = (5, 5)


        norm = self.norm

        name_data = self.name_data

        if nch_out == 1:
            cmap = 'gray'
        else:
            cmap = None

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result_test = os.path.join(self.dir_result, self.scope, name_data, 'test')
        if not os.path.exists(os.path.join(dir_result_test, 'images')):
            os.makedirs(os.path.join(dir_result_test, 'images'))

        dir_data_test = os.path.join(self.dir_data, name_data, 'test')

        transform_test = transforms.Compose([Normalize(mean=0.5, std=0.5), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denormalize(mean=0.5, std=0.5)])
        transform_ts2np = ToNumpy()

        # dataset_test = Dataset(dir_data_test, data_type=self.data_type, transform=transform_test, sgm=(0, 25))
        dataset_test = Dataset(dir_data_test, data_type=self.data_type, transform=transform_test, sgm=25, ratio=1, size_data=size_data, size_window=size_window)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_test = len(dataset_test)

        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        # netG = UNet(nch_in, nch_out, nch_ker, norm)
        netG = ResNet(nch_in, nch_out, nch_ker, norm)
        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        fn_REG = nn.L1Loss().to(device)  # L1
        # fn_REG = nn.MSELoss().to(device)  # L1

        ## load from checkpoints
        st_epoch = 0

        netG, st_epoch = self.load(dir_chck, netG, mode=mode)

        ## test phase
        with torch.no_grad():
            netG.eval()
            # netG.train()

            loss_G_test = []

            for i, data in enumerate(loader_test, 1):
                # input = data['input'].to(device)
                input = data['label'].to(device)
                label = data['label'].to(device)
                mask = data['mask'].to(device)

                output = netG(input)

                loss_G = fn_REG(output * (1 - mask), label * (1 - mask))

                loss_G_test += [loss_G.item()]

                input = transform_inv(input)
                label = transform_inv(label)
                output = transform_inv(output)

                input = np.clip(input, 0, 1)
                label = np.clip(label, 0, 1)
                output = np.clip(output, 0, 1)
                dif = np.clip(abs(label - input), 0, 1)

                for j in range(label.shape[0]):
                    name = batch_size * (i - 1) + j
                    fileset = {'name': name,
                               'input': "%04d-input.png" % name,
                               'output': "%04d-output.png" % name,
                               'label': "%04d-label.png" % name,
                               'dif': "%04d-dif.png" % name,}

                    plt.imsave(os.path.join(dir_result_test, 'images', fileset['input']), input[j, :, :, :].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_test, 'images', fileset['output']), output[j, :, :, :].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_test, 'images', fileset['label']), label[j, :, :, :].squeeze(), cmap=cmap)
                    plt.imsave(os.path.join(dir_result_test, 'images', fileset['dif']), dif[j, :, :, :].squeeze(), cmap=cmap)

                    append_index(dir_result_test, fileset)

                print('TEST: %d/%d: LOSS: %.6f' % (i, num_batch_test, loss_G.item()))
            print('TEST: AVERAGE LOSS: %.6f' % (np.mean(loss_G_test)))


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)

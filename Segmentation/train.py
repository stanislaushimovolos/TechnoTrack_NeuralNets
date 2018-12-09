import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import os
from tqdm import *
import numpy as np
from torch import nn
import torch

import torch.utils.data as dt
import carvana_dataset as cv


def train_net(net, ds, ds_test, net_lr=0.01, useCuda=True, n_epoch=100, batch_sz=32):
    
    if os.path.exists('./log/') == False:
        os.mkdir('./log/')
    tb_writer = SummaryWriter(log_dir='log')
    
    m = net
    criterion =  torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(m.parameters(), net_lr)

    if useCuda == True:
        m = m.cuda()
        criterion= criterion.cuda()

    dl      = dt.DataLoader(ds, shuffle=True, num_workers=4, batch_size=batch_sz)
    dl_test = dt.DataLoader(ds_test, shuffle=False, num_workers=4, batch_size=batch_sz)

    global_iter = 0
    for epoch in range(0, n_epoch):
        print ("Current epoch: ", epoch)
        epoch_loss = 0
        m.train(True)
        for iter, (i, t) in enumerate(tqdm( dl) ):
            i = Variable(i)
            t = Variable(t)
            if useCuda :
                i = i.cuda()
                t = t.cuda()
            o = m(i)
            loss = criterion(o, t)
            loss.backward()
            optimizer.step()

            global_iter += 1
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / float(len(ds))
        print ("Epoch loss", epoch_loss)
        tb_writer.add_scalar('Loss/Train', epoch_loss, epoch)

        print ("Make test")
        test_loss = 0
        m.train(False)

        tb_out = np.random.choice(range(0, len(dl_test)), 3 )
        for iter, (i, t) in enumerate(tqdm(dl_test)):
            with torch.no_grad():
                i = Variable(i)
                t = Variable(t)
            if useCuda :
                i = i.cuda()
                t = t.cuda()
            o = m(i)
            loss = criterion(o, t)
            test_loss += loss.item()

            for k, c in enumerate(tb_out):
                if c == iter:
                    tb_writer.add_image('Image/Test_input_%d'%k,  i[0].cpu(), epoch)  # Tensor
                    tb_writer.add_image('Image/Test_target_%d'%k, t[0].cpu(), epoch)  # Tensor
                    tb_writer.add_image('Image/Test_output_%d'%k, o[0].cpu(), epoch)  # Tensor

        test_loss = test_loss / float(len(ds_test))
        print ("Test loss", test_loss)
        tb_writer.add_scalar('Loss/Test', test_loss, epoch)

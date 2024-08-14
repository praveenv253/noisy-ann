#!/usr/bin/env python3

import os
import models
from functools import wraps


class Params:

    def __init__(self, args):
        self.batch_size = 64
        self.num_epochs = 10
        self.adam_lr = 0.001

        #self.net_name = 'mnist-6l-cnn'
        #self.net_name = 'mnist-6l-cnn-v2'
        #self.net_name = 'mnist-5l-cnn-v3'
        self.net_name = 'mnist-5l-cnn-v3.1'

        if self.net_name == 'mnist-6l-cnn':
            self.Net = models.Mnist_6L_CNN
            self.NoisyNet = models.Noisy_Mnist_6L_CNN
        elif self.net_name == 'mnist-6l-cnn-v2':
            self.Net = models.Mnist_6L_CNN_v2
            self.NoisyNet = models.Noisy_Mnist_6L_CNN_v2
        elif self.net_name == 'mnist-5l-cnn-v3':
            self.Net = models.Mnist_5L_CNN_v3
            self.NoisyNet = models.Noisy_Mnist_5L_CNN_v3
        elif self.net_name == 'mnist-5l-cnn-v3.1':
            self.Net = models.Mnist_5L_CNN_v3_1
            self.NoisyNet = models.Noisy_Mnist_5L_CNN_v3_1

        #self.activn = 'relu'
        self.activn = 'tanh'

        self.args = args
        self.savedir = '../saved'

    def _pathify(func):
        @wraps(func)
        def convert_to_path(self):
            return os.path.join(self.savedir, func(self))
        return convert_to_path

    @_pathify
    def model_filename(self):
        filename = self.net_name
        filename += '--%s' % self.activn
        args = self.args
        if args.noisy:
            filename += '--noisy'
            if args.noisy is not True:
                filename += '-' + args.noisy
            if args.noisy != 'zero':
                filename += '--covrot-%d' % args.covrot
        if not args.noisy or args.noisy == 'zero':
            filename += ('--vert' if args.rotate is None
                         else '--rot-%d' % args.rotate)
        if hasattr(args, 'iter') and args.iter:
            filename += '--%d' % args.iter
        return filename + '.pth'

    @_pathify
    def vert_model_filename(self):
        filename = '%s--%s--vert' % (self.net_name, self.activn)
        if hasattr(self.args, 'iter') and self.args.iter:
            filename += '--%d' % self.args.iter
        return filename + '.pth'

    @_pathify
    def cov_filename(self):
        filename = 'cov--%s--%s--covrot-%d' % (self.net_name, self.activn,
                                               self.args.covrot)
        if hasattr(self.args, 'iter') and self.args.iter:
            filename += '--%d' % self.args.iter
        return filename + '.npy'

    @_pathify
    def perf_filename(self):
        model_filename = os.path.basename(self.model_filename())
        return 'perf--' + model_filename.replace('.pth', '.pkl')

    @_pathify
    def alignment_filename(self):
        filename = ('covariance-alignment--%s--%s--%d'
                    % (self.net_name, self.activn, self.args.covrot))
        if hasattr(self.args, 'iter') and self.args.iter:
            filename += '--%d' % self.args.iter
        return filename + '.csv'

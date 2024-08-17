#!/usr/bin/env python3

import os
import models
from functools import wraps
import argparse


class Params:

    def __init__(self, args_needed, args_list=None):
        self.batch_size = 64
        self.num_epochs = 10
        self.adam_lr = 0.001

        self.net_name = 'mnist-6l-cnn'
        #self.net_name = 'mnist-6l-cnn-v2'
        #self.net_name = 'mnist-5l-cnn-v3'
        #self.net_name = 'mnist-5l-cnn-v3.1'

        if self.net_name == 'mnist-6l-cnn':
            self.Net = models.Mnist_6L_CNN
            #self.NoisyNet = models.Noisy_Mnist_6L_CNN
            self.default_noisy_layer = 3  # (3rd layer *index*)
        elif self.net_name == 'mnist-6l-cnn-v2':
            self.Net = models.Mnist_6L_CNN_v2
            self.NoisyNet = models.Noisy_Mnist_6L_CNN_v2
            self.default_noisy_layer = 4
        elif self.net_name == 'mnist-5l-cnn-v3':
            self.Net = models.Mnist_5L_CNN_v3
            self.NoisyNet = models.Noisy_Mnist_5L_CNN_v3
            self.default_noisy_layer = 4
        elif self.net_name == 'mnist-5l-cnn-v3.1':
            self.Net = models.Mnist_5L_CNN_v3_1
            self.NoisyNet = models.Noisy_Mnist_5L_CNN_v3_1
            self.default_noisy_layer = 4

        #self.activn = 'relu'
        self.activn = 'tanh'

        self.savedir = '../revamped'

        parser = argparse.ArgumentParser()
        if 'rotate' in args_needed:
            parser.add_argument('--rotate', type=int, default=None,
                                help='Rotation angle in degrees to apply to training data')
        if 'noisy' in args_needed:
            parser.add_argument('--noisy', nargs='?', const=True, default=False,
                help=('Instantiate the noiseless model if false. If true, instantiate '
                      'the noisy model and load the covariance matrix computed using the '
                      'rotation given by `covrot`. If zero, use the noisy model but add '
                      'no noise (i.e., to train only post-noise layers). If diagonal, use '
                      'only the diagonal of the covariance matrix. If identity, use an '
                      'identity covariance matrix.'))
        if 'covrot' in args_needed:
            parser.add_argument('--covrot', type=int, default=60,
                                help=('Rotation angle used to compute the covariance matrix '
                                      'for adding noise while training.'))
        if 'iter' in args_needed:
            parser.add_argument('--iter', type=int, default=0,
                                help='Iteration number for multiple runs')

        # If args_list is None (the default), this reads from sys.argv
        self.args = parser.parse_args(args=args_list)


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

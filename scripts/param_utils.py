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

        self.num_train = 60000  # XXX Number of training data points - hard-coded for now XXX

        self.savedir = '../saved'

        parser = argparse.ArgumentParser()
        parser.add_argument('--arch', default='v3',
                            help='Neural network architecture code (see models.py)')
        parser.add_argument('--activn', default='tanh', choices=['relu', 'tanh'],
                            help='Nonlinearity used in the ANN activation function')
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
            parser.add_argument('--reinit', action='store_true')
        parser.add_argument('--noisy-layer', type=int, default=None,
                            help='Layer of network at which to add noise')
        if 'covrot' in args_needed:
            parser.add_argument('--covrot', type=int, default=60,
                                help=('Rotation angle used to compute the covariance matrix '
                                      'for adding noise while training.'))
        if 'iter' in args_needed:
            parser.add_argument('--iter', type=int, default=0,
                                help='Iteration number for multiple runs')
        parser.add_argument('--loo', type=int, default=None,
                            help=('Leave out one digit from the dataset. To be '
                                  'used while computing covariance.'))
        parser.add_argument('--cov-loo', type=int, default=None,
                            help=('Use the covariance matrix computed while '
                                  'leaving out this digit. Autoset by --loo.'))
        if 'confusion' in args_needed:
            parser.add_argument('--confusion', action='store_true',
                                help=('Compute a confusion matrix. To be used '
                                      'while testing.'))

        # If args_list is None (the default), this reads from sys.argv
        self.args = parser.parse_args(args=args_list)

        if self.args.arch == 'v1':
            self.Net = models.Mnist_v1_1C5F
            self.default_noisy_layer = 3  # (3rd layer *index*)
        elif self.args.arch == 'v2':
            self.Net = models.Mnist_v2_3C3F
            self.default_noisy_layer = 3
        elif self.args.arch == 'v3':
            self.Net = models.Mnist_v3_2C3F
            self.default_noisy_layer = 1

        self.activn = self.args.activn

        self.noisy_layer = (self.args.noisy_layer if self.args.noisy_layer is not None
                            else self.default_noisy_layer)

        self.net_name = self.Net.__name__

        if self.args.loo is not None:
            self.args.cov_loo = self.args.loo


    def _pathify(func):
        @wraps(func)
        def convert_to_path(self):
            return os.path.join(self.savedir, func(self))
        return convert_to_path

    @property
    def net_name_nl(self):
        """Net name combined with noisy layer."""
        return self.net_name + '_N%d' % self.noisy_layer

    @_pathify
    def model_filename(self):
        args = self.args
        if (not args.noisy or args.noisy == 'zero') and args.rotate is None:
            # We are in vert mode
            filename = self.net_name
        else:
            filename = self.net_name_nl
        filename += '--%s' % self.activn
        if args.noisy:
            filename += '--noisy'
            if args.noisy is not True:
                filename += '-' + args.noisy
            if args.noisy != 'zero':
                filename += '--covrot-%d' % args.covrot
            if args.cov_loo is not None:
                filename += '--loo-%d' % args.cov_loo
        if hasattr(args, 'reinit') and args.reinit:
            filename += '--reinit'
        if not args.noisy or args.noisy == 'zero':
            filename += ('--vert' if args.rotate is None
                         else '--rot-%d' % args.rotate)
        if hasattr(args, 'iter') and args.iter is not None:
            filename += '--%d' % args.iter
        return filename + '.pth'

    @_pathify
    def vert_model_filename(self):
        # Vert model does not need noisy layer appended to net name
        filename = '%s--%s--vert' % (self.net_name, self.activn)
        if hasattr(self.args, 'iter') and self.args.iter is not None:
            filename += '--%d' % self.args.iter
        return filename + '.pth'

    @_pathify
    def cov_filename(self):
        filename = 'cov--%s--%s--covrot-%d' % (self.net_name_nl, self.activn,
                                               self.args.covrot)
        if self.args.cov_loo is not None:
            filename += '--loo-%d' % self.args.cov_loo
        if hasattr(self.args, 'iter') and self.args.iter is not None:
            filename += '--%d' % self.args.iter
        return filename + '.npy'

    @_pathify
    def perf_filename(self):
        model_filename = os.path.basename(self.model_filename())
        return 'perf--' + model_filename.replace('.pth', '.pkl')

    @_pathify
    def alignment_filename(self):
        filename = ('covariance-alignment--%s--%s--%d'
                    % (self.net_name_nl, self.activn, self.args.covrot))
        if hasattr(self.args, 'iter') and self.args.iter is not None:
            filename += '--%d' % self.args.iter
        return filename + '.csv'

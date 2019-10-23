import sys
import time
import pdb
import ipdb
import collections
import random

import h5py
import numpy as np
import tensorflow as tf
import scipy.stats as stats
import sklearn.metrics as metrics








#以下代码用于准备变量FLAGS = tf.app.flags.FLAGS的基本内容格式
# parameters and data
tf.flags.DEFINE_string('params', '', 'File containing parameter config')
tf.flags.DEFINE_string('data', '', 'hd5 data file')
tf.flags.DEFINE_string('train_data', '', 'train tfrecord file')
tf.flags.DEFINE_string('test_data', '', 'test tfrecord file')

# ensembling/augmentation
tf.flags.DEFINE_boolean('augment_rc', False, 'Augment training with reverse complement.')
tf.flags.DEFINE_boolean('ensemble_rc', False, 'Ensemble prediction with reverse complement.')
tf.flags.DEFINE_string('augment_shifts', '0', 'Augment training with shifted sequences.')
tf.flags.DEFINE_string('ensemble_shifts', '0', 'Ensemble prediction with shifted sequences.')
tf.flags.DEFINE_integer('ensemble_mc', 0, 'Ensemble monte carlo samples.')

# logging
tf.flags.DEFINE_boolean('check_all', False, 'Checkpoint every epoch')
tf.flags.DEFINE_string('logdir', '/tmp/zrl',
                       'directory to keep checkpoints and summaries in')
tf.flags.DEFINE_boolean('log_device_placement', False,
                        'Log device placement (ie, CPU or GPU)')
tf.flags.DEFINE_integer('seed', 1, 'Random seed')

# step counts
tf.flags.DEFINE_integer('train_epochs', None,
                        'Number of training epochs.')
tf.flags.DEFINE_integer('train_epoch_batches', None,
                        'Number of batches per training epoch.')
tf.flags.DEFINE_integer('test_epoch_batches', None,
                        'Number of batches per test epoch.')

# training modes
tf.flags.DEFINE_boolean('no_steps', False, 'Update ops but no step ops')
tf.flags.DEFINE_string('restart', None, 'Restart training the model')
tf.flags.DEFINE_integer('early_stop', 25, 'Stop training if validation loss stagnates.')

FLAGS = tf.app.flags.FLAGS





















#accuracy.py
class Accuracy:

  def __init__(self,
               targets,
               preds,
               targets_na=None,
               loss=None,
               target_losses=None):
    self.targets = targets
    self.preds = preds
    self.targets_na = targets_na
    self.loss = loss
    self.target_losses = target_losses

    self.num_targets = self.targets.shape[-1]

  def r2(self, log=False, pseudocount=1, clip=None):
    """ Compute target R2 vector. """
    r2_vec = np.zeros(self.num_targets)

    for ti in range(self.num_targets):
      if self.targets_na is not None:
        preds_ti = self.preds[~self.targets_na, ti].astype('float64')
        targets_ti = self.targets[~self.targets_na, ti].astype('float64')
      else:
        preds_ti = self.preds[:, :, ti].flatten().astype('float64')
        targets_ti = self.targets[:, :, ti].flatten().astype('float64')

      if clip is not None:
        preds_ti = np.clip(preds_ti, 0, clip)
        targets_ti = np.clip(targets_ti, 0, clip)

      if log:
        preds_ti = np.log2(preds_ti + pseudocount)
        targets_ti = np.log2(targets_ti + pseudocount)

      r2_vec[ti] = metrics.r2_score(targets_ti, preds_ti)

    return r2_vec





























#ops.py
def reverse_complement_transform(data_ops):
  """Reverse complement of batched onehot seq and corresponding label and na."""

  # initialize reverse complemented data_ops
  data_ops_rc = {}

  # extract sequence from dict
  seq = data_ops['sequence']

  # check rank
  rank = seq.shape.ndims
  if rank != 3:
    raise ValueError("input seq must be rank 3.")

  # reverse complement sequence
  seq_rc = tf.gather(seq, [3, 2, 1, 0], axis=-1)
  seq_rc = tf.reverse(seq_rc, axis=[1])
  data_ops_rc['sequence'] = seq_rc

  # reverse labels
  if 'label' in data_ops:
    data_ops_rc['label'] = tf.reverse(data_ops['label'], axis=[1])

  # reverse NA
  if 'na' in data_ops:
    data_ops_rc['na'] = tf.reverse(data_ops['na'], axis=[1])

  return data_ops_rc














#layer.py
#此函数被sequence类用到，我叫他conv_block函数小集合
def conv_block(seqs_repr, conv_params, is_training,
               batch_norm, batch_norm_momentum,
               batch_renorm, batch_renorm_momentum,
               nonlinearity, l2_scale, layer_reprs, name=''):
  """Construct a single (dilated) CNN block.

  Args:
    seqs_repr:    [batchsize, length, num_channels] input sequence
    conv_params:  convolution parameters
    is_training:  whether is a training graph or not
    batch_norm:   whether to use batchnorm
    bn_momentum:  batch norm momentum
    batch_renorm: whether to use batch renormalization in batchnorm
    nonlinearity: relu/gelu/etc
    l2_scale:     L2 weight regularization scale
    name:         optional name for the block

  Returns:
    updated representation for the sequence
  """
  # nonlinearity
  if nonlinearity == 'relu':
      seqs_repr_next = tf.nn.relu(seqs_repr)
      tf.logging.info('ReLU')
  elif nonlinearity == 'gelu':
      seqs_repr_next = tf.nn.sigmoid(1.702 * seqs_repr) * seqs_repr
      tf.logging.info('GELU')
  else:
      print('Unrecognized nonlinearity "%s"' % nonlinearity, file=sys.stderr)
      exit(1)

  # Convolution
  seqs_repr_next = tf.layers.conv1d(
      seqs_repr_next,
      filters=conv_params.filters,
      kernel_size=[conv_params.filter_size],
      strides=conv_params.stride,
      padding='same',
      dilation_rate=[conv_params.dilation],
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in'),
      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
  tf.logging.info('Convolution w/ %d %dx%d filters strided %d, dilated %d' %
                  (conv_params.filters, seqs_repr.shape[2],
                   conv_params.filter_size, conv_params.stride,
                   conv_params.dilation))

  # Batch norm
  if batch_norm:
    if conv_params.skip_layers > 0:
      gamma_init = tf.zeros_initializer()
    else:
      gamma_init = tf.ones_initializer()

    seqs_repr_next = tf.layers.batch_normalization(
        seqs_repr_next,
        momentum=batch_norm_momentum,
        training=is_training,
        gamma_initializer=gamma_init,
        renorm=batch_renorm,
        renorm_clipping={'rmin': 1./4, 'rmax':4., 'dmax':6.},
        renorm_momentum=batch_renorm_momentum,
        fused=True)
    tf.logging.info('Batch normalization')

  # Dropout
  if conv_params.dropout > 0:
    seqs_repr_next = tf.layers.dropout(
        inputs=seqs_repr_next,
        rate=conv_params.dropout,
        training=is_training)
    tf.logging.info('Dropout w/ probability %.3f' % conv_params.dropout)

  # Skip
  if conv_params.skip_layers > 0:
    if conv_params.skip_layers > len(layer_reprs):
      raise ValueError('Skip connection reaches back too far.')

    # Add
    seqs_repr_next += layer_reprs[-conv_params.skip_layers]

  # Dense
  elif conv_params.dense:
    seqs_repr_next = tf.concat(values=[seqs_repr, seqs_repr_next], axis=2)

  # Pool
  if conv_params.pool > 1:
    seqs_repr_next = tf.layers.max_pooling1d(
        inputs=seqs_repr_next,
        pool_size=conv_params.pool,
        strides=conv_params.pool,
        padding='same')
    tf.logging.info('Max pool %d' % conv_params.pool)

  return seqs_repr_next













#augmentation.py
#以下函数被sequence类用到，我叫他们augmentation函数集合
def augment_stochastic_rc(data_ops):
  """Apply a stochastic reverse complement augmentation.

  Args:
    data_ops: dict with keys 'sequence,' 'label,' and 'na.'
  Returns
    data_ops_aug: augmented data
  """
  reverse_preds = tf.random_uniform(shape=[]) > 0.5
  data_ops_aug = tf.cond(reverse_preds, lambda: reverse_complement_transform(data_ops),
                                        lambda: data_ops.copy())
  data_ops_aug['reverse_preds'] = reverse_preds
  return data_ops_aug


def augment_stochastic_shifts(seq, augment_shifts):
  """Apply a stochastic shift augmentation.

  Args:
    seq: input sequence of size [batch_size, length, depth]
    augment_shifts: list of int offsets to sample from
  Returns:
    shifted and padded sequence of size [batch_size, length, depth]
  """
  shift_index = tf.random_uniform(shape=[], minval=0,
      maxval=len(augment_shifts), dtype=tf.int64)
  shift_value = tf.gather(tf.constant(augment_shifts), shift_index)

  seq = tf.cond(tf.not_equal(shift_value, 0),
                lambda: shift_sequence(seq, shift_value),
                lambda: seq)

  return seq


def augment_stochastic(data_ops, augment_rc=False, augment_shifts=[]):
  """Apply stochastic augmentations,

  Args:
    data_ops: dict with keys 'sequence,' 'label,' and 'na.'
    augment_rc: Boolean for whether to apply reverse complement augmentation.
    augment_shifts: list of int offsets to sample shift augmentations.
  Returns:
    data_ops_aug: augmented data
  """
  if augment_shifts:
    data_ops['sequence'] = augment_stochastic_shifts(data_ops['sequence'],
                                                     augment_shifts)

  if augment_rc:
    data_ops = augment_stochastic_rc(data_ops)
  else:
    data_ops['reverse_preds'] = tf.zeros((), dtype=tf.bool)

  return data_ops

def shift_sequence(seq, shift_amount, pad_value=0.25):
  """Shift a sequence left or right by shift_amount.

  Args:
    seq: a [batch_size, sequence_length, sequence_depth] sequence to shift
    shift_amount: the signed amount to shift (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
  """
  if seq.shape.ndims != 3:
    raise ValueError('input sequence should be rank 3')
  input_shape = seq.shape

  pad = pad_value * tf.ones_like(seq[:, 0:tf.abs(shift_amount), :])

  def _shift_right(_seq):
    sliced_seq = _seq[:, :-shift_amount:, :]
    return tf.concat([pad, sliced_seq], axis=1)

  def _shift_left(_seq):
    sliced_seq = _seq[:, -shift_amount:, :]
    return tf.concat([sliced_seq, pad], axis=1)

  output = tf.cond(
      tf.greater(shift_amount, 0), lambda: _shift_right(seq),
      lambda: _shift_left(seq))

  output.set_shape(input_shape)
  return output

def reverse_complement_transform(data_ops):
  """Reverse complement of batched onehot seq and corresponding label and na."""

  # initialize reverse complemented data_ops
  data_ops_rc = {}

  # extract sequence from dict
  seq = data_ops['sequence']

  # check rank
  rank = seq.shape.ndims
  if rank != 3:
    raise ValueError("input seq must be rank 3.")

  # reverse complement sequence
  seq_rc = tf.gather(seq, [3, 2, 1, 0], axis=-1)
  seq_rc = tf.reverse(seq_rc, axis=[1])
  data_ops_rc['sequence'] = seq_rc

  # reverse labels
  if 'label' in data_ops:
    data_ops_rc['label'] = tf.reverse(data_ops['label'], axis=[1])

  # reverse NA
  if 'na' in data_ops:
    data_ops_rc['na'] = tf.reverse(data_ops['na'], axis=[1])

  return data_ops_rc

def augment_deterministic_set(data_ops, augment_rc=False, augment_shifts=[0]):
  """

  Args:
    data_ops: dict with keys 'sequence,' 'label,' and 'na.'
    augment_rc: Boolean
    augment_shifts: List of ints.
  Returns
    data_ops_list:
  """
  augment_pairs = []
  for ashift in augment_shifts:
    augment_pairs.append((False, ashift))
    if augment_rc:
      augment_pairs.append((True, ashift))

  data_ops_list = []
  for arc, ashift in augment_pairs:
    data_ops_aug = augment_deterministic(data_ops, arc, ashift)
    data_ops_list.append(data_ops_aug)

  return data_ops_list

def augment_deterministic(data_ops, augment_rc=False, augment_shift=0):
  """Apply a deterministic augmentation, specified by the parameters.

  Args:
    data_ops: dict with keys 'sequence,' 'label,' and 'na.'
    augment_rc: Boolean
    augment_shift: Int
  Returns
    data_ops: augmented data, with all existing keys transformed
              and 'reverse_preds' bool added.
  """

  data_ops_aug = {}
  if 'label' in data_ops:
    data_ops_aug['label'] = data_ops['label']
  if 'na' in data_ops:
    data_ops_aug['na'] = data_ops['na']

  if augment_shift == 0:
    data_ops_aug['sequence'] = data_ops['sequence']
  else:
    shift_amount = tf.constant(augment_shift, shape=(), dtype=tf.int64)
    data_ops_aug['sequence'] = shift_sequence(data_ops['sequence'], shift_amount)

  if augment_rc:
    data_ops_aug = augment_deterministic_rc(data_ops_aug)
  else:
    data_ops_aug['reverse_preds'] = tf.zeros((), dtype=tf.bool)

  return data_ops_aug

def augment_deterministic_rc(data_ops):
  """Apply a deterministic reverse complement augmentation.

  Args:
    data_ops: dict with keys 'sequence,' 'label,' and 'na.'
  Returns
    data_ops_aug: augmented data ops
  """
  data_ops_aug = reverse_complement_transform(data_ops)
  data_ops_aug['reverse_preds'] = tf.ones((), dtype=tf.bool)
  return data_ops_aug




















#params.py
#以下小函数被params大流程用到
def layer_extend(var, default, layers):
  """Process job input to extend for the proper number of layers."""

  # if it's a number
  if not isinstance(var, list):
    # change the default to that number
    default = var

    # make it a list
    var = [var]

  # extend for each layer
  while len(var) < layers:
    var.append(default)

  return var

def add_cnn_params(params):
  """Define CNN params list."""
  if params.architecture in ['dres', 'dilated_residual']:
    add_cnn_params_dres(params)
  else:
    add_cnn_params_cnn(params)
    
def add_cnn_params_cnn(params):
  """Layer-by-layer CNN parameter mode."""

  params.cnn_params = []
  for ci in range(params.cnn_layers):
    cp = ConvParams(
        filters=params.cnn_filters[ci],
        filter_size=params.cnn_filter_sizes[ci],
        stride=params.cnn_stride[ci],
        pool=params.cnn_pool[ci],
        dropout=params.cnn_dropout[ci],
        dense=params.cnn_dense[ci],
        skip_layers=params.cnn_skip[ci],
        dilation=params.cnn_dilation[ci])
    params.cnn_params.append(cp)
    
class ConvParams(
    collections.namedtuple('ConvParams',
                           ['filters', 'filter_size', 'stride', 'pool',
                            'dilation', 'dropout', 'skip_layers', 'dense'])):
  """Convolution block parameters.

  Args:
    filters: number of convolution filters.
    filter_size: convolution filter size.
    stride: convolution stride.
    pool: max pool width.
    dilation: convolution dilation rate.
    dropout: dropout rate.
    skip_layers: add block result to preceding layer.
    dense: concat block result to preceding layer.
  """
  def __new__(cls, filters=1, filter_size=1, stride=1, pool=1,
              dilation=1, dropout=0, skip_layers=0, dense=False):
    return super(ConvParams, cls).__new__(cls, filters, filter_size,
                                          stride, pool, dilation,
                                          dropout, skip_layers, dense)
    
    
    
#params大流程    
#read_job_params被run函数调用，用于确定卷积神经网络的基础结构，主要就是为了处理接收到的params.txt
def read_job_params(job_file, require=[]):
  """Read job parameters from text table."""

  job = {}

  if job_file is not None:
    for line in open(job_file):
      if line.strip():
        param, val = line.split()

        # require a decimal for floats
        try:
          if val.find('e') != -1:
            val = float(val)
          elif val.find('.') == -1:
            val = int(val)
          else:
            val = float(val)
        except ValueError:
          pass

        if param in job:
          # change to a list
          if not isinstance(job[param], list):
            job[param] = [job[param]]

          # append new value
          job[param].append(val)
        else:
          job[param] = val

    print(job)

  for param in require:
    if param not in require:
      print('Must specify %s in params file' % param, file=sys.stderr)
      exit(1)

  return job


#下面的make_hparams函数被上面紧邻的函数read_job_params调用，用于初始化卷积神经网络格式
def make_hparams(job, num_worker_replicas=None, num_ps_replicas=None):
  """Convert the parsed job args to an params object.

  Args:
    job: a dictionary of parsed parameters.
      See `basenji.google.params.read_job_params` for more information.
    num_worker_replicas: the number of worker replicas, e.g.
      http://google3/learning/brain/contrib/learn/learn.borg?l=112&rcl=174372550
    num_ps_replicas: the number of ps replicas, e.g.
      http://google3/learning/brain/contrib/learn/learn.borg?l=113&rcl=174372550
  """

  hp = tf.contrib.training.HParams()

  ###################################################
  # data

  hp.add_hparam('seq_depth', job.get('seq_depth', 4))
  hp.add_hparam('num_targets', job['num_targets'])
  hp.add_hparam('target_classes', job.get('target_classes', 1))
  hp.add_hparam('target_pool', job.get('target_pool', 1))

  ###################################################
  # batching

  hp.add_hparam('batch_size', job.get('batch_size', 64))
  hp.add_hparam('seq_length', job.get('seq_length', 1024))
  hp.add_hparam('batch_buffer', job.get('batch_buffer', 64))

  hp.add_hparam('batch_norm', bool(job.get('batch_norm', True)))
  hp.add_hparam('batch_renorm', bool(job.get('batch_renorm', False)))
  hp.add_hparam('batch_norm_momentum', 0.9)
  hp.add_hparam('batch_renorm_momentum', 0.9)

  ###################################################
  # training

  optimizer = job.get('optimizer', 'nadam')
  optimizer = job.get('optimization', optimizer)
  hp.add_hparam('optimizer', optimizer.lower())

  hp.add_hparam('learning_rate', job.get('learning_rate', 0.001))
  hp.add_hparam('momentum', job.get('momentum', 0))

  hp.add_hparam('learning_decay_steps', job.get('learning_decay_steps', 200000))
  hp.add_hparam('learning_decay_rate', job.get('learning_decay_rate', 0.2))

  hp.add_hparam('adam_beta1', job.get('adam_beta1', 0.9))
  hp.add_hparam('adam_beta2', job.get('adam_beta2', 0.999))
  hp.add_hparam('adam_eps', job.get('adam_eps', 1e-8))

  hp.add_hparam('grad_clip', job.get('grad_clip', 1.0))

  hp.add_hparam('cnn_l2_scale', job.get('cnn_l2_scale', 0.))
  hp.add_hparam('final_l1_scale', job.get('final_l1_scale', 0.))

  hp.add_hparam('nonlinearity', job.get('nonlinearity', 'relu'))

  ###################################################
  # loss

  link = job.get('link', 'softplus')
  link = job.get('link_function', link)
  hp.add_hparam('link', link)

  loss = job.get('loss', 'poisson')
  loss = job.get('loss_name', loss)
  hp.add_hparam('loss', loss)

  hp.add_hparam('target_clip', job.get('target_clip', None))
  hp.add_hparam('target_sqrt', bool(job.get('target_sqrt', False)))

  ###################################################
  # architecture

  hp.add_hparam('architecture', job.get('architecture', 'cnn'))

  if hp.architecture in ['dres', 'dilated_residual']:
    add_hparams_dres(hp, job)
  else:
    add_hparams_cnn(hp, job)

  # transform CNN hparams to specific params
  add_cnn_params(hp)

  ###################################################
  # google3

  hp.add_hparam('augment_with_complement',
                job.get('augment_with_complement', False))
  hp.add_hparam('shift_augment_offsets', job.get('shift_augment_offsets', None))

  hp.add_hparam('ensemble_during_training',
                job.get('ensemble_during_training', False))
  hp.add_hparam('ensemble_during_prediction',
                job.get('ensemble_during_prediction', False))

  hp.add_hparam('num_plateau_steps', job.get('num_plateau_steps', 5000))

  hp.add_hparam('plateau_delta', job.get('plateau_delta', 0.05))

  hp.add_hparam('stop_early', job.get('stop_early', False))

  hp.add_hparam('stop_early_num_plateau_steps',
                job.get('stop_early_num_plateau_steps', 10000))

  hp.add_hparam('stop_early_plateau_delta',
                job.get('stop_early_plateau_delta', 0.03))

  # If True, collapse into a single per-sequence feature vector by mean pooling.
  hp.add_hparam('pool_by_averaging', job.get('pool_by_averaging', False))

  # If True, unfold into features of size length * channels.
  hp.add_hparam('pool_by_unfolding', job.get('pool_by_unfolding', False))

  if hp.pool_by_averaging and hp.pool_by_unfolding:
    raise ValueError('It is invalid to specify both pool_by_averaging'
                     ' and pool_by_unfolding')

  tf.logging.info('Parsed params from job argument, and got %s',
                  str(hp.values()))

  hp.add_hparam('num_worker_replicas', num_worker_replicas)
  hp.add_hparam('num_ps_replicas', num_ps_replicas)

  return hp

#下面的add_hparams_cnn函数被上面紧邻的make_hparams函数调用
def add_hparams_cnn(params, job):
  """Add CNN hyper-parameters for a standard verbose CNN definition."""

  # establish layer number using filters
  params.add_hparam('cnn_filters',
                    layer_extend(job.get('cnn_filters', []), 16, 1))
  layers = len(params.cnn_filters)
  params.cnn_layers = layers

  # get remainder, or set to default
  params.add_hparam('cnn_filter_sizes',
                    layer_extend(job.get('cnn_filter_sizes', []), 1, layers))
  params.add_hparam('cnn_stride',
                    layer_extend(job.get('cnn_stride', []), 1, layers))
  params.add_hparam('cnn_pool',
                    layer_extend(job.get('cnn_pool', []), 1, layers))
  params.add_hparam('cnn_dense',
                    layer_extend(job.get('cnn_dense', []), False, layers))
  params.add_hparam('cnn_dilation',
                    layer_extend(job.get('cnn_dilation', []), 1, layers))
  params.add_hparam('cnn_skip',
                    layer_extend(job.get('cnn_skip', []), 0, layers))
  params.add_hparam('cnn_dropout',
                    layer_extend(job.get('cnn_dropout', []), 0., layers))

  # g3 dropout parameterization
  if 'non_dilated_cnn_dropout' in job and 'dilated_cnn_dropout' in job:
    params.cnn_dropout = []
    for ci in range(layers):
      if params.cnn_dilation[ci] > 1:
        params.cnn_dropout.append(job['dilated_cnn_dropout'])
      else:
        params.cnn_dropout.append(job['non_dilated_cnn_dropout'])

















#dna_io.py
def hot1_augment(Xb, fwdrc, shift):
  """ Transform a batch of one hot coded sequences to augment training.

    Args:
      Xb:     Batch x Length x 4 array
      fwdrc:  Boolean representing forward versus reverse complement strand.
      shift:  Integer shift

    Returns:
      Xbt:    Transformed version of Xb
    """

  if Xb.dtype == bool:
    nval = 0
  else:
    nval = 1. / Xb.shape[2]

  if shift == 0:
    Xbt = Xb

  elif shift > 0:
    Xbt = np.zeros(Xb.shape)

    # fill in left unknowns
    Xbt[:, :shift, :] = nval

    # fill in sequence
    Xbt[:, shift:, :] = Xb[:, :-shift, :]
    # e.g.
    # Xbt[:,1:,] = Xb[:,:-1,:]

  elif shift < 0:
    Xbt = np.zeros(Xb.shape)

    # fill in right unknowns
    Xbt[:, shift:, :] = nval

    # fill in sequence
    Xbt[:, :shift, :] = Xb[:, -shift:, :]
    # e.g.
    # Xb_shift[:,:-1,:] = Xb[:,1:,:]

  if not fwdrc:
    Xbt = hot1_rc(Xbt)

  return Xbt






















#batcher.py
class Batcher:
  """ Batcher

    Class to manage batches.
    """

  def __init__(self,
               Xf,
               Yf=None,
               NAf=None,
               batch_size=64,
               pool_width=1,
               shuffle=False):
    self.Xf = Xf
    self.num_seqs = self.Xf.shape[0]
    self.seq_len = self.Xf.shape[1]
    self.seq_depth = self.Xf.shape[2]

    self.Yf = Yf
    if self.Yf is not None:
      self.num_targets = self.Yf.shape[2]

    self.NAf = NAf

    self.batch_size = batch_size
    self.pool_width = pool_width
    if self.seq_len % self.pool_width != 0:
      print(
          'Pool width %d does not evenly divide the sequence length %d' %
          (self.pool_width, self.seq_len),
          file=sys.stderr)
      exit(1)

    self.shuffle = shuffle

    self.reset()

  def empty(self):
    return self.start >= self.num_seqs

  def remaining(self):
    return self.num_seqs - self.start

  def next(self, fwdrc=True, shift=0):
    """ Load the next batch from the HDF5. """
    Xb = None
    Yb = None
    NAb = None
    Nb = 0

    stop = self.start + self.batch_size
    if self.start < self.num_seqs:
      # full or partial batch
      if stop <= self.num_seqs:
        Nb = self.batch_size
      else:
        Nb = self.num_seqs - self.start

      # initialize
      Xb = np.zeros(
          (Nb, self.seq_len, self.seq_depth), dtype='float32')
      if self.Yf is not None:
        if self.Yf.dtype == np.uint8:
          ytype = 'int32'
        else:
          ytype = 'float32'

        Yb = np.zeros(
            (Nb, self.seq_len // self.pool_width,
             self.num_targets),
            dtype=ytype)
        NAb = np.zeros(
            (Nb, self.seq_len // self.pool_width), dtype='bool')

      # copy data
      for i in range(Nb):
        si = self.order[self.start + i]
        Xb[i] = self.Xf[si]

        # fix N positions
        Xbi_n = (Xb[i].sum(axis=1) == 0)
        Xb[i] = Xb[i] + (1 / self.seq_depth) * Xbi_n.repeat(
            self.seq_depth).reshape(self.seq_len, self.seq_depth)

        if self.Yf is not None:
          Yb[i] = np.nan_to_num(self.Yf[si])

          if self.NAf is not None:
            NAb[i] = self.NAf[si]

    # reverse complement and shift
    if Xb is not None:
      Xb = hot1_augment(Xb, fwdrc, shift)
    if not fwdrc:
      if Yb is not None:
        Yb = Yb[:, ::-1, :]
      if NAb is not None:
        NAb = NAb[:, ::-1]

    # update start
    self.start = min(stop, self.num_seqs)

    return Xb, Yb, NAb, Nb

  def reset(self):
    self.start = 0
    self.order = list(range(self.num_seqs))
    if self.shuffle:
      random.shuffle(self.order)














































        
#在训练过程中class SeqNNModel不存在实例对象，而仅仅是作为class SeqNN的一个父类。在训练过程中，test_h5函数本质上可以归并到class SeqNN当中
#seqnn_util.py
class SeqNNModel(object):

  def test_h5(self, sess, batcher, test_batches=None):
    """ Compute model accuracy on a test set.

        Args:
          sess:         TensorFlow session
          batcher:      Batcher object to provide data
          test_batches: Number of test batches

        Returns:
          acc:          Accuracy object
        """
    # setup feed dict
    fd = self.set_mode('test')

    # initialize prediction and target arrays
    preds = []
    targets = []
    targets_na = []

    batch_losses = []
    batch_target_losses = []
    batch_sizes = []

    # get first batch
    batch_num = 0
    Xb, Yb, NAb, Nb = batcher.next()

    while Xb is not None and (test_batches is None or
                              batch_num < test_batches):
      # update feed dict
      fd[self.inputs_ph] = Xb
      fd[self.targets_ph] = Yb

      # make predictions
      run_ops = [self.targets_eval, self.preds_eval,
                 self.loss_eval, self.loss_eval_targets]
      run_returns = sess.run(run_ops, feed_dict=fd)
      targets_batch, preds_batch, loss_batch, target_losses_batch = run_returns

      # accumulate predictions and targets
      preds.append(preds_batch[:Nb,:,:].astype('float16'))
      targets.append(targets_batch[:Nb,:,:].astype('float16'))
      targets_na.append(np.zeros([Nb, self.preds_length], dtype='bool'))

      # accumulate loss
      batch_losses.append(loss_batch)
      batch_target_losses.append(target_losses_batch)
      batch_sizes.append(Nb)

      # next batch
      batch_num += 1
      Xb, Yb, NAb, Nb = batcher.next()

    # reset batcher
    batcher.reset()

    # construct arrays
    targets = np.concatenate(targets, axis=0)
    preds = np.concatenate(preds, axis=0)
    targets_na = np.concatenate(targets_na, axis=0)

    # mean across batches
    batch_losses = np.array(batch_losses, dtype='float64')
    batch_losses = np.average(batch_losses, weights=batch_sizes)
    batch_target_losses = np.array(batch_target_losses, dtype='float64')
    batch_target_losses = np.average(batch_target_losses, axis=0, weights=batch_sizes)

    # instantiate accuracy object
    acc = Accuracy(targets, preds, targets_na,
                            batch_losses, batch_target_losses)

    return acc















#seqnn.py
class SeqNN(SeqNNModel):

  def __init__(self):
    self.global_step = tf.train.get_or_create_global_step()
    self.hparams_set = False

  def build_feed(self, job, augment_rc=False, augment_shifts=[0],
                 ensemble_rc=False, ensemble_shifts=[0],
                 embed_penultimate=False, target_subset=None):
    """Build training ops that depend on placeholders."""

    self.hp = make_hparams(job)
    self.hparams_set = True
    data_ops = self.make_placeholders()

    self.build_from_data_ops(job, data_ops,
          augment_rc=augment_rc,
          augment_shifts=augment_shifts,
          ensemble_rc=ensemble_rc,
          ensemble_shifts=ensemble_shifts,
          embed_penultimate=embed_penultimate,
          target_subset=target_subset)

  def build_from_data_ops(self, job, data_ops,
                          augment_rc=False, augment_shifts=[0],
                          ensemble_rc=False, ensemble_shifts=[0],
                          embed_penultimate=False, target_subset=None):
    """Build training ops from input data ops."""
    if not self.hparams_set:
      self.hp = params.make_hparams(job)
      self.hparams_set = True

    # training conditional
    self.is_training = tf.placeholder(tf.bool, name='is_training')

    ##################################################
    # training

    # training data_ops w/ stochastic augmentation
    data_ops_train = augment_stochastic(
        data_ops, augment_rc, augment_shifts)

    # compute train representation
    self.preds_train = self.build_predict(data_ops_train['sequence'],
                                          None, embed_penultimate, target_subset,
                                          save_reprs=True)
    self.target_length = self.preds_train.shape[1].value

    # training losses
    if not embed_penultimate:
      loss_returns = self.build_loss(self.preds_train, data_ops_train['label'], target_subset)
      self.loss_train, self.loss_train_targets, self.targets_train = loss_returns

      # optimizer
      self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.build_optimizer(self.loss_train)

      # allegedly correct, but outperformed by skipping
      # with tf.control_dependencies(self.update_ops):
      #   self.build_optimizer(self.loss_train)


    ##################################################
    # eval

    # eval data ops w/ deterministic augmentation
    data_ops_eval = augment_deterministic_set(
        data_ops, ensemble_rc, ensemble_shifts)
    data_seq_eval = tf.stack([do['sequence'] for do in data_ops_eval])
    data_rev_eval = tf.stack([do['reverse_preds'] for do in data_ops_eval])

    # compute eval representation
    map_elems_eval = (data_seq_eval, data_rev_eval)
    build_rep = lambda do: self.build_predict(do[0], do[1], embed_penultimate, target_subset)
    self.preds_ensemble = tf.map_fn(build_rep, map_elems_eval, dtype=tf.float32, back_prop=False)
    self.preds_eval = tf.reduce_mean(self.preds_ensemble, axis=0)

    # eval loss
    if not embed_penultimate:
      loss_returns = self.build_loss(self.preds_eval, data_ops['label'], target_subset)
      self.loss_eval, self.loss_eval_targets, self.targets_eval = loss_returns

    # update # targets
    if target_subset is not None:
      self.hp.num_targets = len(target_subset)

    # helper variables
    self.preds_length = self.preds_train.shape[1]
    
  def make_placeholders(self):
    """Allocates placeholders to be used in place of input data ops."""
    # batches
    self.inputs_ph = tf.placeholder(
        tf.float32,
        shape=(None, self.hp.seq_length, self.hp.seq_depth),
        name='inputs')

    if self.hp.target_classes == 1:
      self.targets_ph = tf.placeholder(
          tf.float32,
          shape=(None, self.hp.seq_length // self.hp.target_pool,
                 self.hp.num_targets),
          name='targets')
    else:
      self.targets_ph = tf.placeholder(
          tf.int32,
          shape=(None, self.hp.seq_length // self.hp.target_pool,
                 self.hp.num_targets),
          name='targets')

    self.targets_na_ph = tf.placeholder(tf.bool,
        shape=(None, self.hp.seq_length // self.hp.target_pool),
        name='targets_na')

    data = {
        'sequence': self.inputs_ph,
        'label': self.targets_ph,
        'na': self.targets_na_ph
    }
    return data

  def build_predict(self, inputs, reverse_preds=None, embed_penultimate=False, target_subset=None, save_reprs=False):
    """Construct per-location real-valued predictions."""
    assert inputs is not None
    print('Targets pooled by %d to length %d' %
          (self.hp.target_pool, self.hp.seq_length // self.hp.target_pool))

    ###################################################
    # convolution layers
    ###################################################
    filter_weights = []
    layer_reprs = [inputs]

    seqs_repr = inputs
    for layer_index in range(self.hp.cnn_layers):
      with tf.variable_scope('cnn%d' % layer_index, reuse=tf.AUTO_REUSE):
        # convolution block
        args_for_block = self._make_conv_block_args(layer_index, layer_reprs)
        seqs_repr = conv_block(seqs_repr=seqs_repr, **args_for_block)

        # save representation
        layer_reprs.append(seqs_repr)

    if save_reprs:
      self.layer_reprs = layer_reprs

    # final nonlinearity
    if self.hp.nonlinearity == 'relu':
      seqs_repr = tf.nn.relu(seqs_repr)
    elif self.hp.nonlinearity == 'gelu':
      seqs_repr = tf.nn.sigmoid(1.702 * seqs_repr) * seqs_repr
    else:
      print('Unrecognized nonlinearity "%s"' % self.hp.nonlinearity, file=sys.stderr)
      exit(1)

    ###################################################
    # slice out side buffer
    ###################################################

    # update batch buffer to reflect pooling
    seq_length = seqs_repr.shape[1].value
    pool_preds = self.hp.seq_length // seq_length
    assert self.hp.batch_buffer % pool_preds == 0, (
        'batch_buffer %d not divisible'
        ' by the CNN pooling %d') % (self.hp.batch_buffer, pool_preds)
    batch_buffer_pool = self.hp.batch_buffer // pool_preds

    # slice out buffer
    seq_length = seqs_repr.shape[1]
    seqs_repr = seqs_repr[:, batch_buffer_pool:
                          seq_length - batch_buffer_pool, :]
    seq_length = seqs_repr.shape[1]

    ###################################################
    # final layer
    ###################################################
    if embed_penultimate:
      final_repr = seqs_repr
    else:
      with tf.variable_scope('final', reuse=tf.AUTO_REUSE):
        final_filters = self.hp.num_targets * self.hp.target_classes
        final_repr = tf.layers.dense(
            inputs=seqs_repr,
            units=final_filters,
            activation=None,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in'),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(self.hp.final_l1_scale))
        print('Convolution w/ %d %dx1 filters to final targets' %
            (final_filters, seqs_repr.shape[2]))

        if target_subset is not None:
          # get convolution parameters
          filters_full = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'final/dense/kernel')[0]
          bias_full = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'final/dense/bias')[0]

          # subset to specific targets
          filters_subset = tf.gather(filters_full, target_subset, axis=1)
          bias_subset = tf.gather(bias_full, target_subset, axis=0)

          # substitute a new limited convolution
          final_repr = tf.tensordot(seqs_repr, filters_subset, 1)
          final_repr = tf.nn.bias_add(final_repr, bias_subset)

        # expand length back out
        if self.hp.target_classes > 1:
          final_repr = tf.reshape(final_repr,
                                  (-1, seq_length, self.hp.num_targets,
                                   self.hp.target_classes))

    # transform for reverse complement
    if reverse_preds is not None:
      final_repr = tf.cond(reverse_preds,
                           lambda: tf.reverse(final_repr, axis=[1]),
                           lambda: final_repr)

    ###################################################
    # link function
    ###################################################
    if embed_penultimate:
      predictions = final_repr
    else:
      # work-around for specifying my own predictions
      # self.preds_adhoc = tf.placeholder(
      #     tf.float32, shape=final_repr.shape, name='preds-adhoc')

      # float 32 exponential clip max
      exp_max = 50

      # choose link
      if self.hp.link in ['identity', 'linear']:
        predictions = tf.identity(final_repr, name='preds')

      elif self.hp.link == 'relu':
        predictions = tf.relu(final_repr, name='preds')

      elif self.hp.link == 'exp':
        final_repr_clip = tf.clip_by_value(final_repr, -exp_max, exp_max)
        predictions = tf.exp(final_repr_clip, name='preds')

      elif self.hp.link == 'exp_linear':
        predictions = tf.where(
            final_repr > 0,
            final_repr + 1,
            tf.exp(tf.clip_by_value(final_repr, -exp_max, exp_max)),
            name='preds')

      elif self.hp.link == 'softplus':
        final_repr_clip = tf.clip_by_value(final_repr, -exp_max, 10000)
        predictions = tf.nn.softplus(final_repr_clip, name='preds')

      else:
        print('Unknown link function %s' % self.hp.link, file=sys.stderr)
        exit(1)

      # clip
      if self.hp.target_clip is not None:
        predictions = tf.clip_by_value(predictions, 0, self.hp.target_clip)

      # sqrt
      if self.hp.target_sqrt:
        predictions = tf.sqrt(predictions)

    return predictions
    
  def _make_conv_block_args(self, layer_index, layer_reprs):
    """Packages arguments to be used by layers.conv_block."""
    return {
        'conv_params': self.hp.cnn_params[layer_index],
        'is_training': self.is_training,
        'nonlinearity': self.hp.nonlinearity,
        'batch_norm': self.hp.batch_norm,
        'batch_norm_momentum': self.hp.batch_norm_momentum,
        'batch_renorm': self.hp.batch_renorm,
        'batch_renorm_momentum': self.hp.batch_renorm_momentum,
        'l2_scale': self.hp.cnn_l2_scale,
        'layer_reprs': layer_reprs,
        'name': 'conv-%d' % layer_index
    }   

  def build_loss(self, preds, targets, target_subset=None):
    """Convert per-location real-valued predictions to a loss."""

    # slice buffer
    tstart = self.hp.batch_buffer // self.hp.target_pool
    tend = (self.hp.seq_length - self.hp.batch_buffer) // self.hp.target_pool
    targets = tf.identity(targets[:, tstart:tend, :], name='targets_op')

    if target_subset is not None:
      targets = tf.gather(targets, target_subset, axis=2)

    # clip
    if self.hp.target_clip is not None:
      targets = tf.clip_by_value(targets, 0, self.hp.target_clip)

    # sqrt
    if self.hp.target_sqrt:
      targets = tf.sqrt(targets)

    loss_op = None

    # choose loss
    if self.hp.loss == 'gaussian':
      loss_op = tf.squared_difference(preds, targets)

    elif self.hp.loss == 'poisson':
      loss_op = tf.nn.log_poisson_loss(
          targets, tf.log(preds), compute_full_loss=True)

    elif self.hp.loss == 'cross_entropy':
      loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=(targets - 1), logits=preds)

    else:
      raise ValueError('Cannot identify loss function %s' % self.hp.loss)

    # reduce lossses by batch and position
    loss_op = tf.reduce_mean(loss_op, axis=[0, 1], name='target_loss')
    loss_op = tf.check_numerics(loss_op, 'Invalid loss', name='loss_check')
    target_losses = loss_op

    if target_subset is None:
      tf.summary.histogram('target_loss', loss_op)
      for ti in np.linspace(0, self.hp.num_targets - 1, 10).astype('int'):
        tf.summary.scalar('loss_t%d' % ti, loss_op[ti])

    # fully reduce
    loss_op = tf.reduce_mean(loss_op, name='loss')

    # add regularization terms
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_sum = tf.reduce_sum(reg_losses)
    tf.summary.scalar('regularizers', reg_sum)
    loss_op += reg_sum

    # track
    tf.summary.scalar('loss', loss_op)

    return loss_op, target_losses, targets

  def build_optimizer(self, loss_op):
    """Construct optimization op that minimizes loss_op."""

    # adaptive learning rate
    self.learning_rate_adapt = tf.train.exponential_decay(
        learning_rate=self.hp.learning_rate,
        global_step=self.global_step,
        decay_steps=self.hp.learning_decay_steps,
        decay_rate=self.hp.learning_decay_rate,
        staircase=True)
    tf.summary.scalar('learning_rate', self.learning_rate_adapt)

    if self.hp.optimizer == 'adam':
      self.opt = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate_adapt,
          beta1=self.hp.adam_beta1,
          beta2=self.hp.adam_beta2,
          epsilon=self.hp.adam_eps)

    elif self.hp.optimizer == 'nadam':
      self.opt = tf.contrib.opt.NadamOptimizer(
          learning_rate=self.learning_rate_adapt,
          beta1=self.hp.adam_beta1,
          beta2=self.hp.adam_beta2,
          epsilon=self.hp.adam_eps)

    elif self.hp.optimizer in ['sgd', 'momentum']:
      self.opt = tf.train.MomentumOptimizer(
          learning_rate=self.learning_rate_adapt,
          momentum=self.hp.momentum)
    else:
      print('Cannot recognize optimization algorithm %s' % self.hp.optimizer)
      exit(1)

    # compute gradients
    self.gvs = self.opt.compute_gradients(
        loss_op,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    # clip gradients
    if self.hp.grad_clip is not None:
      gradients, variables = zip(*self.gvs)
      gradients, _ = tf.clip_by_global_norm(gradients, self.hp.grad_clip)
      self.gvs = zip(gradients, variables)

    # apply gradients
    self.step_op = self.opt.apply_gradients(
        self.gvs, global_step=self.global_step)

    # summary
    self.merged_summary = tf.summary.merge_all()

  def train_epoch_h5(self,
                     sess,
                     batcher,
                     sum_writer=None,
                     epoch_batches=None,
                     no_steps=False):
    """Execute one training epoch using HDF5 data,
       and compute-graph augmentation"""

    # initialize training loss
    train_loss = []
    batch_sizes = []
    global_step = 0

    # setup feed dict
    fd = self.set_mode('train')

    # get first batch
    Xb, Yb, NAb, Nb = batcher.next()

    batch_num = 0
    while Xb is not None and (epoch_batches is None or batch_num < epoch_batches):
      # update feed dict
      fd[self.inputs_ph] = Xb
      fd[self.targets_ph] = Yb

      if no_steps:
        run_returns = sess.run([self.merged_summary, self.loss_train] + \
                                self.update_ops, feed_dict=fd)
        summary, loss_batch = run_returns[:2]
      else:
        run_ops = [self.merged_summary, self.loss_train, self.global_step, self.step_op]
        run_ops += self.update_ops
        summary, loss_batch, global_step = sess.run(run_ops, feed_dict=fd)[:3]

      # add summary
      if sum_writer is not None:
        sum_writer.add_summary(summary, global_step)

      # accumulate loss
      train_loss.append(loss_batch)
      batch_sizes.append(Nb)

      # next batch
      Xb, Yb, NAb, Nb = batcher.next()
      batch_num += 1

    # reset training batcher if epoch considered all of the data
    if epoch_batches is None:
      batcher.reset()

    avg_loss = np.average(train_loss, weights=batch_sizes)

    return avg_loss, global_step

  def set_mode(self, mode):
    """ Construct a feed dictionary to specify the model's mode. """
    fd = {}

    if mode in ['train', 'training']:
      fd[self.is_training] = True

    elif mode in ['test', 'testing', 'evaluate']:
      fd[self.is_training] = False

    elif mode in [
        'test_mc', 'testing_mc', 'evaluate_mc', 'mc_test', 'mc_testing',
        'mc_evaluate'
    ]:
      fd[self.is_training] = False

    else:
      print('Cannot recognize mode %s' % mode)
      exit(1)

    return fd

    
    
    
    




























    
    
    
    




#basenji_train_h5.py
def run(params_file, data_file, train_epochs, train_epoch_batches, test_epoch_batches):

  #######################################################
  # load data
  #######################################################
  data_open = h5py.File(data_file)

  train_seqs = data_open['train_in']
  train_targets = data_open['train_out']
  train_na = None
  if 'train_na' in data_open:
    train_na = data_open['train_na']

  valid_seqs = data_open['valid_in']
  valid_targets = data_open['valid_out']
  valid_na = None
  if 'valid_na' in data_open:
    valid_na = data_open['valid_na']

  #######################################################
  # model parameters and placeholders
  #######################################################
  job = read_job_params(params_file)

  job['seq_length'] = train_seqs.shape[1]
  job['seq_depth'] = train_seqs.shape[2]
  job['num_targets'] = train_targets.shape[2]
  job['target_pool'] = int(np.array(data_open.get('pool_width', 1)))

  augment_shifts = [int(shift) for shift in FLAGS.augment_shifts.split(',')]
  ensemble_shifts = [int(shift) for shift in FLAGS.ensemble_shifts.split(',')]

  t0 = time.time()
  model = SeqNN()
  model.build_feed(job, augment_rc=FLAGS.augment_rc, augment_shifts=augment_shifts,
     ensemble_rc=FLAGS.ensemble_rc, ensemble_shifts=ensemble_shifts)

  print('Model building time %f' % (time.time() - t0))

  # adjust for fourier
  job['fourier'] = 'train_out_imag' in data_open
  if job['fourier']:
    train_targets_imag = data_open['train_out_imag']
    valid_targets_imag = data_open['valid_out_imag']

  #######################################################
  # prepare batcher
  #######################################################
  if job['fourier']:
    batcher_train = batcher.BatcherF(
        train_seqs,
        train_targets,
        train_targets_imag,
        train_na,
        model.hp.batch_size,
        model.hp.target_pool,
        shuffle=True)
    batcher_valid = batcher.BatcherF(valid_seqs, valid_targets,
                                     valid_targets_imag, valid_na,
                                     model.batch_size, model.target_pool)
  else:
    batcher_train = Batcher(
        train_seqs,
        train_targets,
        train_na,
        model.hp.batch_size,
        model.hp.target_pool,
        shuffle=True)
    batcher_valid = Batcher(valid_seqs, valid_targets, valid_na,
                                    model.hp.batch_size, model.hp.target_pool)
  print('Batcher initialized')

  #######################################################
  # train
  #######################################################

  # checkpoints
  saver = tf.train.Saver()

  config = tf.ConfigProto()
  if FLAGS.log_device_placement:
    config.log_device_placement = True
  with tf.Session(config=config) as sess:
    t0 = time.time()

    # set seed
    tf.set_random_seed(FLAGS.seed)

    if FLAGS.logdir:
      train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
    else:
      train_writer = None

    if FLAGS.restart:
      # load variables into session
      saver.restore(sess, FLAGS.restart)
    else:
      # initialize variables
      print('Initializing...')
      sess.run(tf.global_variables_initializer())
      print('Initialization time %f' % (time.time() - t0))

    train_loss = None
    best_loss = None
    early_stop_i = 0

    epoch = 0
    while (train_epochs is None or epoch < train_epochs) and early_stop_i < FLAGS.early_stop:
      t0 = time.time()

      # train
      train_loss, steps = model.train_epoch_h5(sess, batcher_train,
                                               sum_writer=train_writer,
                                               epoch_batches=train_epoch_batches,
                                               no_steps=FLAGS.no_steps)

      # validate
      valid_acc = model.test_h5(sess, batcher_valid, test_batches=test_epoch_batches)
      valid_loss = valid_acc.loss
      valid_r2 = valid_acc.r2().mean()
      del valid_acc

      best_str = ''
      if best_loss is None or valid_loss < best_loss:
        best_loss = valid_loss
        best_str = ', best!'
        early_stop_i = 0
        saver.save(sess, '%s/model_best.tf' % FLAGS.logdir)
      else:
        early_stop_i += 1

      # measure time
      et = time.time() - t0
      if et < 600:
        time_str = '%3ds' % et
      elif et < 6000:
        time_str = '%3dm' % (et / 60)
      else:
        time_str = '%3.1fh' % (et / 3600)

      # print update
      print(
          'Epoch: %3d,  Steps: %7d,  Train loss: %7.5f,  Valid loss: %7.5f,  Valid R2: %7.5f,  Time: %s%s'
          % (epoch + 1, steps, train_loss, valid_loss, valid_r2, time_str, best_str))
      sys.stdout.flush()

      if FLAGS.check_all:
        saver.save(sess, '%s/model_check%d.tf' % (FLAGS.logdir, epoch))

      # update epoch
      epoch += 1


    if FLAGS.logdir:
      train_writer.close()


if __name__ == '__main__':
  np.random.seed(FLAGS.seed)

  run(params_file=FLAGS.params,
      data_file=FLAGS.data,
      train_epochs=None,
      train_epoch_batches=None,
      test_epoch_batches=None)
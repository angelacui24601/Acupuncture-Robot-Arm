from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def get_dataset(dataset):
  if dataset == 'hand':
    from .datasets.Hand import Hand
    return Hand
  if dataset in ('cervicocranial', 'acusim'):
    from .datasets.CervicoCranial import CervicoCranial
    return CervicoCranial
  raise KeyError(f'Unsupported dataset: {dataset}')


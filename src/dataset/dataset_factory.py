from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .datasets.Hand import Hand

dataset_factory = {
  'hand': Hand,
}

def get_dataset(dataset):
  return dataset_factory[dataset]


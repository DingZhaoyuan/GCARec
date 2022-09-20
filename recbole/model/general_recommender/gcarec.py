

import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import torch.nn.functional as F


class GCARec(GeneralRecommender):
    r"""
    Being collated, to be released soon
    """


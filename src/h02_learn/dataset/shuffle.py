import numpy as np
# import pandas as pd
import torch
# from torch.utils.data import Dataset

# from h01_data.process import get_data_file_base as get_file_names
from .base import BaseDataset
from .pos_tag import PosTagDataset
from .dep_label import DepLabelDataset
from .parse import ParseDataset
# from util import util


class ShuffledDataset(BaseDataset):
    # pylint: disable=abstract-method

    def process(self, classes, words):
        super().process(classes, words)
        np.random.shuffle(self.y)


class ShufflePosTagDataset(ShuffledDataset, PosTagDataset):
    pass


class ShuffleDepLabelDataset(ShuffledDataset, DepLabelDataset):
    pass


class ShuffleParseDataset(ParseDataset):
    # pylint: disable=access-member-before-definition

    def process(self, classes, words):
        super().process(classes, words)
        shuffled_y = []
        for sentence in self.y:
            new_sentence = [-1]
            new_sentence += list(np.random.permutation(sentence[1:]))
            shuffled_y += [torch.from_numpy(np.array(new_sentence))]

        self.y = shuffled_y
        # import ipdb; ipdb.set_trace()
        # np.random.shuffle(self.y)

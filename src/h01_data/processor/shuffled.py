import copy
import random

from .bert import BertProcessor
from .albert import AlbertProcessor
from .roberta import RobertaProcessor


class ShuffledBertProcessor(BertProcessor):
    name = 'bertshuffled'

    @staticmethod
    def iterate_sentence(tokens):
        tokens = copy.copy(tokens)
        random.shuffle(tokens)
        return tokens


class ShuffledAlbertProcessor(ShuffledBertProcessor, AlbertProcessor):
    name = 'albertshuffled'


class ShuffledRobertaProcessor(ShuffledBertProcessor, RobertaProcessor):
    name = 'robertashuffled'

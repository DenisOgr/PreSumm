from os.path import join as pjoin
from os import environ
assert environ.get('PATH_TO_ROOT'), 'Empty enviroment variable ENV. Please set up it: export PATH_TO_ROOT=/path/to/root'
print('PATH_TO_ROOT: %s'%environ.get('PATH_TO_ROOT'))

_MAP = {
    'rubert-deeppavlov': {
        'model': "rubert_deeppavlov/rubert_cased_L-12_H-768_A-12_v1/pt_bert_model.bin",
        'config': "rubert_deeppavlov/rubert_cased_L-12_H-768_A-12_v1/bert_config.json",
        'vocab': 'rubert_deeppavlov/rubert_cased_L-12_H-768_A-12_v1/vocab.txt'
    }
}

def mapper(name, type):
    return pjoin(environ.get('PATH_TO_ROOT'), _MAP[name][type])

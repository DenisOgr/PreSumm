from os.path import join as pjoin
MAIN_PATH = '/Users/denisporplenko/Documents/UCU_master_degree/diploma/raw/PreSumm/other_models/'

_MAP = {
    'rubert-deeppavlov': {
        'model': "rubert_deeppavlov/rubert_cased_L-12_H-768_A-12_v1/pt_bert_model.bin",
        'config': "rubert_deeppavlov/rubert_cased_L-12_H-768_A-12_v1/bert_config.json",
        'vocab': 'rubert_deeppavlov/rubert_cased_L-12_H-768_A-12_v1/vocab.txt'
    }
}

def mapper(name, type):
    return pjoin(MAIN_PATH, _MAP[name][type])

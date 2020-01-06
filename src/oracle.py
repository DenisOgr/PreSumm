import os
import argparse
from os import path
from prepro.data_builder import greedy_selection
from nltk.tokenize import word_tokenize, sent_tokenize
from rouge import Rouge
import pickle
from shutil import copyfile

def clear():
   os.system("cls" if os.name == "nt" else "clear")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", type=str, required=True)
    parser.add_argument("-tgt",  type=str, required=True)
    parser.add_argument("-tgt_path_src",  type=str, required=True)
    parser.add_argument("-count",  type=int, required=True)
    parser.add_argument("-is_store_stat",  type=bool,  default=True)

    args = parser.parse_args()
    assert path.isfile(args.src)
    assert path.isfile(args.tgt)
    assert path.isdir(args.tgt_path_src)
    assert args.count > 0
    tgt_path_src = path.join(args.tgt_path_src, path.basename(args.src))
    assert not path.isfile(tgt_path_src), "File %s exist" % tgt_path_src

    tgt_path_tgt = path.join(args.tgt_path_src, path.basename(args.tgt))
    assert not path.isfile(tgt_path_tgt), "File %s exist" % tgt_path_src

    rouge = Rouge()
    scores_f1 = []
    scores_f2 = []
    scores_fl = []
    number_of_sents = []
    hyp_src_lines = []
    ref_src_lines = []
    with(open(args.src,  encoding='utf-8')) as src:
        with(open(args.tgt,  encoding='utf-8')) as tgt:
            while True:
                line_src = src.readline()
                line_tgt = tgt.readline()
                if not line_src and not line_tgt:
                    break
                if len(hyp_src_lines) % 10 == 0:
                    clear()
                    print(len(hyp_src_lines))
                hyp = [word_tokenize(s, language="russian") for s in sent_tokenize(line_src, language="russian")]
                ref = [word_tokenize(s, language="russian") for s in sent_tokenize(line_tgt, language="russian")]
                result_idx = greedy_selection(hyp, ref, args.count)
                if len(result_idx) == 0:
                    print("Skip!")
                    continue
                result_idx.sort()
                hyp_src_line = " ".join([" ".join(hyp[idx]) for idx in result_idx])
                try:
                    scores = rouge.get_scores(hyp_src_line, line_tgt)
                except:
                    hyp_src_line = "rouge_error"
                hyp_src_lines.append(hyp_src_line)
                ref_src_line = " ".join([" ".join(r) for r in ref])
                ref_src_line = ref_src_line.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!").replace(" »", "»").replace(" «", "«")
                ref_src_lines.append(ref_src_line)
                number_of_sents.append(len(result_idx))
                scores_f1.append((scores[0]['rouge-1']['p'],scores[0]['rouge-1']['r'],scores[0]['rouge-1']['f']))
                scores_f2.append((scores[0]['rouge-2']['p'],scores[0]['rouge-2']['r'],scores[0]['rouge-2']['f']))
                scores_fl.append((scores[0]['rouge-l']['p'],scores[0]['rouge-l']['r'],scores[0]['rouge-l']['f']))
    with(open(tgt_path_src, 'w', encoding='utf-8')) as src:
        for line in hyp_src_lines:
            src.write(line + "\n")

    with(open(tgt_path_tgt, 'w', encoding='utf-8')) as tgt:
        for line in ref_src_lines:
            tgt.write(line + "\n")

    #copyfile(args.tgt, tgt_path_tgt)

    if args.is_store_stat:
        file_pref = "".join(path.basename(args.src).split('.')[:-1])
        with(open(path.join(args.tgt_path_src,file_pref+"_scores_f1"), 'wb')) as f:
            pickle.dump(scores_f1, f)
        with(open(path.join(args.tgt_path_src, file_pref + "_scores_f2"), 'wb')) as f:
            pickle.dump(scores_f2, f)
        with(open(path.join(args.tgt_path_src, file_pref + "_scores_fl"), 'wb')) as f:
            pickle.dump(scores_fl, f)
        with(open(path.join(args.tgt_path_src, file_pref+"_number_of_sents"), 'wb')) as f:
            pickle.dump(number_of_sents, f)




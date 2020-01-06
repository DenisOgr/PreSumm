from glob import glob
from os.path import basename
from others.utils import test_rouge

dirs = list(glob("/Users/denisporplenko/Documents/UCU_master_degree/diploma/raw/PreSumm/result_final/results/origin_*"))
for dir in dirs:
    print("Dir: ", basename(dir))
    for file in glob("%s/r.*.gold"%dir):
        num = basename(file).split(".")[1]
        print(num)
        cand = "%s/r.%s.candidate"%(dir, num)
        ref = "%s/r.%s.gold" % (dir, num)
        try:
            print(test_rouge("", cand, ref))
        except:
            print("Error")

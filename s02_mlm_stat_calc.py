
from mlm_shared import CACHE_DIR

import os
import math


files_list = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.el') and os.path.isfile(os.path.join(CACHE_DIR, f))]

totals = {}

for fn in files_list:
    with open(fn, 'r', encoding='utf-8') as elf:
        for line in elf:
            line = line.strip()
            indices = line.split(' ')
            indices = zip(indices[::2], indices[1::2])
            for p in indices:
                totals[p] = totals.get(p, 0) + 1

totals = sorted(totals.items(), key=lambda item: item[1], reverse=True)


tmax = 300000

stfn = os.path.join(CACHE_DIR, 'stat.info')
with open(stfn, 'w', encoding='utf-8') as stf:
    for k,v in totals:
        if v > tmax:
            skip_prob = 1.0 - float(tmax)/float(v)
            if skip_prob > 0.975:
                skip_prob = 0.975
            stf.write("{} {} {} {}\n".format( k[0],k[1], v, skip_prob ))
        else:
            stf.write("{} {} {} {}\n".format( k[0],k[1], v, 0 ))

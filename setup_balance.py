# Скрипт для переопределения балансировочных коэффициентов модели

from embeddings_reader import EmbeddingsReader

import argparse
import sys
import numpy as np


def setup_special(cat_balances, ass_balances, gra_balances, stem_idx, sfx_idx, stem_val, sfx_val):
    cat_balances[stem_idx] = stem_val
    ass_balances[stem_idx] = stem_val
    gra_balances[sfx_idx] = sfx_val


def main():
    parser = argparse.ArgumentParser(description='setup_balance')
    parser.add_argument('cat_value', type=float, default=0.75, help='new balance_value for categorial part')
    parser.add_argument('ass_value', type=float, default=0.7, help='new balance_value for associative part')
    parser.add_argument('gra_value', type=float, default=0.7, help='new balance_value for grammatical part')
    parser.add_argument('spc_value', type=float, default=0.3, help='new balance_value for specials')
    args = parser.parse_args()
    # if not args.cat_value:
    #     parser.print_help()
    #     sys.exit(-1)
    # load balances data
    BALANCES_FN = 'lle_weights/balances.npz'
    with np.load( BALANCES_FN ) as wdict:
        cat_balances = wdict['cat_balances']
        ass_balances = wdict['ass_balances']
        gra_balances = wdict['gra_balances']
    # setup new value
    cat_balances[:] = args.cat_value
    ass_balances[:] = args.ass_value
    gra_balances[:] = args.gra_value
    # setup special values: numbers, punctuation
    vm = EmbeddingsReader('vectors-rue.c2v')
    num_stem, num_sfx = vm.token2ids('1')
    setup_special(cat_balances, ass_balances, gra_balances, num_stem, num_sfx, args.spc_value, args.spc_value)
    signs = '.,!?;…:-—–‒ʼˮ«“„‘‚»”‟’‛([{⟨)]}⟩\\\''
    for sign in signs:
        p_stem, p_sfx = vm.token2ids(sign)
        setup_special(cat_balances, ass_balances, gra_balances, p_stem, p_sfx, args.spc_value, args.spc_value)
    # setup PAD and OOV
    setup_special(cat_balances, ass_balances, gra_balances, EmbeddingsReader.PAD_IDX, EmbeddingsReader.PAD_IDX, 0.0, 0.0)
    setup_special(cat_balances, ass_balances, gra_balances, EmbeddingsReader.OOV_IDX, EmbeddingsReader.OOV_IDX, 1.0, 1.0)
    
    wdict = {}
    wdict['cat_balances'] = cat_balances
    wdict['ass_balances'] = ass_balances
    wdict['gra_balances'] = gra_balances
    
    np.savez(BALANCES_FN, **wdict)


if __name__ == '__main__':
    main()



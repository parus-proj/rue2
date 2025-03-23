
from embeddings_reader import EmbeddingsReader
from tokenizer import Tokenizer

import os
#import tensorflow as tf
import numpy as np


test_words = ['производительность', 'артиллерист', 'милитаризация', 'аккордеон', 'кофе', 'патп', 'фиолетовая', 'вскоре', 'мы', 'чего-то', 'приземлился', 
              'европейского', 'государств', 'галку', 'президент', 'работу', 'императору',
              'судов', 'суда', 'судам', 'судах', 'орган', 'органы', 'карта', 'в', 'бумаги', 'банка', 'рада', 'рады', 'стоит', 'корабль', 'звезда', 'уже', 'чайку', 'чаек', 'чумы', 'врывавшийся', 'и', 'на', 'военный', 'крыло', 'крылья', 
              'аппарат', 'система', 'комплекс', 'средство', 'публикация', 'публикации', 'управление', 'управлении', 'организация', 'организации',
              'иванова', 'владимира', 
              '111', '.', ',', 'несловоOOV']


# Загрузим векторную модель (одновременно добавив в нее служебные векторы)
print('Loading embeddings...')
vm = EmbeddingsReader("vectors-rue.c2v")
# Создадим токенизатор
print('Creating tokenizer...')
tokenizer = Tokenizer(vm)

encoded_test_words = tokenizer.tokens2subtokens([w.lower() for w in test_words])

wdict = np.load(os.path.join('lle_weights', 'balances') +'.npz')


for w, ew in zip(test_words, encoded_test_words):
    print( '{}: {}, {}, {}'.format(w, wdict['cat_balances'][ew[0]], wdict['ass_balances'][ew[0]], wdict['gra_balances'][ew[1]]) )


# Скрипт для сохранения обучающих данных в виде numpy-массивов с разбиением на батчи

from mlm_shared import MAX_LEN
from mlm_shared import ENCODER_FIXED_SEQ_LEN
from mlm_shared import DECODER_FIXED_SEQ_LEN
from mlm_shared import BATCH_SIZE
from mlm_shared import CACHE_DIR

from embeddings_reader import EmbeddingsReader
from tokenizer import Tokenizer
import sentencepiece as spm

import numpy as np
import os
import sys
import random



# Функция удаления всех файлов в каталоге и его подкаталогах (не удаляет сами каталоги)
def remove_files_from_dir(directory):
    for root, dirs, files in os.walk(directory):
        for f in files:
            os.remove(os.path.join(root, f))    


# Загрузим векторную модель (одновременно добавив в нее служебные векторы)
print('Loading embeddings...')
vm = EmbeddingsReader("vectors-rue.c2v", force_oov_with_sfx=True)
print('  stems emb size = {}'.format(vm.stems_size))
print('  sfx emb size = {}'.format(vm.sfx_size))
print('  stems emb count = {}'.format(vm.stems_count))
print('  sfx emb count = {}'.format(vm.sfx_count))

# Создадим токенизатор
print('Creating tokenizer...')
tokenizer = Tokenizer(vm)

# Загрузим sentencepiece-модель
sp = spm.SentencePieceProcessor(model_file='sp_m.model')


# Преобразуем обучающие данные (из stdin) в текстовые файлы с массивами данных размером в несколько batch'ей (для генератора)
# (текстовые компактней, чем numpy с выравниванием на max_len)
print('Creating data chunks...')
os.makedirs(CACHE_DIR, exist_ok=True)
remove_files_from_dir(CACHE_DIR)
CHUNK_SIZE = BATCH_SIZE * 1024
BD_CHUNK_SIZE = CHUNK_SIZE * 2
chunk_count, sentence_count, taken_sentence_count, bd_chunk_count, bd_examples_count = 0, 0, 0, 0, 0
current_sentence, X, BD = [], [], []
for line in sys.stdin:
    line = line.strip()
    if line == '':
        # вычитали предложение, порождаем данные
        sentence_count += 1
        if len(current_sentence) < 10: # не берем короткие предложение
            current_sentence = []
            continue
        current_sentence = current_sentence[:MAX_LEN]
        current_sentence_e = tokenizer.tokens2subtokens(current_sentence)
        # данные для обучения контекстуализации
        oov_in_current_sentence = False
        for idx1, idx2 in current_sentence_e:
            if idx1 == EmbeddingsReader.OOV_IDX or idx2 == EmbeddingsReader.OOV_IDX:
                oov_in_current_sentence = True
                break
        if not oov_in_current_sentence: # не берем предложения, содержащие OOV
            X.append(current_sentence_e)
            if len(X) == CHUNK_SIZE:
                fn = os.path.join(CACHE_DIR, "{:0>10d}.el".format(chunk_count))
                with open(fn, 'w', encoding='utf-8') as f:
                    for s in X:
                        cs = ' '.join([' '.join([str(e1), str(e2)]) for e1, e2 in s])
                        f.write( cs + '\n')
                taken_sentence_count += CHUNK_SIZE
                chunk_count += 1
                X = []
        # данные для обучения балансировке
        oov_in_fixed_seq = False
        for idx1, idx2 in current_sentence_e[:ENCODER_FIXED_SEQ_LEN]:
            if idx1 == EmbeddingsReader.OOV_IDX or idx2 == EmbeddingsReader.OOV_IDX:
                oov_in_fixed_seq = True
                break
        if not oov_in_fixed_seq: # не берем предложения, содержащие OOV в части предложения, подаваемой на энкодер
            current_sentence_len = len(current_sentence)
            offset = 0
            while True:
                tail_pos = offset + ENCODER_FIXED_SEQ_LEN
                if current_sentence_len < tail_pos + 2:
                    break
                tail_str = ' '.join(current_sentence[tail_pos:])
                tail_sp  = sp.encode(tail_str, out_type=int)
                if len(tail_sp) < DECODER_FIXED_SEQ_LEN:
                    break
                skip = len(current_sentence[tail_pos]) < 4 and random.random() > 0.01 # пореже предсказываем коротышек (знаки препинания, союзы/предлоги/местоимения)
                if not skip:
                    BD.append([current_sentence_e[offset:tail_pos], tail_sp[:DECODER_FIXED_SEQ_LEN]])
                if ( current_sentence_e[tail_pos][0] == EmbeddingsReader.OOV_IDX or current_sentence_e[tail_pos][1] == EmbeddingsReader.OOV_IDX ): # если дальше OOV, то прерываем обработку предложения (проверка нужна, т.к. изначально была проверена только начальная часть предложения)
                    break
                offset += 1
            
            if len(BD) >= BD_CHUNK_SIZE:
                BD = BD[:BD_CHUNK_SIZE]
                fn = os.path.join(CACHE_DIR, "{:0>10d}.bd".format(bd_chunk_count))
                with open(fn, 'w', encoding='utf-8') as f:
                    for s, sps in BD:
                        cs = ' '.join([' '.join([str(e1), str(e2)]) for e1, e2 in s])
                        cs_sp = ' '.join(str(i) for i in sps)
                        f.write( cs + '/' + cs_sp + '\n')
                bd_examples_count += BD_CHUNK_SIZE
                bd_chunk_count += 1
                BD = []
        current_sentence = []        
        
    elif line[0] == '#':
        continue
    else:
        fields = line.split('\t')
        current_sentence.append(fields[1].lower()) # векторная модель содержит только слова в нижнем регистре

print('  chunks count = {}'.format(chunk_count))
print('  sentence count = {}'.format(sentence_count))
print('  taken sentence count = {}'.format(taken_sentence_count))
print()
print('  bd_chunks count = {}'.format(bd_chunk_count))
print('  bd examples count = {}'.format(bd_examples_count))

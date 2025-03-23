
from embeddings_reader import EmbeddingsReader
from tokenizer import Tokenizer
from low_level_encoder import LowLevelEncoder
from mlm_shared import MAX_LEN, CATEG_DIMS, ASSOC_DIMS, GRAMM_DIMS, LEX_DIMS, TOTAL_DIMS

import tensorflow as tf
import numpy as np

import argparse
import random


sentences = [
                  "Я видел озорную галку сегодня утром, когда шел на работу!"
                , "Президент В. Путин прибыл в Санкт-Петербург на встречу глав государств европейского региона."
#                  , ["Синий", "и", "белый", "автомобили", "неспешно", "ехали", "по", "мокрому", "шоссе", "."]
#                  , ["На", "рыбалке", "мы", "провели", "более", "часа", "и", "поймали", "десяток", "увесистых", "окуней", "."]
#                  , ["На", "каждый", "урок", "приходит", "8", "учеников", "класса", ",", "но", "все", "двоечники", "."]
#                  , ["На", "берегу", "стоит", "дуб", ",", "а", "под", "дубом", "растёт", "крупная", "черника", "."]
#                  , ["Слабая", "организация", "стала", "причиной", "провала", "конференции", ",", "что", "сказалось", "на", "результатах", "."]
                , "В воинскую часть доставили четыре зенитно-ракетных системы и два новых танка."
#                  , ["На", "судах", "оборудование", "доставили", "в", "черноморский", "порт", "Керчь", ",", "далее", "везли", "по", "суше", "."]
		, "Часть транспортных средств ярославского ПАТП находится в залоге у банка «Югра»."
#                  , ["На", "повестке", "дня", "регионального", "законодательного", "органа", "вопросы", "экологии", "и", "развития", "сферы", "туризма", "."]
#                  , ["Получив", "повестку", "из", "военкомата", ",", "студенты", "отправляются", "на", "медкомиссию", "проверять", "внутренние", "органы", "."]
#                  , ["Потерпевший", "обратился", "в", "органы", "на", "следующий", "день", "после", "инцидента", "с", "соседями", "."]
#                  , ["Звуки", "старинного", "органа", "наполняли", "своды", "храма", ",", "а", "в", "хрустальных", "сферах", "люстры", "мельтешили", "веселые", "огоньки", "."]
                , "Запуск космического челнока намечен на субботу, 21 марта 2021 г."
                , "6 воздушных судов с гуманитарным грузом направлены в Египет из Москвы."
                , "Графическая карта с новыми гигагерцовыми ядрами выйдет на рынок в следующем году."
                , "На игральной карте были изображены артиллерист в зелёном мундире и пушка с ядрами."
#                  , ["Д", ".", "Песков", "сообщил", "журналистам", ",", "что", "в", "аристократической", "среде", "появились", "заговорщики", "!"]
#                  , ["К", "среде", "ФСБ", "обнаружила", "конспиративную", "квартиру", "с", "оружием", "и", "запрещенной", "литературой", "."]
#                  , ["Где", "может", "находиться", "штаб-квартира", "террористической", "ячейки", "и", "её", "главари", "?"]
#                  , ["В", "следующей", "главе", "книги", "повествуется", "о", "жизни", "экс-губернатора", "в", "Лондоне", "."]
		, "Стали нержавеющей нет, но пятно стекло на пол."
#                  , ["Глокая", "куздра", "штеко", "будланула", "сугмого", "бокра", "и", "курдячит", "бокрёнка", "."]
                , "Когда Иван Данилович приезжал к нам, он привозил конфеты и вафли."
                , "В реакции полимеризации участвуют мономеры, в молекулах которых имеются кратные связи: этилен, пропилен, стирол, диеновые углеводороды и другие ненасыщенные вещества."
            ]



class SentenceProcessor(object):
    def __init__(self, oov_percent, near_count, show_lengthes, show_attention, show_balance, balance_cat, balance_ass, balance_gra, **kwargs):
        super().__init__(**kwargs)
        self.oov_percent = oov_percent
        self.near_count = near_count
        self.show_lengthes = show_lengthes
        self.show_attention = show_attention
        self.show_balance = show_balance
        self.balance_cat = balance_cat
        self.balance_ass = balance_ass
        self.balance_gra = balance_gra
        # Загрузим векторную модель
        print('Loading embeddings...')
        self.vm = EmbeddingsReader("vectors-rue.c2v")
        # Создадим токенизатор
        print('Creating tokenizer...')
        self.tokenizer = Tokenizer(self.vm)
        # Создадим модель
        print('Creating model...')
        input = tf.keras.layers.Input(shape=(MAX_LEN,2), name='data')
        output = LowLevelEncoder( CATEG_DIMS, ASSOC_DIMS, GRAMM_DIMS, 
                                  self.vm.stems_count, self.vm.sfx_count, 
                                  ctxer_training=False, balance_training_cat=False, balance_training_ass=False, balance_training_gra=False, return_attention=self.show_attention,
                                  name='lle' ) (input)
        self.model = tf.keras.Model(inputs=input, outputs=output, name='MLM')
        lle = self.model.get_layer('lle')
        lle.load_embeddings(self.vm)
        lle.load_own_weights('lle_weights')
        self.model.summary()
        # Переопределим балансы
        self.balances_dict = lle.get_balances_dict()
        if self.balance_cat is not None:
            wm = self.balances_dict['cat_balances'].weights
            self.balances_dict['cat_balances'].set_weights( tf.ones_like(wm) * self.balance_cat ) 
        if self.balance_ass is not None:
            wm = self.balances_dict['ass_balances'].weights
            self.balances_dict['ass_balances'].set_weights( tf.ones_like(wm) * self.balance_ass ) 
        if self.balance_gra is not None:
            wm = self.balances_dict['gra_balances'].weights
            self.balances_dict['gra_balances'].set_weights( tf.ones_like(wm) * self.balance_gra ) 
        # Подготовим нормированные матрицы для вычисления косинусного расстояния
        self.vm_stem_dims_normalized = tf.nn.l2_normalize( self.vm.stems_embs, -1 )
        self.vm_cat_dims_normalized = tf.nn.l2_normalize( self.vm.stems_embs[:, :CATEG_DIMS], -1 )
        self.vm_ass_dims_normalized = tf.nn.l2_normalize( self.vm.stems_embs[:, CATEG_DIMS:], -1 )
        self.vm_gra_dims_normalized = tf.nn.l2_normalize( self.vm.sfx_embs, -1 )
        # Подготовим обратные индексы для словарей векторной модели
        self.vm_reverse_stems_dict = { idx: word for word, idx in self.vm.stems_dict.items() }
        self.vm_reverse_stems_dict[EmbeddingsReader.PAD_IDX] = EmbeddingsReader.PAD_NAME
        self.vm_reverse_stems_dict[EmbeddingsReader.OOV_IDX] = EmbeddingsReader.OOV_NAME
        self.vm_reverse_sfx_dict = { idx: word for word, idx in self.vm.sfx_dict.items() }
        self.vm_reverse_sfx_dict[EmbeddingsReader.PAD_IDX] = EmbeddingsReader.PAD_NAME
        self.vm_reverse_sfx_dict[EmbeddingsReader.OOV_IDX] = EmbeddingsReader.OOV_NAME
        
    def print_table_header(self, is_attention_relevant = False):
        s = '    '
        s = s + '{:35}'.format('Слово предложения')
        if self.oov_percent > 0:
            s = s + ' | {:3}'.format('OOV')
        if self.show_lengthes:
            s = s + ' | {:4}'.format('LEN')
            s = s + ' | {:4}'.format('SLEN')
        if self.show_balance:
            s = s + ' | {:4}'.format('BAL')
        if is_attention_relevant and self.show_attention:
            s = s + ' | {:5}'.format('ATT')
        s = s + ' | Ближайшие'
        print(s)
        print('    ----------------------------------------------------------------------------------')

    def run_sentence(self, idx):
        print()
        print(sentences[idx])
        # токенизируем текст предложения
        tokens, tokens_poses = self.tokenizer.process_text(sentences[idx])
        tokens_initial = tokens
        stems_initial = [[stm] for (stm,sfx) in tokens_initial]
        sfx_initial = [[sfx] for (stm,sfx) in tokens_initial]
        token_strings = [ sentences[idx][tokens_poses[i]:tokens_poses[i+1]].rstrip() for i in range(len(tokens_poses)-1) ] + [sentences[idx][tokens_poses[-1]:].rstrip()]
        tokens_cnt = len(tokens)
        # применяем oov-маскирование
        randoms_for_mask = random.sample(range(0, 100), len(tokens))
        oov_mask = [ True if w < self.oov_percent else False for w in randoms_for_mask]
        tokens = [ [EmbeddingsReader.OOV_IDX, EmbeddingsReader.OOV_IDX] if m else t for t,m in zip(tokens,oov_mask)]
        # выравниваем длину предложения на размер входной последовательности у модели
        tokens = tokens[:MAX_LEN]
        tokens = tokens + [[EmbeddingsReader.PAD_IDX, EmbeddingsReader.PAD_IDX]] * (MAX_LEN - len(tokens))
        # делаем предсказание моделью
        x = tf.convert_to_tensor([tokens], dtype=tf.int32)
        if self.show_attention:
            p, att = self.model(x, training=False)
            att = tf.squeeze(att, axis=0)
            att = att[:tokens_cnt].numpy().tolist()
        else:
            p = self.model(x, training=False)
        p = tf.squeeze(p, axis=0)
        p = p[:tokens_cnt, :]
        # вычисляем ближайшие по стем-измерениям
        print('  STEM PART PREDICTION')
        p_stem_dims_normalized = tf.nn.l2_normalize(p[:, 0:LEX_DIMS],-1)
        stem_cos_similarities = tf.matmul(self.vm_stem_dims_normalized, p_stem_dims_normalized, transpose_b=True)
        stem_idxs = tf.math.argmax(stem_cos_similarities, axis=0).numpy().tolist()
        stem_strlist = [ self.vm_reverse_stems_dict[st] for st in stem_idxs ]
        print('    {}'.format(' '.join(stem_strlist)))
        # вычисляем ближайшие по категориальным измерениям
        print('  CATEGORIAL PART PREDICTION')
        p_cat_dims_normalized = tf.nn.l2_normalize(p[:, 0:CATEG_DIMS],-1)
        cat_cos_similarities = tf.matmul(self.vm_cat_dims_normalized, p_cat_dims_normalized, transpose_b=True)
        cat_idxs_top = tf.math.top_k( tf.transpose(cat_cos_similarities), k=self.near_count, sorted=True ).indices
        cat_idxs_top = cat_idxs_top.numpy().tolist()
        cat_strlist = [' '.join([self.vm_reverse_stems_dict[v] for v in st])  for st in cat_idxs_top ]
        cat_lengthes = tf.norm( p[:, 0:CATEG_DIMS], ord='euclidean', axis=1 )
        scat_vectors = tf.gather_nd(self.vm.stems_embs[:, 0:CATEG_DIMS], stems_initial)
        scat_lengthes = tf.norm( scat_vectors, ord='euclidean', axis=1 ) # длины соответствующих статических эмбеддингов
        cat_balances = tf.gather_nd(self.balances_dict['cat_balances'].weights[0], stems_initial)
        cat_balances = tf.squeeze(cat_balances, axis=1)
        self.print_table_header()
        for i in range(tokens_cnt):
            s = '    '
            s = s + '{:35}'.format(token_strings[i])
            if self.oov_percent > 0:
                s = s + ' | {:3}'.format(' X ' if oov_mask[i] else '')
            if self.show_lengthes:
                s = s + ' | {:4.1f}'.format(cat_lengthes[i])
                s = s + ' | {:4.1f}'.format(scat_lengthes[i])
            if self.show_balance:
                s = s + ' | {:4.2f}'.format(cat_balances[i])
            s = s + ' | ' + cat_strlist[i]
            print(s)
        # вычисляем ближайшие по ассоциативным измерениям
        print('  ASSOCIATIVE PART PREDICTION')
        p_ass_dims_normalized = tf.nn.l2_normalize(p[:, CATEG_DIMS:LEX_DIMS],-1)
        ass_cos_similarities = tf.matmul(self.vm_ass_dims_normalized, p_ass_dims_normalized, transpose_b=True)
        ass_idxs_top = tf.math.top_k( tf.transpose(ass_cos_similarities), k=self.near_count, sorted=True ).indices
        ass_idxs_top = ass_idxs_top.numpy().tolist()
        ass_strlist = [' '.join([self.vm_reverse_stems_dict[v] for v in st])  for st in ass_idxs_top ]
        ass_lengthes = tf.norm( p[:, CATEG_DIMS:LEX_DIMS], ord='euclidean', axis=1 )
        sass_vectors = tf.gather_nd(self.vm.stems_embs[:, CATEG_DIMS:LEX_DIMS], stems_initial)
        sass_lengthes = tf.norm( sass_vectors, ord='euclidean', axis=1 )
        ass_balances = tf.gather_nd(self.balances_dict['ass_balances'].weights[0], stems_initial)
        ass_balances = tf.squeeze(ass_balances, axis=1)
        self.print_table_header(True)
        for i in range(tokens_cnt):
            s = '    '
            s = s + '{:35}'.format(token_strings[i])
            if self.oov_percent > 0:
                s = s + ' | {:3}'.format(' X ' if oov_mask[i] else '')
            if self.show_lengthes:
                s = s + ' | {:4.1f}'.format(ass_lengthes[i])
                s = s + ' | {:4.1f}'.format(sass_lengthes[i])
            if self.show_balance:
                s = s + ' | {:4.2f}'.format(ass_balances[i])
            if self.show_attention:
                s = s + ' | {:5.3f}'.format(att[i])
            s = s + ' | ' + ass_strlist[i]
            print(s)
        # вычисляем ближайшие по грамматическим измерениям
        print('  GRAMMATICAL PART PREDICTION')
        p_gra_dims_normalized = tf.nn.l2_normalize(p[:, LEX_DIMS:],-1)
        gra_cos_similarities = tf.matmul(self.vm_gra_dims_normalized, p_gra_dims_normalized, transpose_b=True)
        gra_idxs_top = tf.math.top_k( tf.transpose(gra_cos_similarities), k=self.near_count, sorted=True ).indices
        gra_idxs_top = gra_idxs_top.numpy().tolist()
        gra_strlist = [' '.join([self.vm_reverse_sfx_dict[v] for v in st])  for st in gra_idxs_top ]
        gra_lengthes = tf.norm( p[:, LEX_DIMS:], ord='euclidean', axis=1 )
        sgra_vectors = tf.gather_nd(self.vm.sfx_embs, sfx_initial)
        sgra_lengthes = tf.norm( sgra_vectors, ord='euclidean', axis=1 )
        gra_balances = tf.gather_nd(self.balances_dict['gra_balances'].weights[0], sfx_initial)
        gra_balances = tf.squeeze(gra_balances, axis=1)
        self.print_table_header()
        for i in range(tokens_cnt):
            s = '    '
            s = s + '{:35}'.format(token_strings[i])
            if self.oov_percent > 0:
                s = s + ' | {:3}'.format(' X ' if oov_mask[i] else '')
            if self.show_lengthes:
                s = s + ' | {:4.1f}'.format(gra_lengthes[i])
                s = s + ' | {:4.1f}'.format(sgra_lengthes[i])
            if self.show_balance:
                s = s + ' | {:4.2f}'.format(gra_balances[i])
            s = s + ' | ' + gra_strlist[i]
            print(s)
        
    
    def run(self):
        print()
        print('Processing sentences...')
        sentences_count = len(sentences)
        for i in range(sentences_count):
            self.run_sentence(i)
        
    


def main():
    parser = argparse.ArgumentParser(description='see mlm')
    parser.add_argument('--oov_percent', type=int, default=5, help='Random OOV substitutions to input sentence')
    parser.add_argument('--near_count', type=int, default=3, help='Output <top K> similar words')
    parser.add_argument('--show_lengthes', action='store_true', help='Show vector lengthes')
    parser.add_argument('--show_attention', action='store_true', help='Show thematic attention')
    parser.add_argument('--show_balance', action='store_true', help='Show balance values')
    parser.add_argument('--balance_cat', type=float, help='Static/Dynamic balance for categorial')
    parser.add_argument('--balance_ass', type=float, help='Static/Dynamic balance for associative')
    parser.add_argument('--balance_gra', type=float, help='Static/Dynamic balance for grammatical')
    args = parser.parse_args()
    #if not args.mode or not args.time:
    #    parser.print_help()
    #    sys.exit(-1)
    sp = SentenceProcessor( args.oov_percent, args.near_count, args.show_lengthes, args.show_attention, 
                            args.show_balance, args.balance_cat, args.balance_ass, args.balance_gra )
    sp.run()


if __name__ == "__main__":
    main()


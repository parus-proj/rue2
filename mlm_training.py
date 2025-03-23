# Скрипт для обучения модели

# Использовать ли все GPU на машине
USE_ALL_GPU = False

from mlm_shared import MAX_LEN
#from mlm_shared import BATCH_SIZE
from mlm_shared import CACHE_DIR

from mlm_shared import CATEG_DIMS
from mlm_shared import ASSOC_DIMS
from mlm_shared import GRAMM_DIMS
from mlm_shared import LEX_DIMS
#from mlm_shared import TOTAL_DIMS

# from mlm_shared import ENC_WARMUP_STEPS
# from mlm_shared import ENC_STEPS_TO_CHANGE
# from mlm_shared import ENC_MIN_LR
# from mlm_shared import ENC_MAX_LR
# from mlm_shared import ENC_CHANGE_VALUE



from embeddings_reader import EmbeddingsReader

from low_level_encoder import LowLevelEncoder


import argparse
import os
import time
import random
#import numpy as np
import tensorflow as tf


# # контроллер скорости обучения для сети Энкодер
# class CustomScheduleEncOnly(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, warmup_steps=2000, min_lr = 1e-5, max_lr=1e-3, steps_to_change=300, change_val=5e-6):
#         super(CustomScheduleEncOnly, self).__init__()
#         self.warmup_steps = warmup_steps
#         self.min_lr = min_lr
#         self.max_lr = max_lr
#         self.steps_to_change = steps_to_change
#         self.change_val = change_val
#      
#     def __call__(self, step):
#         return tf.where( tf.less_equal( step,  self.warmup_steps ), self.incr(step), self.decr(step) )
#     def incr(self, step):
#         return (self.max_lr-self.min_lr)/self.warmup_steps * step + self.min_lr
#     def decr(self, step):
#         current = self.max_lr - (step-self.warmup_steps)/self.steps_to_change * self.change_val
#         return tf.where(current<self.min_lr, self.min_lr, current)



# загрузим информацию о частотах отдельных словоформ в обучающем множестве
#   для уменьшения влияния сверхчастотных словоформ на обучение (пунктуации, предлогов, местоимений и т.п.)
# также загрузим  коэффициенты значимости
#   для снижения вклада многозначных слов в ошибку
statfn = os.path.join(CACHE_DIR, 'stat2.info')
stat_keys, stat_vals, sign_vals = [], [], []
with open(statfn, 'r', encoding='utf-8') as sf:
    for line in sf:
        line = line.strip().split()
        stat_keys.append("{}_{}".format(line[0], line[1]))
        stat_vals.append(float(line[3]))
        sign_vals.append(float(line[4]))
stat_lookup_init = tf.lookup.KeyValueTensorInitializer(stat_keys, stat_vals, key_dtype=tf.string, value_dtype=tf.float32)
stat_lookup_table = tf.lookup.StaticHashTable(stat_lookup_init, default_value=0.0)
sign_lookup_init = tf.lookup.KeyValueTensorInitializer(stat_keys, sign_vals, key_dtype=tf.string, value_dtype=tf.float32)
sign_lookup_table = tf.lookup.StaticHashTable(sign_lookup_init, default_value=1.0)


# мэппинг записи из файла данных в тензор, аналогичный по структуре тому, что выдает для предложения токенизатор
# вторым результатом (в выходном кортеже) идет статистический показатель для сабсэмплинга частотных словоформ
# третьим -- коэффициент понижения вклада в ошибку (снижения влияния многозначных слов)
def record_as_tokenized(x):
    l1 = tf.strings.split(x, sep=tf.constant(' '))
    l2 = tf.strings.to_number(l1, out_type=tf.int32)
    l3 = tf.reshape(l2, [-1, 2])                                # (sentence_len, 2) -- входное предложение после токенизации
    l4 = tf.reshape(l1, [-1, 2])
    l5 = tf.strings.reduce_join(l4, axis=-1, separator="_")
    l6 = stat_lookup_table.lookup(l5)                           # (sentence_len) -- вероятности сабсэмплирования токена в предложении
    l7 = sign_lookup_table.lookup(l5)                           # (sentence_len) -- коэффициент понижения вклада в ошибку
    return l3, l6, l7

# паддинг записи данных (а также формирование маски сабсэмплирования)
def record_padding_and_subsampling(x, stat, signif):
    seq_len = tf.shape(x)[0]
    pad_len = MAX_LEN - seq_len

    sentece_pad_seq = tf.tile( [[EmbeddingsReader.PAD_IDX, EmbeddingsReader.PAD_IDX]], [pad_len, 1] )

    stat_padded = tf.concat([stat, tf.tile( [1.0], [pad_len])], 0)
    rnd = tf.random.uniform([MAX_LEN], minval=0.0, maxval=0.999, dtype=tf.float32)
    subsampled_sentece_msk = tf.where(rnd < stat_padded, tf.zeros_like(stat_padded, dtype=tf.int32), tf.ones_like(stat_padded, dtype=tf.int32)) # маска входной последовательности с учетом сабсэмплинга (единицы -- токены, нули -- PAD и стёртое сабсэмплингом)

    signif_padded = tf.concat([signif, tf.tile( [0.0], [pad_len])], 0)

    
    return tf.concat([x, sentece_pad_seq], 0), subsampled_sentece_msk, signif_padded



# функция создания модели
def create_model_for_mlm(vm, args):
    input = tf.keras.layers.Input(shape=(MAX_LEN,2), name='data')
    smsk_input = tf.keras.layers.Input(shape=(MAX_LEN), name='smsk')
    output = LowLevelEncoder( CATEG_DIMS, ASSOC_DIMS, GRAMM_DIMS, vm.stems_count, vm.sfx_count, 
                              ctxer_training=True, balance_training_cat=False, balance_training_ass=False, balance_training_gra=False,  return_attention=False,
                              drp0v=args.drp0, drp1v=args.drp1, drpfv=args.drpf,
                              name='lle') ([input, smsk_input])
    return tf.keras.Model(inputs=[input, smsk_input], outputs=output, name='MLM')


# def stem_idx_to_str(idx, vm):
#     if idx == EmbeddingsReader.PAD_IDX:
#         return EmbeddingsReader.PAD_NAME
#     if idx == EmbeddingsReader.OOV_IDX:
#         return EmbeddingsReader.OOV_NAME
#     all_stems = [word for word, i in vm.stems_dict.items() if i == idx]
#     return all_stems[0]
#   
# def sfx_idx_to_str(idx, vm):
#     if idx == EmbeddingsReader.PAD_IDX:
#         return EmbeddingsReader.PAD_NAME
#     if idx == EmbeddingsReader.OOV_IDX:
#         return EmbeddingsReader.OOV_NAME
#     idx = idx - EmbeddingsReader.SPECIAL_EMBEDDINGS_COUNT
#     return list(vm.sfx_dict)[idx + 1]



class MlmTrainingController(object):
    def __init__(self, mode, time, reset_optimizer_frequently, randomize_dataset, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.time = time
        self.reset_optimizer_frequently = reset_optimizer_frequently
        
        # Создадим объект-стратегию
        #print('GPU physical devices: {}'.format(tf.config.list_physical_devices('GPU')))
        #print('GPU logical devices: {}'.format(tf.config.list_logical_devices('GPU')))
        if USE_ALL_GPU:
            self.strategy = tf.distribute.MirroredStrategy() # стратегия single host multi gpu
        else:
            self.strategy = tf.distribute.get_strategy() # умолчательная стратегия
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

        # Загрузим векторную модель (одновременно добавив в нее служебные векторы)
        print('Static model loading')
        self.vm = EmbeddingsReader("vectors-rue.c2v")

        # Создадим датасет для доступа к данным
        print('Prepare dataset')
        self.GLOBAL_BATCH_SIZE = batch_size * self.strategy.num_replicas_in_sync
        files_list = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.el') and os.path.isfile(os.path.join(CACHE_DIR, f))]
        if randomize_dataset:
            random.shuffle(files_list)
        self.dataset = tf.data.TextLineDataset(files_list) \
                        .map(record_as_tokenized, num_parallel_calls=tf.data.AUTOTUNE) \
                        .map(record_padding_and_subsampling, num_parallel_calls=tf.data.AUTOTUNE) \
                        .batch(self.GLOBAL_BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False) \
                        .prefetch(tf.data.AUTOTUNE)
#         for x, x_stat, x_sign in self.dataset.take(1): # 1 batch
#             for i in range(10): # N sentences
#                 s = ""
#                 for j in range(MAX_LEN):
#                     j_stem, j_sfx = x[i][j]
#                     if j_stem == EmbeddingsReader.PAD_IDX:
#                         break
#                     j_stem = stem_idx_to_str(j_stem, self.vm)
#                     j_sfx = sfx_idx_to_str(j_sfx, self.vm)
#                     s += "  {}~{}[{}]".format(j_stem, j_sfx, x_stat[i][j])
#                 print(s)
#                 print()
#         import sys        
#         sys.exit(0)
        if USE_ALL_GPU:
            # донастраиваем датасет под стратегию
            ds_options = tf.data.Options()
            ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            self.dataset = self.dataset.with_options(ds_options)
            self.dataset = self.strategy.experimental_distribute_dataset(self.dataset)
        
        # Создадим loss-объекты  
        print('Loss creation')
        self.loss_object = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        self.secondary_loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def __del__(self):
        ## fix: https://github.com/tensorflow/tensorflow/issues/50487
        #import atexit
        #atexit.register(self.strategy._extended._collective_ops._pool.close) # для корректного закрытия ThreadPool
        pass
        
    def prepare(self, args):
        # Создадим модель и оптимизатор
        with self.strategy.scope():
            print('Create model')
            self.mlm_model = create_model_for_mlm(self.vm, args)
        # Загрузим веса
        print('Weights loading')
        self.mlm_model.get_layer('lle').load_embeddings(self.vm)
        if self.mode == 'continue':
            self.mlm_model.get_layer('lle').load_own_weights('lle_weights')
        # получим доступ к слоям эмбеддингов -- они используются как перекодировщики при вычислении loss
        self.stems_embedder = self.mlm_model.get_layer('lle').stems_emb_layer
        self.sfx_embedder   = self.mlm_model.get_layer('lle').sfx_emb_layer
        # выведем данные о модели
        self.mlm_model.summary()
    
    #@tf.function    
    def reset_optimizer(self, lr):
        # Создадим/пересоздадим оптимизатор
        with self.strategy.scope():
            self.enc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    
    @tf.function
    def enc_compute_loss(self, true_vals, predictions, smsk, signif):
        # Т.к. вычисление loss без редукции (Reduction.NONE), результатом применения и той, и другой loss-функции будут тензоры (batch, seq_len).
        # Надо свести всё к (batch), чтобы потом корректно считать distributed-усреднение.
        # Среднюю косинусную меру для предложения (по всей seq_len) считаем с учетом маски.
        # Среднюю mse тоже с учетом маски, т.к. среднее, а не сумму считать будем.
        # На выходе (с учетом distributed-усреднения) будем иметь величину ошибки в рассчете на слово.
        msk_counts = tf.cast( tf.reduce_sum(smsk, -1), dtype=tf.float32 )
        msk_counts = tf.where( msk_counts == 0, tf.ones_like(msk_counts), msk_counts ) # защита от деления на 0
        # косинусная ошибка
        loss_1 = self.loss_object(true_vals, predictions)
        loss_1 = loss_1 + 1                                                            # для удобства интерпретации будем стремить loss к нулю (переенос в диапазон [0,2])
        loss_1 = loss_1 * signif                                                       # снижаем вклад многозначных слов в ошибку (после переноса в [0,2] умножение OK)
        loss_1 = tf.where(smsk == 1, loss_1, tf.zeros_like(loss_1))                    # игнорируем ошибку вне маски
        loss_1 = tf.reduce_sum(loss_1, -1) / msk_counts                                # схлопываем до ошибки в расчете на слово
        # среднеквадратичная ошибка
        loss_2 = self.secondary_loss_object(true_vals, predictions)                    # >= 0
        loss_2 = loss_2 * signif
        loss_2 = tf.where(smsk == 1, loss_2, tf.zeros_like(loss_2))
        loss_2 = tf.reduce_sum(loss_2, -1) / msk_counts
        
        per_example_loss = loss_1 + 0.1 * loss_2
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.GLOBAL_BATCH_SIZE)
    
       
    @tf.function
    def enc_get_loss_and_grads(self, x):
        with tf.GradientTape() as tape:
            x_real = x[0]  # x_real = (batch, seq_len, 2)  -- обучающие примеры
            x_smsk = x[1]  # x_smsk = (batch, seq_len)     -- маска сабсэмплинга
            x_sign = x[2]  # x_sign = (batch, seq_len)     -- требуемый вклад в ошибку
            # выполняем шаг обучения
            # в замаскированных согласно x_smsk позициях модель возвращает PAD !
            y = self.mlm_model( [x_real,x_smsk], training=True )  # y = (batch, seq_len, dims)
            # получим входные данные в виде эмбеддингов
            embedded_x = tf.concat([self.stems_embedder(x_real[:,:,0]), self.sfx_embedder(x_real[:,:,1])], -1)  # embedded_x = (batch, seq_len, dims)
            loss_value = self.enc_compute_loss(embedded_x, y, x_smsk, x_sign)
            #loss_value += tf.math.add_n(mlm_model.losses) # прибавим loss от L2-регуляризаторов
            cat_loss_value = self.enc_compute_loss(embedded_x[:,:,:CATEG_DIMS], y[:,:,:CATEG_DIMS], x_smsk, x_sign)
            ass_loss_value = self.enc_compute_loss(embedded_x[:,:,CATEG_DIMS:LEX_DIMS], y[:,:,CATEG_DIMS:LEX_DIMS], x_smsk, x_sign)
            gra_loss_value = self.enc_compute_loss(embedded_x[:,:,LEX_DIMS:], y[:,:,LEX_DIMS:], x_smsk, x_sign)
        grads = tape.gradient(loss_value, self.mlm_model.trainable_weights)
        return loss_value, grads, cat_loss_value, ass_loss_value, gra_loss_value


    @tf.function
    def enc_train_step(self, x):
        loss_value, grads, cat_loss_value, ass_loss_value, gra_loss_value = self.enc_get_loss_and_grads(x)
        self.enc_optimizer.apply_gradients(zip(grads, self.mlm_model.trainable_weights))
        return loss_value, cat_loss_value, ass_loss_value, gra_loss_value

    @tf.function
    def enc_distributed_train_step(self, x):
        per_replica_losses, cat_loss, ass_loss, gra_loss = self.strategy.run(self.enc_train_step, args=(x,))
        return [ self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None),
                 self.strategy.reduce(tf.distribute.ReduceOp.SUM, cat_loss, axis=None),
                 self.strategy.reduce(tf.distribute.ReduceOp.SUM, ass_loss, axis=None),
                 self.strategy.reduce(tf.distribute.ReduceOp.SUM, gra_loss, axis=None) ]


    def run(self, rst_lr):
        # цикл обучения
        start_time = time.perf_counter()
        timeout_flag = False
        gstep = -1
        # цикл по эпохам
        for epoch in range(5):
            print('\nStart of epoch {}'.format(epoch))
            start_epoch_time = time.perf_counter()
            enc_epoch_loss, epoch_cat_loss, epoch_ass_loss, epoch_gra_loss = 0.0, 0.0, 0.0, 0.0
            enc_step = 0
            # цикл по батчам датасета
            for step, x in enumerate(self.dataset):
                gstep += 1
                
                # выполняем шаг обучения
                #self.mlm_model.get_layer('lle').setup_dims_bypass([True,False,True])
                enc_loss_value, cat_loss, ass_loss, gra_loss = self.enc_distributed_train_step( x )
                enc_epoch_loss += enc_loss_value
                epoch_cat_loss += cat_loss
                epoch_ass_loss += ass_loss
                epoch_gra_loss += gra_loss
                enc_step += 1
                
                # выводим проткол через каждые N батчей.
                if step % 250 == 0:
                    print('Step No:   {}'.format(step))
                    #enc_opt_current_step = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, self.enc_optimizer.iterations, axis=None).numpy()
                    #print('Current encoder learning rate:      {:.5f}'.format(self.enc_lr_schedule(enc_opt_current_step)))
                    print('Enc. training loss:                 {:.4f} ({:.4f} / {:.4f} / {:.4f})'.format(enc_epoch_loss/enc_step, epoch_cat_loss/enc_step, epoch_ass_loss/enc_step, epoch_gra_loss/enc_step))
                    print('Enc. training loss for last batch:  {:.4f}'.format(enc_loss_value))
                    epoch_samples_count = (step+1)*self.GLOBAL_BATCH_SIZE
                    print('Seen so far:                        {} samples'.format(epoch_samples_count))
                    eptime = time.perf_counter() - start_epoch_time
                    print('Speed:                              {:.1f} samples per second'.format(epoch_samples_count/eptime))
                    etime = time.perf_counter() - start_time
                    print('Execution time:                     {:.1f} sec == {:.1f} hours'.format(etime, etime/3600))
                    # критерий остановки по времени обучения
                    if etime > self.time:
                        print('Stop training by timeout')
                        timeout_flag = True
                        break

                if self.reset_optimizer_frequently and (gstep < 4000) and (gstep % 2 == 0):
                    #print('Reset optimizer. LR={}'.format(rst_lr))
                    self.reset_optimizer(lr=rst_lr)
                                    
            if timeout_flag:
                break
           
        # сохраняем все веса LowLevelEncoder
        self.mlm_model.get_layer('lle').save_own_weights('lle_weights')




def main():
    parser = argparse.ArgumentParser(description='mlm_training')
    parser.add_argument('mode', type=str, choices=['new', 'continue'], help='training mode')
    parser.add_argument('time', type=int, default=24*60*60, help='time to train (sec)')
    parser.add_argument('--rst_optimizer', action='store_true', help='reset the optimizer frequently')
    parser.add_argument('--rnd_dataset', action='store_true', help='randomize dataset files list')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--drp0', type=float, default=0.25, help='dropout 0')
    parser.add_argument('--drp1', type=float, default=0.5, help='dropout 1')
    parser.add_argument('--drpf', type=float, default=0.15, help='dropout 2')
    parser.add_argument('--batch', type=int, default=1024, help='batch')
    args = parser.parse_args()
    if not args.mode or not args.time:
        parser.print_help()
        sys.exit(-1)

    mtc = MlmTrainingController(args.mode, args.time, args.rst_optimizer, args.rnd_dataset, args.batch)
    mtc.prepare(args)
    mtc.reset_optimizer(lr=args.lr)
    mtc.run(rst_lr=args.lr)


if __name__ == "__main__":
    main()



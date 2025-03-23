
from embeddings_reader import EmbeddingsReader

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


# class AttCell(tf.keras.layers.Layer):
#     def __init__(self, name='AttCell', **kwargs):
#         super(AttCell, self).__init__(name=name, **kwargs)
# #        self.drp0 = tf.keras.layers.Dropout(0.1)
#         self.dns0 = tf.keras.layers.Dense(64, activation='tanh')
#         self.drp1 = tf.keras.layers.Dropout(0.05)
#         self.dns1 = tf.keras.layers.Dense(32, activation='tanh')
#       
#     def call(self, input, training): # преобразует (d0, d1, ..., K) -> (d0, d1, ..., 64)
# #        x = self.drp0(input, training=training)
#         x = self.dns0(input)
#         x = self.drp1(x, training=training)
#         x = self.dns1(x)
#         return x
#   
#     def get_name2sublayer_dict(self):
#         return {
#                 'dns0': self.dns0,
#                 'dns1': self.dns1,
#                }
#       
#     def fill_weights_dict(self, wdict, relative_name, dir = 'lle_weights'):
#         for k, v in self.get_name2sublayer_dict().items():
#             w = v.get_weights()
#             for i in range(len(w)):
#                 wdict['{}.{}_w{}'.format(relative_name, k, i)] = w[i]
#     def restore_weights(self, wdict, relative_name, dir = 'lle_weights'):
#         for k, v in self.get_name2sublayer_dict().items():
#             w_list = []
#             for i in range(len(v.get_weights())):
#                 w_list.append( wdict['{}.{}_w{}'.format(relative_name, k, i)] )
#             v.set_weights( w_list )
#   
#       
#   
# class AttWindow10(tf.keras.layers.Layer):
#     def __init__(self, name='AttWindow', **kwargs):
#         super(AttWindow10, self).__init__(name=name, **kwargs)
#         self.ac_00 = AttCell('AttCell_00')
#         self.ac_01 = AttCell('AttCell_01')
#         self.ac_02 = AttCell('AttCell_02')
#         self.ac_03 = AttCell('AttCell_03')
#         self.ac_04 = AttCell('AttCell_04')
#         self.ac_05 = AttCell('AttCell_05')
#         self.ac_06 = AttCell('AttCell_06')
#         self.ac_07 = AttCell('AttCell_07')
#         self.ac_08 = AttCell('AttCell_08')
#         self.ac_09 = AttCell('AttCell_09')
#         self.dists = self.make_smoothed_distance_matrix(10)
#         self.dists = tf.expand_dims(self.dists, 0)
#         self.dists = tf.expand_dims(self.dists, -1)
#   
#     def pairwise_window(self, t):
#         t1 = tf.expand_dims(t, -2)
#         t1 = tf.repeat(t1, repeats=[10], axis=-2)
#         t2 = tf.expand_dims(t, -3)
#         t2 = tf.repeat(t2, repeats=[10], axis=-3)
#         return tf.concat([t1, t2], axis=-1)
#   
#     # построение сглаженной матрицы расстояний
#     def make_smoothed_distance_matrix(self, seq_len):
#         return tf.tanh( tf.constant( [[ (x - y)/5 for x in range(seq_len)] for y in range(seq_len)], dtype=tf.float32 ) )
#       
#     def apply_att_cell(self, cell, input):
#         x = cell(input)                             # (subnet_batch, window_size, 64)
#         x = tf.reduce_sum(x, -2)                    # (subnet_batch, 64)
#         x = tf.expand_dims(x, -2)                   # (subnet_batch, 1, 64)
#         return x
#            
#     def call(self, input, training):
#         ishape = tf.shape(input)                                        # (subnet_batch, window_size=10, emb_dims)
#         subnet_batch_size = ishape[0]
#         pinput = self.pairwise_window(input)                            # (subnet_batch, window_size, window_size, 2*emb_dims)
#         dists = tf.repeat(self.dists, repeats=[subnet_batch_size], axis=0)
#         pinput = tf.concat([pinput, dists], -1)                         # (subnet_batch, window_size, window_size, 2*emb_dims+1)
#         x0 = self.apply_att_cell(self.ac_00, pinput[:,0,:,:])           # (subnet_batch, 1, 64)
#         x1 = self.apply_att_cell(self.ac_01, pinput[:,1,:,:])
#         x2 = self.apply_att_cell(self.ac_02, pinput[:,2,:,:])
#         x3 = self.apply_att_cell(self.ac_03, pinput[:,3,:,:])
#         x4 = self.apply_att_cell(self.ac_04, pinput[:,4,:,:])
#         x5 = self.apply_att_cell(self.ac_05, pinput[:,5,:,:])
#         x6 = self.apply_att_cell(self.ac_06, pinput[:,6,:,:])
#         x7 = self.apply_att_cell(self.ac_07, pinput[:,7,:,:])
#         x8 = self.apply_att_cell(self.ac_08, pinput[:,8,:,:])
#         x9 = self.apply_att_cell(self.ac_09, pinput[:,9,:,:])
#         return tf.concat([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9], -2)
#   
#     def get_name2sublayer_dict(self):
#         return {
#                 'AttCell_00': self.ac_00,
#                 'AttCell_01': self.ac_01,
#                 'AttCell_02': self.ac_02,
#                 'AttCell_03': self.ac_03,
#                 'AttCell_04': self.ac_04,
#                 'AttCell_05': self.ac_05,
#                 'AttCell_06': self.ac_06,
#                 'AttCell_07': self.ac_07,
#                 'AttCell_08': self.ac_08,
#                 'AttCell_09': self.ac_09,
#                }
#       
#     def fill_weights_dict(self, wdict, relative_name, dir = 'lle_weights'):
#         for k, v in self.get_name2sublayer_dict().items():
#             v.fill_weights_dict(wdict, '{}.{}'.format(relative_name, k), dir)
#     def restore_weights(self, wdict, relative_name, dir = 'lle_weights'):
#         for k, v in self.get_name2sublayer_dict().items():
#             v.restore_weights(wdict, '{}.{}'.format(relative_name, k), dir)



# # Вспомогательный класс контекстуализатора
# class CtxNetHelper(tf.keras.layers.Layer):
#     def __init__(self, output_dims, input_drop_rate, lstm_units, window_size, name='CtxNetHelper', **kwargs):
#         super(CtxNetHelper, self).__init__(name=name, **kwargs)
#         self.window_size = window_size
#         # создаем сеть
#         self.drp_li = tf.keras.layers.Dropout(input_drop_rate)
#         self.drp_ri = tf.keras.layers.Dropout(input_drop_rate)
# 
# #        self.aw10 = AttWindow10()
#         self.lstm0 = tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(200, return_sequences=True), merge_mode='concat' )
# 
#         self.drp1 = tf.keras.layers.Dropout(0.05)
#         self.lctx_lstm = tf.keras.layers.LSTM(lstm_units)
#         self.rctx_lstm = tf.keras.layers.LSTM(lstm_units)
# 
#         self.drp_00 = tf.keras.layers.Dropout(0.05)
#         self.dns_00 = tf.keras.layers.Dense(output_dims+20, activation='tanh')
# 
#         self.drp_pf = tf.keras.layers.Dropout(0.05)
#         self.dns_final = tf.keras.layers.Dense(output_dims)
# 
#     def call(self, lctx, rctx, training):
#         # shape of lctx/rctx = (quasi_batch, window, input_dims)
# 
#         # базовый drop входных данных
#         lctx = self.drp_li(lctx, training=training)
#         rctx = self.drp_ri(rctx, training=training)
#          
# #         ctx = tf.concat([lctx,rctx], -2)
# #         ctx = self.aw10(ctx)
# #         ctx = self.drp1(ctx, training=training)
# #         lctx2 = ctx[:, :self.window_size, :]
# #         rctx2 = ctx[:, self.window_size:, :] 
# 
#         ctx = tf.concat([lctx,rctx], -2)
#         ctx = self.lstm0(ctx)
#         ctx = self.drp1(ctx, training=training)
#         lctx2 = ctx[:, :self.window_size, :]
#         rctx2 = ctx[:, self.window_size:, :] 
#         
#         lctx = tf.concat([lctx, lctx2], -1)
#         rctx = tf.concat([rctx, rctx2], -1)
# 
#         rctx = tf.reverse(rctx, [-2])
#         # рекуррентная обработка
#         l_inf = self.lctx_lstm( lctx, training=training )
#         r_inf = self.rctx_lstm( rctx, training=training )
#         x = tf.concat([l_inf, r_inf], -1)
# 
#         x = self.drp_00(x, training=training)
#         x = self.dns_00(x)
#         
#         x = self.drp_pf(x, training=training)
#         return self.dns_final(x) # (quasi_batch, output_dims)
#              
#              
#     def get_name2sublayer_dict(self):
#         return {
#                 'lctx_lstm': self.lctx_lstm,
#                 'rctx_lstm': self.rctx_lstm,
#                 'lstm0': self.lstm0,
#                 'dns_f': self.dns_final,
#                 'dns_00': self.dns_00,
#                }
#              
#     def save_own_weights(self, filename):
#         n2sl = self.get_name2sublayer_dict()
#         wdict = {}
#         for k,v in n2sl.items():
#             w = v.get_weights()
#             for i in range(len(w)):
#                 wdict[k+'_w{}'.format(i)] = w[i]
# #        self.aw10.fill_weights_dict(wdict, 'aw10', dir)
#         np.savez(filename, **wdict)
#     def load_own_weights(self, filename):
#         wdict = np.load(filename+'.npz')
#         n2sl = self.get_name2sublayer_dict()
#         for k,v in n2sl.items():
#             w_list = []
#             for i in range(len(v.get_weights())):
#                 w_list.append( wdict[k+'_w{}'.format(i)] )
#             v.set_weights( w_list )
# #        self.aw10.restore_weights(wdict, 'aw10', dir)
 
 
   
# Слой контекстуализации
class Ctxer(tf.keras.layers.Layer):
    def __init__(self, cat_dims, ass_dims, gra_dims, stems_count, sfx_count, return_attention, drp0v=0.1, drp1v=0.4, drpfv=0.05, name="Ctxer", **kwargs):
        super(Ctxer, self).__init__(name=name, **kwargs)
        self.cat_dims = cat_dims
        self.ass_dims = ass_dims
        self.stems_dims = cat_dims + ass_dims
        self.gra_dims = gra_dims
        self.sfx_dims = gra_dims
        self.dims = cat_dims + ass_dims + gra_dims
        self.stems_count = stems_count
        self.sfx_count = sfx_count
        self.return_attention = return_attention
        self.window_size = 7
        self.dws = self.window_size * 2 

        # настраиваем паттерн для oov-маскирования контекста        
#        fwd_wnd_distr = tf.constant([[ [0.02], [0.02], [0.02], [0.02], [0.02] ]], dtype=tf.float32)        # (1, 5, 1)
#        fwd_wnd_distr = tf.constant([[ [0.02], [0.02], [0.02], [0.03], [0.05] ]], dtype=tf.float32)        # (1, 5, 1)
        fwd_wnd_distr = tf.constant([[ [0.02], [0.02], [0.02], [0.02], [0.02], [0.03], [0.05] ]], dtype=tf.float32)        # (1, 7, 1)
        bwd_wnd_distr = tf.reverse(fwd_wnd_distr, [-2]) 
        prt_distt = [[ tf.repeat([1, 0.5], [self.stems_dims, self.sfx_dims]) ]]  # (1, 1, input_dims)
        self.l_rand_mask = tf.math.multiply( fwd_wnd_distr, prt_distt )      # (1, 5, input_dims)
        self.r_rand_mask = tf.math.multiply( bwd_wnd_distr, prt_distt )

        # расстояния (в окне) до предсказываемого слова (с разными скоростями убывания)
        d11 = tf.reverse(tf.constant( [[ [pow((x+0.1)/4.1, 3)] for x in range(self.window_size)]], dtype=tf.float32 ), [-2])
        d12 = tf.reverse(tf.constant( [[ [pow((x+0.2)/5, 2)] for x in range(self.window_size)]], dtype=tf.float32 ), [-2])
        d2 = tf.tanh(tf.constant( [[[(self.window_size-x-0.7)/2] for x in range(self.window_size)]], dtype=tf.float32 ))
        d3 = tf.tanh(tf.constant( [[[(self.window_size-x-0.2)/4] for x in range(self.window_size)]], dtype=tf.float32 ))
        d4 = tf.constant( [[[(self.window_size-x)/10] for x in range(self.window_size)]], dtype=tf.float32 )
        self.dists = tf.concat([d11,d12,d2,d3,d4], -1)

        if drp0v > 0:
            self.drp_li = tf.keras.layers.Dropout(drp0v)
            self.drp_ri = tf.keras.layers.Dropout(drp0v)
        else:
            self.drp_li = None
            self.drp_ri = None
        pre_dims = 40 # preview для противоположной стороны (можно и 100)
        self.pre_lctx_lstm = tf.keras.layers.LSTM(pre_dims)
        self.pre_rctx_lstm = tf.keras.layers.LSTM(pre_dims)
#         pv_dims = 40 # preview для своей стороны
#         self.pv_lctx_lstm = tf.keras.layers.LSTM(pv_dims)
#         self.pv_rctx_lstm = tf.keras.layers.LSTM(pv_dims)
        if drp1v > 0:
            self.drp_bl = tf.keras.layers.Dropout(drp1v)
            self.drp_br = tf.keras.layers.Dropout(drp1v)
#             self.drp_pvl = tf.keras.layers.Dropout(drp1v)
#             self.drp_pvr = tf.keras.layers.Dropout(drp1v)
        else:
            self.drp_bl = None
            self.drp_br = None
#             self.drp_pvl = None
#             self.drp_pvr = None
        lstm_units = 700
        self.lctx_lstm = tf.keras.layers.LSTM(lstm_units)
        self.rctx_lstm = tf.keras.layers.LSTM(lstm_units)
        if drpfv > 0:
            self.drp_pf = tf.keras.layers.Dropout(drpfv)
        else:
            self.drp_pf = None
        self.dns_final = tf.keras.layers.Dense(self.dims)
        
        self.make_att_subnet()
 
        
    def build(self, input_shape):
        # !!! слои необходимо создавать в build, а не конструкторе, 
        #     иначе не заработает сохранение моделей, содержащих эту, в формате h5 (назначение имен слоям нужно)
        super(Ctxer, self).build(input_shape)

    def make_att_subnet(self):
        self.att_drp_1 = tf.keras.layers.Dropout(0.1, name = 'att_drp_1')
        self.att_dns_1 = tf.keras.layers.Dense(256, activation='tanh', name = 'att_dns_1') # 160
        self.att_drp_2 = tf.keras.layers.Dropout(0.05, name = 'att_drp_2')
        self.att_dns_2 = tf.keras.layers.Dense(128, activation='tanh', name = 'att_dns_2') # 80
        self.att_drp_3 = tf.keras.layers.Dropout(0.05, name = 'att_drp_3')
#        self.att_dns_final = tf.keras.layers.Dense(1, activation='sigmoid', name = 'att_dns_final') 
        self.att_dns_final = tf.keras.layers.Dense(self.ass_dims, activation='sigmoid', name = 'att_dns_final') 
    def apply_att_subnet(self, input, training):
        x = self.att_drp_1(input, training=training)
        x = self.att_dns_1(x)
        x = self.att_drp_2(x, training=training)
        x = self.att_dns_2(x)
        x = self.att_drp_3(x, training=training)
        return self.att_dns_final(x)     
            
    def get_name2sublayer_dict(self):
        return {
                'pre_lctx_lstm': self.pre_lctx_lstm,
                'pre_rctx_lstm': self.pre_rctx_lstm,
#                 'pv_lctx_lstm': self.pv_lctx_lstm,
#                 'pv_rctx_lstm': self.pv_rctx_lstm,
                'lctx_lstm': self.lctx_lstm,
                'rctx_lstm': self.rctx_lstm,
                'dns_final': self.dns_final,
                'att_dns_1': self.att_dns_1,
                'att_dns_2': self.att_dns_2,
                'att_dns_final': self.att_dns_final,
               }
    def save_own_weights(self, dir = 'lle_weights'):
        wdict = {}
        for k,v in self.get_name2sublayer_dict().items():
            w = v.get_weights()
            for i in range(len(w)):
                wdict[k+'_w{}'.format(i)] = w[i]
        np.savez(os.path.join(dir, 'lle_ctxer'), **wdict)
    def load_own_weights(self, dir = 'lle_weights'):
        wdict = np.load(os.path.join(dir, 'lle_ctxer') +'.npz')
        n2ksl = self.get_name2sublayer_dict()
        for k,v in n2ksl.items():
            w_list = []
            for i in range(len(v.get_weights())):
                w_list.append( wdict[k+'_w{}'.format(i)] )
            v.set_weights( w_list )
             
             
 
    def call(self, input, training):
        x_joined = input[0]
        line_msk = input[1]
 
        xj_shape = tf.shape(x_joined)
        batch_size, seq_len = xj_shape[0], xj_shape[1]
 
        # найдем координаты слов в тензоре
        w_indices = tf.where( line_msk == 1 )
 
        # преобразуем входную последовательность в наборы левых и правых контекстов (длиной window_size) для каждого слова
        padding_block = tf.fill([batch_size, self.window_size, self.dims], EmbeddingsReader.PAD_FILLER)
        padded_input = tf.concat([padding_block, x_joined, padding_block], -2) 
        padded_input = tf.expand_dims(padded_input, -2)
        sequences = tf.image.extract_patches(images=padded_input, sizes=[1, self.window_size, 1, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='VALID')
        lctx = tf.reshape( sequences[:, :-(self.window_size+1), :, :], [batch_size, seq_len, self.window_size*self.dims] )
        rctx = tf.reshape( sequences[:, (self.window_size+1):, :, :], [batch_size, seq_len, self.window_size*self.dims] )
        
        # для обработки возьмем только контексты слов, участвующих в обучении/выводе (не отфильтрованных и не pad-ов)
        lctx = tf.gather_nd(lctx, w_indices)
        rctx = tf.gather_nd(rctx, w_indices)
        lctx = tf.reshape(lctx, [-1, self.window_size, self.dims])    # shape of lctx/rctx = (quasi_batch, window, input_dims)
        rctx = tf.reshape(rctx, [-1, self.window_size, self.dims])
        
        # случайная маскировка контекстов
        # чем ближе к предсказываемому слову, тем выше шанс быть превращенным в oov
        # цель -- больше вовлечь дальний контекст в процесс предсказания
        if training:
            lwc = tf.shape(lctx)[0]
            l_rnd = tf.random.uniform([lwc, self.window_size], minval=0.0, maxval=1.0, dtype=tf.float32)
            l_rnd = tf.expand_dims(l_rnd, -1)
            lctx = tf.where( tf.math.logical_and( l_rnd < self.l_rand_mask, lctx != [[[EmbeddingsReader.PAD_FILLER]]]), 
                                 [[[EmbeddingsReader.OOV_FILLER]]], 
                                 lctx )
            rwc = tf.shape(rctx)[0]
            r_rnd = tf.random.uniform([rwc, self.window_size], minval=0.0, maxval=1.0, dtype=tf.float32)
            r_rnd = tf.expand_dims(r_rnd, -1)
            rctx = tf.where( tf.math.logical_and( r_rnd < self.r_rand_mask, rctx != [[[EmbeddingsReader.PAD_FILLER]]]), 
                                 [[[EmbeddingsReader.OOV_FILLER]]], 
                                 rctx )

        # разворачиваем правые контексты по направлению к предсказываемому слову
        rctx = tf.reverse(rctx, [-2])                               # (N, WINDOW_SIZE, DIMS)
        
        # дополним контекст общим тематическим вектором
        ctxs_counts = tf.reduce_sum(line_msk, -1)                # кол-во слов в каждом предложении
        x_selected = tf.gather_nd(x_joined, w_indices)           # список всех значимых слов (без pad)
        att_w = self.apply_att_subnet(x_selected, training)      # для каждого слова определяем его значимость в качестве источника тематической информации
        t = x_selected[:, self.cat_dims:self.stems_dims]         # теперь возьмем ассоц.части всех значимых слов
        t = t * att_w                                            # взвесим их
        t = tf.RaggedTensor.from_row_lengths(t, ctxs_counts)     # сгруппируем в предложения
        t = tf.reduce_mean(t, axis=-2)                           # найдем тематический вектор для предложения путем усреднения взвешенных ассоц.частей всех слов
        t = tf.repeat(t, ctxs_counts, -2)                        # растиражируем по числу слов в предложении
        

        # информация о расстояниях (от контекстных слов в окне до предсказываемого)
        quasi_batch_size = tf.shape(lctx)[0]
        xd = tf.repeat(self.dists, repeats=[quasi_batch_size], axis=0) # (N, WINDOW_SIZE, DISTS_CNT)
        lctx = tf.concat([lctx, xd], -1)                               # (N, WINDOW_SIZE, DIMS+DISTS_CNT)
        rctx = tf.concat([rctx, xd], -1)

        if self.drp_li:
            lctx = self.drp_li(lctx, training=training)
            rctx = self.drp_ri(rctx, training=training)

        # получаем предварительное свернутое представление о левом и правом контексте
        pl_inf = self.pre_lctx_lstm( lctx, training=training )      # (N, PRE_LSTM_SIZE)
        pr_inf = self.pre_rctx_lstm( rctx, training=training )
        if self.drp_bl:
            pl_inf = self.drp_bl(pl_inf, training=training)
            pr_inf = self.drp_br(pr_inf, training=training)
        pl_inf = tf.expand_dims(pl_inf, -2)                         # (N, 1, PRE_LSTM_SIZE)
        pr_inf = tf.expand_dims(pr_inf, -2)
        pl_inf = tf.tile(pl_inf, [1,self.window_size,1])            # (N, WINDOW_SIZE, PRE_LSTM_SIZE)
        pr_inf = tf.tile(pr_inf, [1,self.window_size,1])

#         pvl_inf = self.pv_lctx_lstm( lctx, training=training )      # (N, PV_LSTM_SIZE)
#         pvr_inf = self.pv_rctx_lstm( rctx, training=training )
#         if self.drp_pvl:
#             pvl_inf = self.drp_pvl(pvl_inf, training=training)
#             pvr_inf = self.drp_pvr(pvr_inf, training=training)
#         pvl_inf = tf.expand_dims(pvl_inf, -2)                       # (N, 1, PV_LSTM_SIZE)
#         pvr_inf = tf.expand_dims(pvr_inf, -2)
#         pvl_inf = tf.tile(pvl_inf, [1,self.window_size,1])          # (N, WINDOW_SIZE, PV_LSTM_SIZE)
#         pvr_inf = tf.tile(pvr_inf, [1,self.window_size,1])
        
        # расширяем левый контекст свернутым представлением о правом контексте,
        #   а правый контекст -- свернутым представлением о левом контексте
        lctx = tf.concat([lctx, pr_inf], -1)
        rctx = tf.concat([rctx, pl_inf], -1)
        # применяем основные lstm для получения окончательных представлений для левого и правого контекстов
        l_inf = self.lctx_lstm( lctx, training=training )
        r_inf = self.rctx_lstm( rctx, training=training )
        # конкатенируем
        x = tf.concat([l_inf, r_inf, t], -1)
        # получаем результирующее предсказание
        if self.drp_pf:
            x = self.drp_pf(x, training=training)
        x = self.dns_final(x)
 
        # поставим слова в их позиции в предложениях
        x = tf.scatter_nd(w_indices, x, [batch_size, seq_len, self.dims])
        cat_x = x[:,:,:self.cat_dims]
        ass_x = x[:,:,self.cat_dims:self.stems_dims]
        gra_x = x[:,:,self.stems_dims:]
         
        if self.return_attention:
            # return cat_x, ass_x, gra_x, tf.scatter_nd(w_indices, tf.reshape(att_w, [-1]), [batch_size, seq_len])
            #return cat_x, ass_x, gra_x, tf.scatter_nd(w_indices, tf.reshape(tf.reduce_max(att_w, axis=-1), [-1]), [batch_size, seq_len])
            return cat_x, ass_x, gra_x, tf.scatter_nd(w_indices, tf.reshape(tf.reduce_mean(att_w, axis=-1), [-1]), [batch_size, seq_len])
        else:
            return cat_x, ass_x, gra_x
         
 
 
 
# Модель "низкоуровневого энкодера"
# Включает в себя слои статических эбмеддингов, низкоуровневой контекстуализации и смешивания
class LowLevelEncoder(tf.keras.Model):
    def __init__( self, cat_dims, ass_dims, gra_dims, stems_count, sfx_count, 
                  ctxer_training=False, balance_training_cat=False, balance_training_ass=False, balance_training_gra=False, return_attention=False,
                  drp0v=0.1, drp1v=0.4, drpfv=0.05,
                  hide_sfx=False,
                  name="LowLevelEncoder", **kwargs ):
        super().__init__(name=name, **kwargs)
        self.cat_dims, self.ass_dims, self.gra_dims = cat_dims, ass_dims, gra_dims
        self.stems_dims, self.sfx_dims = cat_dims + ass_dims, gra_dims
        self.dims = cat_dims + ass_dims + gra_dims
        self.stems_count = stems_count
        self.sfx_count = sfx_count
        self.bypass = False   # режим отключения контекстуализатора: на выход пробрасываются статические представления
        self.ctxer_training = ctxer_training   # режим обучения контекстуализатора (выключает смешивание со статическими эмбеддингами)
        self.balance_training_cat = balance_training_cat   # режим обучения балансов
        self.balance_training_ass = balance_training_ass   # режим обучения балансов
        self.balance_training_gra = balance_training_gra
        self.return_attention = return_attention
        self.hide_sfx = hide_sfx #режим отключения выдачи наружу представления для грамматической части
        self.stems_emb_layer = tf.keras.layers.Embedding(self.stems_count, self.stems_dims, trainable=False, name='stems_embeddings')
        self.sfx_emb_layer   = tf.keras.layers.Embedding(self.sfx_count, self.sfx_dims, trainable=False, name='sfx_embeddings')
        self.cat_balance     = tf.keras.layers.Embedding(self.stems_count, 1, 
                                                         embeddings_initializer=tf.keras.initializers.Constant(0.9), 
                                                         #embeddings_constraint=lambda x: tf.clip_by_value(x, 0.01, 0.99), 
                                                         trainable=balance_training_cat, name='cat_balances')
        self.ass_balance     = tf.keras.layers.Embedding(self.stems_count, 1, 
                                                         embeddings_initializer=tf.keras.initializers.Constant(0.9), 
                                                         #embeddings_constraint=lambda x: tf.clip_by_value(x, 0.01, 0.99), 
                                                         trainable=balance_training_ass, name='ass_balances')
        self.gra_balance     = tf.keras.layers.Embedding(self.sfx_count, 1, 
                                                         embeddings_initializer=tf.keras.initializers.Constant(0.9), 
                                                         #embeddings_constraint=lambda x: tf.clip_by_value(x, 0.01, 0.99), 
                                                         trainable=balance_training_gra, name='gra_balances')
        self.ctxer = Ctxer(cat_dims, ass_dims, gra_dims, stems_count, sfx_count, return_attention, drp0v, drp1v, drpfv, trainable=ctxer_training, name='ctxer')
        self.dims_bypass = tf.Variable([False,False,False], trainable=False)
     
    def load_embeddings(self, vm):
        assert vm.stems_size == self.stems_dims
        assert vm.sfx_size == self.sfx_dims
        self.stems_emb_layer.set_weights([vm.stems_embs])
        self.sfx_emb_layer.set_weights([vm.sfx_embs])
    def get_balances_dict(self):
        return {
                'cat_balances': self.cat_balance,
                'ass_balances': self.ass_balance,
                'gra_balances': self.gra_balance
                # спец.обработка с clip для балансов... нельзя сюда добавлять другие веса
               }
    def save_own_weights(self, dir = 'lle_weights'):
        self.ctxer.save_own_weights(dir)
        wdict = {}
        bdict = self.get_balances_dict()
        for k,v in bdict.items():
            w = v.get_weights()
            if not w:
                wdict[k] = np.full( (v.input_dim, 1), 0.9 )
            else:
                wdict[k] = np.clip( v.get_weights()[0], 0.01, 0.99 )
        np.savez(os.path.join(dir, 'balances'), **wdict)
    def load_own_weights(self, dir = 'lle_weights'):
        self.ctxer.load_own_weights(dir)
        if not self.ctxer_training: # в конфигурации ctxer_training эти веса не конструируются (см. код call)
                wdict = np.load(os.path.join(dir, 'balances') +'.npz')
                bdict = self.get_balances_dict()
                for k,v in bdict.items():
                    v.set_weights( [wdict[k]] )
         
    def switch_bypass(self, value):
        self.bypass = value
    def setup_dims_bypass(self, value):
        self.dims_bypass.assign(value)
    def switch_ctxer_trainable(self, value):
        self.ctxer.trainable = value
    def switch_balancer_trainable(self, value):
        self.cat_balance.trainable = value
        self.ass_balance.trainable = value
        self.gra_balance.trainable = value
    

    def call(self, input, training):
         
        if self.ctxer_training:
            # в тренировочном режиме в input приходит пара (результат_токенизации, маска)
            input_toks = input[0]
            line_msk = tf.cast( input[1], dtype=tf.int32 )  # (batch, seq_len)
        else:
            # в режиме вывода в input приходит только результат_токенизации
            input_toks = input
            # построим pad-маску
            line_msk = tf.math.not_equal(input_toks[:,:,0], EmbeddingsReader.PAD_IDX)  # (batch, seq_len)
            line_msk = tf.cast( line_msk, dtype=tf.int32 )
         
        input_shape = tf.shape(input_toks)
        batch_size, seq_len = input_shape[0], input_shape[1]
         
        # перекодируем индексы в эмбеддинги
        x_stems = self.stems_emb_layer(input_toks[:,:,0])
        x_sfxs = self.sfx_emb_layer(input_toks[:,:,1])
        x_joined = tf.concat([x_stems, x_sfxs], -1)
         
        if self.bypass:
            return x_joined

        if self.return_attention:
            cat_x, ass_x, gra_x, att = self.ctxer([x_joined, line_msk, self.dims_bypass], training=self.ctxer_training)
        else:
            cat_x, ass_x, gra_x = self.ctxer([x_joined, line_msk, self.dims_bypass], training=self.ctxer_training)
 
        # смешивание с исходными статическими эмбеддингами        
        if not self.ctxer_training:
            # получаем текущие коэффициенты баланса
            cat_ratio = self.cat_balance(input_toks[:,:,0])
            ass_ratio = self.ass_balance(input_toks[:,:,0])
            gra_ratio = self.gra_balance(input_toks[:,:,1])
            # обход в связи с ошибкой constraint у слоя embeddings
            if self.balance_training_cat:
                cat_ratio = tf.clip_by_value(cat_ratio, 0.01, 0.99)
            if self.balance_training_ass:
                ass_ratio = tf.clip_by_value(ass_ratio, 0.01, 0.99)
            if self.balance_training_gra:
                gra_ratio = tf.clip_by_value(gra_ratio, 0.01, 0.99)
            # смешивание со статическими представлениями
            cat_x = tf.reduce_sum( tf.stack([cat_x*cat_ratio, x_joined[:,:,:self.cat_dims]*(1.0-cat_ratio)], -2), -2 )  # weighted average
            ass_x = tf.reduce_sum( tf.stack([ass_x*ass_ratio, x_joined[:,:,self.cat_dims:self.stems_dims]*(1.0-ass_ratio)], -2), -2 )
            gra_x = tf.reduce_sum( tf.stack([gra_x*gra_ratio, x_joined[:,:,self.stems_dims:]*(1.0-gra_ratio)], -2), -2 )
        # собираем единый вектор
        if not self.hide_sfx:
            result = tf.concat([cat_x, ass_x, gra_x], -1)
        else:
            result = tf.concat([cat_x, ass_x], -1) # для обучения балансам (экспериментальное)
        # маскировка отступов
        result = tf.where( tf.equal(tf.expand_dims(line_msk,-1), 0), tf.ones_like(result)*EmbeddingsReader.PAD_FILLER, result )
        # выдача результата
        if self.return_attention:
            return result, att
        else:
            return result



# # Заглушка энкодера (просто проброс статических эмбеддингов)
# class LowLevelEncoderStub(tf.keras.Model):
#     def __init__(self, cat_dims, ass_dims, gra_dims, stems_count, sfx_count, name="LowLevelEncoderStub", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.stems_dims, self.sfx_dims = cat_dims + ass_dims, gra_dims
#         self.stems_count = stems_count
#         self.sfx_count = sfx_count
#         self.stems_emb_layer = tf.keras.layers.Embedding(self.stems_count, self.stems_dims, trainable=False, name='stems_embeddings')
#         self.sfx_emb_layer   = tf.keras.layers.Embedding(self.sfx_count, self.sfx_dims, trainable=False, name='sfx_embeddings')
#     
#     def load_embeddings(self, vm):
#         assert vm.stems_size == self.stems_dims
#         assert vm.sfx_size == self.sfx_dims
#         self.stems_emb_layer.set_weights([vm.stems_embs])
#         self.sfx_emb_layer.set_weights([vm.sfx_embs])
#     def save_own_weights(self, dir = 'lle_weights'):
#         pass
#     def load_own_weights(self, dir = 'lle_weights'):
#         pass
#     def setup_balance(self, balance):
#         pass
# 
#     def call(self, input, training):
#         x_stems = self.stems_emb_layer(input[:,:,0])
#         x_sfxs = self.sfx_emb_layer(input[:,:,1])
#         return tf.concat([x_stems, x_sfxs], -1)
        



 
# # код для отладки и самодиагностики
# print('Tensorflow version: {}'.format(tf.version.VERSION))
# print()
#       
# input_data = [
#              [
#                [8,4],
#                [6,9],
#                [2,3],
#                [0,0],
#              ],
#              [
#                [4,12],
#                [30,2],
#                [0,0],
#                [0,0],
#              ]
#              ]
#     
# input_data_msk = [ [1,1,1,0], [1,1,0,0]]                                                  
#                                                       
# tinpd = tf.convert_to_tensor(input_data, dtype=tf.int32)
# tinpdm = tf.convert_to_tensor(input_data_msk, dtype=tf.int32)
# print()
# print('Входные данные:')
# print(tinpd.numpy())
#                                                       
# print()
# print('СОЗДАНИЕ И ИНИЦИАЛИЗАЦИЯ СЛОЯ')                    
# att_layer = LowLevelEncoder(2, 2, 2, 100, 100, ctxer_training=True, balance_training_cat=False, balance_training_ass=False, balance_training_gra=False)
#                                 
# print()
# print('FORWARD PASS')
# loss_fn = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.Adam()
# with tf.GradientTape() as tape:
#     ctx_vecs = att_layer( [tinpd,tinpdm], training=True )
#     print('Результат:')
#     print('Вектора контекстов:')
#     print(ctx_vecs.numpy())
#     loss_value = loss_fn([ [[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[10,10,10,10,10,10]],
#                            [[-2,-2,-2,-2,-2,-2],[-3,-3,-3,-3,-3,-3],[10,10,10,10,10,10],[10,10,10,10,10,10]] ], ctx_vecs) # Compute the loss value
#     print('loss tensor')
#     print(loss_value)



# Максимальная длина предложения
MAX_LEN = 80

# Размер пакета
BATCH_SIZE = 1024
#BATCH_SIZE = 256

# Имя каталога для хранения кэшированных обучающих данных (в виде numpy-массивов)
CACHE_DIR = "mlm_cached_dataset"

# Размерности статических векторов
CATEG_DIMS = 60
ASSOC_DIMS = 40
GRAMM_DIMS = 20
LEX_DIMS = CATEG_DIMS + ASSOC_DIMS
TOTAL_DIMS = LEX_DIMS + GRAMM_DIMS


# # количество шагов прогрева для модели обучения энкодера
# ENC_WARMUP_STEPS = 3000
# # количество шагов, через которое происходит изменение скорости обучения для модели энкодера
# ENC_STEPS_TO_CHANGE = 300
# # минимальная скорость обучения энкодера
# ENC_MIN_LR = 3e-5
# # максимальная скорость обучения энкодера
# ENC_MAX_LR = 1e-3
# # величина изменения скорости обучения на участке "отжига" (каждые ENC_STEPS_TO_CHANGE шагов)
# ENC_CHANGE_VALUE = 5e-6
# 
# # количество шагов до включения обучения модели энкодер-декодер
# ED_DELAY_STEPS = 100000
# # количество шагов прогрева для модели энкодер-декодер
# ED_WARMUP_STEPS = 3000
# # количество шагов до разморозки энкодера после прогрева декодера
# ED_RESTORE_STEPS = 7000


# Конфигурация обучающего примера для тренировки баланса
#ENCODER_FIXED_SEQ_LEN = 10
#DECODER_FIXED_SEQ_LEN = 4
#ENCODER_FIXED_SEQ_LEN = 14
#DECODER_FIXED_SEQ_LEN = 10
ENCODER_FIXED_SEQ_LEN = 12
DECODER_FIXED_SEQ_LEN = 5

# Размерности декодера-трансформера для тренировки баланса
DECO_LAYERS = 2
DECO_DIMS = 512 #256
DECO_HEADS = 8
DECO_FF = 512 #256

#DECO_LAYERS = 4 #2
#DECO_DIMS = 256 #512 #128
#DECO_HEADS = 8
#DECO_FF = 256 #1024 #512 #384

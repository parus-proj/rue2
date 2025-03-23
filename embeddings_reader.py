import numpy as np

# Класс для загрузки векторных представлений в память + добаление специальных эмбеддингов
class EmbeddingsReader(object):
    # Константы
    SPECIAL_EMBEDDINGS_COUNT = 2
    PAD_NAME = '_PAD_'
    OOV_NAME = '_OOV_'
    PAD_IDX = 0
    OOV_IDX = 1
    PAD_FILLER = 10.0
    OOV_FILLER = 0.0
    
    # Конструктор
    # на выходе имеем:
    #   self.stems_count -- размер векторной модели лексических значений
    #   self.sfx_count   -- размер векторной модели суффиксов
    #   self.stems_size  -- размерность векторов лексических значений
    #   self.sfx_size    -- размерность векторов суффиксов
    #   self.stems_embs  -- матрица векторных представлений лексических значений (двумерный numpy-массив)
    #   self.sfx_embs    -- матрица векторных представлений суффиксов (двумерный numpy-массив)
    
    #   self.stems_dict, self.sfx_dict -- для использования внутри
    def __init__(self, embeddings_filename, force_oov_with_sfx=False):
        self.force_oov_with_sfx = force_oov_with_sfx
        
        # Загрузка векторных представлений
        em_stems, v_stems = self.load_lex(embeddings_filename+'.t2lex', embeddings_filename+'.lex')
        em_sfx,   v_sfx   = self.load_w2v(embeddings_filename+'.gramm')
        
        # Вычисляем размеры матриц псевдооснов и суффиксов
        self.stems_size = em_stems.shape[1]
        self.sfx_size = em_sfx.shape[1]
        
        self.stems_count = em_stems.shape[0] + EmbeddingsReader.SPECIAL_EMBEDDINGS_COUNT
        self.sfx_count = em_sfx.shape[0] + EmbeddingsReader.SPECIAL_EMBEDDINGS_COUNT

        # Создаем матрицы        
        self.stems_embs = np.zeros((self.stems_count, self.stems_size), dtype=np.float32)
        self.sfx_embs = np.zeros((self.sfx_count, self.sfx_size), dtype=np.float32)
        
        # Наполняем матрицы
        self.stems_embs[EmbeddingsReader.PAD_IDX, :] = np.full((self.stems_size,), EmbeddingsReader.PAD_FILLER)
        self.sfx_embs[EmbeddingsReader.PAD_IDX, :]   = np.full((self.sfx_size,), EmbeddingsReader.PAD_FILLER)
        self.stems_embs[EmbeddingsReader.OOV_IDX, :] = np.full((self.stems_size,), EmbeddingsReader.OOV_FILLER)
        self.sfx_embs[EmbeddingsReader.OOV_IDX, :]   = np.full((self.sfx_size,), EmbeddingsReader.OOV_FILLER)
        
        offset = EmbeddingsReader.SPECIAL_EMBEDDINGS_COUNT
        for i in range(em_stems.shape[0]):
            self.stems_embs[offset+i, :] = em_stems[i, :]
        for i in range(em_sfx.shape[0]):
            self.sfx_embs[offset+i, :] = em_sfx[i, :]
            
        # Создадим словари для последующих преобразований текста в индексы эмбеддингов
        self.stems_dict = {}
        for word, idx in v_stems.items():
            self.stems_dict[word] = idx + EmbeddingsReader.SPECIAL_EMBEDDINGS_COUNT
        self.sfx_dict = {}
        self.sfx_dict[''] = EmbeddingsReader.OOV_IDX # добавим пустой суффикс с отсылкой к _OOV_
        for idx, word in enumerate(v_sfx):
            self.sfx_dict[word] = idx + EmbeddingsReader.SPECIAL_EMBEDDINGS_COUNT
        
        em_stems, em_sfx = None, None

    def load_lex(self, vocab_fn, embs_fn):
        # считываем сначала отображение из токенов в индексы эмбеддингов
        lex_vocab = {}
        with open(vocab_fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                lex_vocab[line[0]] = int(line[1])
        # считываем матрицу эмбеддингов для представления лексического значения
        with open(embs_fn, 'rb') as f:
            data = f.read()
            # парсим заголовок
            first_line_end = data.find('\n'.encode('utf-8'))
            if first_line_end == -1:
                raise RuntimeError('Loading error: {}'.format(embs_fn))
            first_line_end += 1
            first_line = data[:first_line_end].decode('utf-8')
            header = first_line.strip().split()
            if (len(header) < 2):
                raise RuntimeError('Loading error: {}'.format(embs_fn))
            header = [int(i) for i in header]
            vectors = np.zeros((header[0], header[1]), dtype=np.float32)
            # парсим записи
            cur_pos = first_line_end
            for i in range(header[0]):
                vec_end = cur_pos + header[1]*4
                vectors[i, :] = np.frombuffer(data[cur_pos : vec_end], dtype=np.float32)
                cur_pos = vec_end
            return vectors, lex_vocab

    def load_w2v(self, filename):
        #print('Loading {}...'.format(filename))
        with open(filename, 'rb') as f:
            data = f.read()
            # парсим заголовок
            first_line_end = data.find('\n'.encode('utf-8'))
            if first_line_end == -1:
                raise RuntimeError('Loading error: {}'.format(filename))
            first_line_end += 1
            first_line = data[:first_line_end].decode('utf-8')
            header = first_line.strip().split()
            if (len(header) != 2):
                raise RuntimeError('Loading error: {}'.format(filename))
            header = [int(i) for i in header]
            #print('Dims: {} x {}'.format(*header))
            # создаем массив для данных и словарь
            vocab = []
            vectors = np.zeros((header[0], header[1]), dtype=np.float32)
            # парсим записи
            cur_pos = first_line_end
            space_byte = ' '.encode('utf-8')
            for i in range(header[0]):
                space_pos = data.find(space_byte, cur_pos)
                if space_pos == -1:
                    raise RuntimeError('Loading error: {}'.format(filename))
                vocab.append( data[cur_pos:space_pos].decode('utf-8').strip() )  # strip для отрезания перевода строки от пред.слова
                cur_pos = space_pos + 1
                vec_end = cur_pos + header[1]*4
                vectors[i, :] = np.frombuffer(data[cur_pos : vec_end], dtype=np.float32)
                cur_pos = vec_end
            return vectors, vocab
            

    def token2ids(self, token_form, oov_info = None):
        
        # простейшее решение для цифровых последовательностей
        if token_form.isnumeric():
            if oov_info:
                oov_info["numeric"] += 1
            lex_idx = self.stems_dict['@num@']
            gra_idx = self.sfx_dict['@num@']
            return [lex_idx, gra_idx]

        tl = len(token_form)

        # попробуем разрезать, как нумероид
        if tl >= 3 and token_form[0] in '0123456789' and token_form[-1] in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдеёжзийклмнопрстуфхцчшщьыъэюя':
            numeroid_head_chars = '0123456789‐‒–—-−˗―─,'
            numeroid_tail_chars = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдеёжзийклмнопрстуфхцчшщьыъэюя-‐'
            for i, c in enumerate([c for c in token_form]):
                if not c in numeroid_head_chars:
                    nhead_idx = i;
                    break;
            for i, c in reversed(list(enumerate([c for c in token_form]))):
                if not c in numeroid_tail_chars:
                    ntail_idx = i;
                    break;
            if nhead_idx-1 == ntail_idx+1:
                token_form = '@num@-' + token_form[nhead_idx:]
                tl = len(token_form)
                if oov_info:
                    oov_info["numeric"] += 1
        
        # ищем в словаре
        if token_form in self.stems_dict and token_form in self.sfx_dict:
            lex_idx = self.stems_dict[token_form]
            gra_idx = self.sfx_dict[token_form]
            return [lex_idx, gra_idx]
        
        # пытаемся найти хотя бы суффикс (сколь возможно длинный, т.е. более информативный)
        if tl > 6:
            MAX_SFX_SIZE = 5
            MIN_STEM_SIZE = 3
            sfx_range = min( tl-MIN_STEM_SIZE, MAX_SFX_SIZE )
            for i in range(-MAX_SFX_SIZE, 0, 1):
                suffix = EmbeddingsReader.OOV_NAME + token_form[i:]
                if suffix in self.sfx_dict:
                    if oov_info:
                        oov_info["oov_sfx"] += 1
                    #stem_idx = self.stems_dict[EmbeddingsReader.OOV_NAME+suffix] if not self.force_oov_with_sfx else EmbeddingsReader.OOV_IDX
                    stem_idx = EmbeddingsReader.OOV_IDX
                    return [stem_idx, self.sfx_dict[suffix]]  # the most long suffix
        
        # не нашли вообще ничего -- полный _OOV_
        if oov_info:
            oov_info["oov"] += 1
        return [EmbeddingsReader.OOV_IDX, EmbeddingsReader.OOV_IDX]





# # код для самодиагностики
# print('Loading embeddings...')
# vm = EmbeddingsReader("vectors-rue.c2v")
# print('  stems emb size = {}'.format(vm.stems_size))
# print('  sfx emb size = {}'.format(vm.sfx_size))
# print('  stems emb count = {}'.format(vm.stems_count))
# print('  sfx emb count = {}'.format(vm.sfx_count))
            
    
# print()        
# for token in ['в', 'ясными', 'гравитационная', '100%-ного', '3d-принтере', '3d-принтер', '1920-1930-', 'jhdfjkahdf', 'оылврзация', ';', '45', '10-летнюю']:
#     print('{} - {}'.format(token, vm.token2ids(token)))
    
# print()
# print('Проверка сходства лексических значений')
# for t1, t2 in [ ['президент', 'премьер-министром'], ['бежал', 'улепетывать'], ['синий', 'из-под'], ['синих', 'лиловый'], 
#                 ['идеально', 'дверь'], ['президентом', 'президентов'], ['суд', 'судно'], ['суда', 'судно'], ['суда', 'суд'],
#                 ['10-летняя', '20-летнее'], ['456', '72'], ['12', 'двенадцать'] ]:
#     idx1, _ = vm.token2ids(t1)
#     idx2, _ = vm.token2ids(t2)
#     vec1 = vm.stems_embs[idx1]
#     vec2 = vm.stems_embs[idx2]
#     cos_sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
#     print('{}, {} -- {}'.format(t1, t2, cos_sim))
    
# print()
# print('Проверка сходства cat-части')
# for t1, t2 in [ ['президент', 'премьер-министром'], ['конь', 'седло'], ['конь', 'аллюр'], ['шина', 'капот'], 
#                 ['идеально', 'дверь'], ['президентом', 'президентов'], ['суд', 'судно'], ['суда', 'судно'], ['суда', 'суд'],
#                 ['10-летняя', '20-летнее'], ['456', '72'], ['12', 'двенадцать'] ]:
#     idx1, _ = vm.token2ids(t1)
#     idx2, _ = vm.token2ids(t2)
#     vec1 = vm.stems_embs[idx1][:60]
#     vec2 = vm.stems_embs[idx2][:60]
#     cos_sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
#     print('{}, {} -- {}'.format(t1, t2, cos_sim))
    
# print()
# print('Проверка сходства assoc-части')
# for t1, t2 in [ ['президент', 'премьер-министром'], ['конь', 'седло'], ['конь', 'аллюр'], ['шина', 'капот'], 
#                 ['идеально', 'дверь'], ['президентом', 'президентов'], ['суд', 'судно'], ['суда', 'судно'], ['суда', 'суд'],
#                 ['10-летняя', '20-летнее'], ['456', '72'], ['12', 'двенадцать'] ]:
#     idx1, _ = vm.token2ids(t1)
#     idx2, _ = vm.token2ids(t2)
#     vec1 = vm.stems_embs[idx1][60:]
#     vec2 = vm.stems_embs[idx2][60:]
#     cos_sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
#     print('{}, {} -- {}'.format(t1, t2, cos_sim))
 
# print()
# print('Проверка сходства грамматических значений')
# for t1, t2 in [ ['синюю', 'железную'], ['в', 'к'], ['в', 'и'], ['президентом', 'президентов'], 
#                 ['идеально', 'дверь'], ['идеально', 'идеальный'], ['бегущий', 'бежать'], ['суда', 'судно'], ['суда', 'суд'],
#                 ['10-летняя', '20-летнее'], ['456', '72'], ['12', 'двенадцать'] ]:
#     _, idx1 = vm.token2ids(t1)
#     _, idx2 = vm.token2ids(t2)
#     vec1 = vm.sfx_embs[idx1]
#     vec2 = vm.sfx_embs[idx2]
#     cos_sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
#     print('{}, {} -- {}'.format(t1, t2, cos_sim))

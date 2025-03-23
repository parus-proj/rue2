
import re


class Tokenizer(object):
    
    def __init__(self, embeddings_reader):
        self.embeddings_reader = embeddings_reader
        
    def process_text(self, text):
        tlist = [] # сами токены
        for i in re.split(r'\s+', text):
            ws = [j for j in re.split(r'([«»‛‚’‘‟„”“ˮʼ‒–—…,.:;"!?()[\]{}⟨⟩<>=≈≠+±/%°§])', i) if j]
            tlist.extend(ws)
        plist = [] # смещения токенов в тексте
        curpos = 0
        for t in tlist:
            curpos = text.find(t, curpos)
            plist.append(curpos)
            curpos += len(t)
        return self.tokens2subtokens([t.lower() for t in tlist]), plist
    
    def tokens2subtokens(self, tokens_list, oov_info = None):
        return [self.token2subtokens(t, oov_info) for t in tokens_list]
    
    def token2subtokens(self, token, oov_info = None):
        return self.embeddings_reader.token2ids(token, oov_info)
    
    
    
# # код для самодиагностики
# from embeddings_reader_2 import EmbeddingsReader
#  
# print('Loading embeddings...')
# vm = EmbeddingsReader("vectors-rue.c2v")
# print('  stems emb size = {}'.format(vm.stems_size))
# print('  sfx emb size = {}'.format(vm.sfx_size))
# print('  stems emb count = {}'.format(vm.stems_count))
# print('  sfx emb count = {}'.format(vm.sfx_count))
#  
#  
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
#  
#  
# s = 'Потом Крапотников повел Славку смотреть на больных.'
# #s = 'Когда в Twitter пришла полиция, И.Маск весело рассмеялся!'
# print(s)
#  
# t = Tokenizer(vm)
# s, _ = t.process_text(s)
# print(s)
#  
# rs = ''
# for w in s:
#     l_idx, s_idx = w
#     j_stem = stem_idx_to_str(l_idx, vm)
#     j_sfx = sfx_idx_to_str(s_idx, vm)
#     rs += '  {}~{}'.format(j_stem, j_sfx)
# print(rs)



"""
Código para criação dos modelos de reranking
"""
import time
from copy import deepcopy
from typing import  List, Dict

from  code.fine_tuning.trata_reader import Reader

# para usar novo modelo, precisa atualizar tabela tratar_reader.prediction_tokens com vtoken_false e vtoken_true


reader = {}
reader_descr = {}
reader_extractive = {}

def adicionar_modelo_reader(parm_nome:str, parm_descr:str, parm_se_extractive:bool):
    """
    Adiciona modelo aos dicionários globais
    """
    # global reader, reader_descr, reader_mono
    nome_caminho_modelo = "models/fine_tuning/" + parm_nome
    reader[parm_nome] = Reader(pretrained_model_name_or_path=nome_caminho_modelo)
    reader_descr[parm_nome] = parm_descr
    reader_extractive[parm_nome] = parm_se_extractive

"""
Modelo abaixo responde fora dos limites
  não adicionar_modelo_reader("deepset/xlm-roberta-base-squad2-distilled",
                        "Multilingual Q&A model (1.03 gb) teached by XLM-RoBERTa-large-squad2. Training data: SQuAD 2.0. Eval data: SQuAD 2.0 dev set (F1: 83.9; EM: 79.8) - German MLQA - German XQuAD",
                        parm_se_extractive=True) #384
"""
adicionar_modelo_reader('pierreguillou/bert-large-cased-squad-v1.1-portuguese',
                        'Modelo Q&A (413 mb) baseado no BERTimbau Large com refinamento no squad_v1_pt. (F1:84.4; EM=78.4)',
                        parm_se_extractive=True) # 512

"""
# outros candidatos

adicionar_modelo_reader('pierreguillou/bert-base-cased-squad-v1.1-portuguese',
                        'Modelo Q&A (1.24 gb) baseado no BERTimbau Base com refinamento no squad_v1_pt. (F1:82.5; EM=70.4)',
                        parm_se_extractive=True)


adicionar_modelo_reader("deepset/xlm-roberta-base-squad2",
                        "Multilingual Q&A model (1.03 gb) over XLM-RoBERTa base. Training data: SQuAD 2.0. Eval data: SQuAD 2.0 dev set - German MLQA - German XQuAD",
                        parm_se_extractive=True)

# Palak/xlm-roberta-large_squad eval-f1 92
"""

lista_modelos_reader = list(reader.keys())
regex_lista_modelo_reader_valida_json  = "" # pylint: disable=invalid-name # é uma variável
for sigla_modelo in lista_modelos_reader:
    print(f'modelo reader inicializados: {sigla_modelo} Hora: {time.strftime("[%Y-%b-%d %H:%M:%S]")}')
    regex_lista_modelo_reader_valida_json += sigla_modelo + "|"

regex_lista_modelo_reader_valida_json = regex_lista_modelo_reader_valida_json[:-1]

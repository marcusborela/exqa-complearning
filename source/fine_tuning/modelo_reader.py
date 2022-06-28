"""
Código para criação dos modelos de reranking
"""
import time
from copy import deepcopy
from typing import  List, Dict

from  source.fine_tuning.trata_reader import Reader

# para usar novo modelo, precisa atualizar tabela tratar_reader.prediction_tokens com vtoken_false e vtoken_true


reader = {}
reader_descr = {}

def adicionar_modelo_reader(parm_nome:str, parm_descr:str):
    """
    Adiciona modelo aos dicionários globais
    """
    # global reader, reader_descr, reader_mono
    nome_caminho_modelo = "models/fine_tuning/" + parm_nome
    reader[parm_nome] = Reader(pretrained_model_name_or_path=nome_caminho_modelo)
    reader_descr[parm_nome] = parm_descr

"""
Modelo abaixo responde fora dos limites
  não adicionar_modelo_reader("deepset/xlm-roberta-base-squad2-distilled",
                        "Multilingual Q&A model (1.03 gb) teached by XLM-RoBERTa-large-squad2. Training data: SQuAD 2.0. Eval data: SQuAD 2.0 dev set (F1: 83.9; EM: 79.8) - German MLQA - German XQuAD",
                        parm_se_extractive=True) #384
"""
adicionar_modelo_reader('pierreguillou/bert-large-cased-squad-v1.1-portuguese',
                        'Modelo Q&A (1.24 gb) baseado no BERTimbau Base com refinamento no squad_v1_pt. (F1:84.43; EM=72.68)',
                        )

adicionar_modelo_reader('distilbert-base-cased-distilled-squad',
                        "Model (261 mb) is a fine-tune checkpoint of DistilBERT-base-cased, fine-tuned using (a second step of) knowledge distillation on SQuAD v1.1. This model reaches a F1 score of 87.1 on the dev set (for comparison, BERT bert-base-cased version reaches a F1 score of 88.7).",
                        )


"""
# outros candidatos

adicionar_modelo_reader('pierreguillou/bert-base-cased-squad-v1.1-portuguese',
                        'Modelo Q&A (413 mb) baseado no BERTimbau Base com refinamento no squad_v1_pt. (F1:82.5; EM=70.4)',
                        )


adicionar_modelo_reader("deepset/xlm-roberta-base-squad2",
                        "Multilingual Q&A model (1.03 gb) over XLM-RoBERTa base. Training data: SQuAD 2.0. Eval data: SQuAD 2.0 dev set - German MLQA - German XQuAD",
                        )

# Palak/xlm-roberta-large_squad eval-f1 92
"""
sigla_reader_en = 'distilbert-base-cased-distilled-squad'
sigla_reader_pt = 'pierreguillou/bert-large-cased-squad-v1.1-portuguese'

reader_en = reader[sigla_reader_en]
reader_pt = reader[sigla_reader_pt]


"""
Boas referências: https://github.com/nlp-with-transformers/notebooks
"""

import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'./.'))
from typing import Dict

from source.data_related import squad_related
from source.calculation import squad_evaluate_v1_1
from source.calculation.transfer_learning.trata_reader import Reader
from source.calculation import util_modelo



def evaluate_transfer_method(parm_language:str, parm_dict_config_model:Dict, parm_interval_print:int=1000):

    assert parm_language in ('en','pt'), f"parm_language must be in ('en','pt'). Found: {parm_language}"
    # tentativa de deixar os resultados repetíveis
    util_modelo.inicializa_seed(123) # para tentar repetir cálculo

    if parm_language == 'en':
        squad_dataset = squad_related.load_squad_dataset_1_1(parm_language='en')
        name_model = 'distilbert-base-cased-distilled-squad'
        model = Reader(parm_name_model=name_model, parm_dict_config=parm_dict_config_model)
    else:
        squad_dataset = squad_related.load_squad_dataset_1_1(parm_language='pt')
        name_model = 'pierreguillou/bert-large-cased-squad-v1.1-portuguese'
        model = Reader(parm_name_model=name_model, parm_dict_config=parm_dict_config_model)


    resultado = squad_evaluate_v1_1.evaluate_learning_method_nested(parm_dataset=squad_dataset,
                    parm_reader = model,
                    parm_dict_config_model=dict_config_model,
                    parm_dict_config_eval=dict_config_eval,
                    parm_interval_print= parm_interval_print  )

    """
        resultado = squad_evaluate_v1_1.evaluate_learning_method_dataset(parm_dataset=squad_dataset,
                    parm_reader = model,
                    parm_dict_config_model=dict_config_model,
                    parm_dict_config_eval=dict_config_eval,
                    parm_interval_print= parm_interval_print  )
    """




dict_config_model = {"num_doc_stride":128,\
               "num_top_k":3, \
               "num_batch_size":128, \
               "num_max_answer_length":40, \
               "if_handle_impossible_answer":False, \
               "num_factor_multiply_top_k":10}

# dict_config_eval = {}
dict_config_eval = {"num_question_max": 2}

evaluate_transfer_method('pt', dict_config_model)
#evaluate_transfer_method('en', dict_config_model)



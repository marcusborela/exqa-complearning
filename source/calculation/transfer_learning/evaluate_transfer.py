import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'./.'))

from source.data_related import squad_related
from source.calculation.transfer_learning.squad_evaluate_v1_1 import evaluate_transfer

squad_dataset_en = squad_related.carregar_squad_1_1(parm_language='en')
squad_dataset_pt = squad_related.carregar_squad_1_1(parm_language='pt')

dict_config_model = {"num_doc_stride":128,\
               "num_top_k":3, \
               "num_max_answer_length":30, \
               "if_handle_impossible_answer":False, \
               "num_factor_multiply_top_k":3}

dict_config_eval = {"num_question_max":30}

resultado_en = evaluate_transfer(parm_dataset=squad_dataset_en,
                parm_dict_config_model=dict_config_model,
                parm_dict_config_eval=dict_config_eval,
                parm_interval_print= 15  )

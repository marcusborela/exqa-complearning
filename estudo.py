print('oi')
import copy
import time
import os, sys
print(sys.path)
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'.\.'))
# from source.data_related import squad_related, rastro_evaluation_qa
# from source.calculation.transfer_learning.squad_evaluate_v1_1 import evaluate_transfer

from datasets import load_dataset
from source.calculation.transfer_learning.squad_evaluate_v1_1 import evaluate_transfer
from source.data_related import squad_related

squad_dataset_en =squad_related.load_squad_dataset_1_1('en')

dict_config_model = {"num_doc_stride":128,\
               "num_top_k":3, \
               "num_max_answer_length":30, \
               "if_handle_impossible_answer":False, \
               "num_factor_multiply_top_k":10}

dict_config_eval = {"num_question_max":30}

resultado = evaluate_transfer(parm_dataset=squad_dataset_en,
                parm_dict_config_model=dict_config_model,
                parm_dict_config_eval=dict_config_eval,
                parm_if_record=False,
                parm_interval_print= 1000)




exit()

squad_dataset_en = squad_related.load_squad_dataset_1_1(parm_language='en')
squad_dataset_pt = squad_related.load_squad_dataset_1_1(parm_language='pt')

dict_config_model = {"num_doc_stride":40,\
               "num_top_k":3, \
               "num_max_answer_length":50, \
               "if_handle_impossible_answer":False, \
               "num_factor_multiply_top_k":3}

dict_config_eval = {"num_question_max":20}

resultado_en = evaluate_transfer(parm_dataset=squad_dataset_en,
                parm_dict_config_model=dict_config_model,
                parm_dict_config_eval=dict_config_eval,
                parm_interval_print= 15  )


"""

from source.data_related import rastro_evaluation_qa


rastro_eval_qa = rastro_evaluation_qa.RastroEvaluationQa()
rastro_eval_qa.imprime()
rastro_eval_qa.save()

"""
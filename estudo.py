print('oi')
import copy
import time
import os, sys
print(sys.path)
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'.\.'))
from source.data_related import squad_related

squad_dataset = squad_related.load_squad_dataset_1_1(parm_language='en')



"""



from source.data_related import rastro_evaluation_qa

GPUs = GPUtil.getGPUs()
gpu = GPUs[0]
print(f"GPU Used: {gpu.memoryUsed :.0f}MB | Util {gpu.memoryUtil*100:3.0f}% | RAM Free: {gpu.memoryFree :.0f}MB | Total {gpu.memoryTotal:.0f}MB")

exit()
rastro_eval_qa = rastro_evaluation_qa.RastroEvaluationQa()
rastro_eval_qa.imprime()

rastro_eval_qa = rastro_evaluation_qa.RastroEvaluationQa()
# rastro_eval_qa.imprime()
print(rastro_eval_qa.df_evaluation.iloc[1])
print(rastro_eval_qa.df_evaluation['name_model'].iloc[1])

# rastro_eval_qa.save()


prompt_format = '''Instrução: Usando palavras do texto abaixo, responda à pergunta:

Texto: {context}

Pergunta: {question}
Resposta:'''

prompt_format = repr('''Instrução: Usando palavras do texto abaixo, responda à pergunta:

Texto: {context}

Pergunta: {question}
Resposta:''')



print(str(prompt_format))
print('-'*20)
print(repr(prompt_format))
print('-'*20)


answer_stop -> list_stop_words
generator_sample -> if_do_sample
val_length_penalty
generator_temperature -> num_temperature
prompt_format > descr_prompt_format

from datasets import load_dataset
from source.calculation.transfer_learning import squad_evaluate_v1_1
from source.data_related import squad_related
from source.calculation.transfer_learning.trata_reader import Reader

squad_dataset_en =squad_related.load_squad_dataset_1_1('en')

dict_config_model = {"num_doc_stride":128,\
               "num_top_k":3, \
               "num_batch_size":128, \
               "num_max_answer_length":80, \
               "if_handle_impossible_answer":False, \
               "num_factor_multiply_top_k":10}

dict_config_eval =  {"num_question_max":5} # {}


if squad_dataset_en.language == 'en':
    name_model = 'distilbert-base-cased-distilled-squad'
    path_model = "models/transfer_learning/"+name_model
    model = Reader(pretrained_model_name_or_path=path_model, parm_dict_config=dict_config_model)
else:
    name_model = 'pierreguillou/bert-large-cased-squad-v1.1-portuguese'
    path_model = "models/transfer_learning/"+name_model
    model = Reader(pretrained_model_name_or_path=path_model, parm_dict_config=dict_config_model)

resultado = squad_evaluate_v1_1.evaluate_transfer_dataset(parm_dataset=squad_dataset_en,
                parm_dict_config_model=dict_config_model,
                parm_dict_config_eval=dict_config_eval,
                parm_if_record=True,
                parm_interval_print= 1000)

# resposta = model.answer(dts)

print(resultado)
exit()


resposta = model.answer(texto_contexto=[squad_dataset_en.dataset[0]['context'], squad_dataset_en.dataset[2]['context']],\
                        texto_pergunta=[squad_dataset_en.dataset[0]['question'], squad_dataset_en.dataset[2]['question']])

resultado = squad_evaluate_v1_1.evaluate_transfer_dataset(parm_dataset=squad_dataset_en,
                parm_dict_config_model=dict_config_model,
                parm_dict_config_eval=dict_config_eval,
                parm_if_record=False,
                parm_interval_print= 1000)


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




from source.data_related import rastro_evaluation_qa


rastro_eval_qa = rastro_evaluation_qa.RastroEvaluationQa()
rastro_eval_qa.imprime()
rastro_eval_qa.save()

"""
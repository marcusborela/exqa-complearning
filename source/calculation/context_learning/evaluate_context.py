"""
Boas referências: https://github.com/nlp-with-transformers/notebooks
"""

import os
import sys
from time import pthread_getcpuclockid
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'./.'))
from typing import Dict

from source.data_related import squad_related
from source.calculation import squad_evaluate_v1_1
from source.calculation.context_learning import trata_reader_pipe
from source.calculation import util_modelo



def evaluate_context_method(parm_language:str,
                 parm_name_model:str,
                 parm_dict_config_model:Dict,
                 parm_dict_config_eval:Dict,
                 parm_if_trace:bool=False,
                 parm_if_record:bool=True,
                 parm_interval_print:int=1000):

    assert parm_language in ('en','pt'), f"parm_language must be in ('en','pt'). Found: {parm_language}"
    # tentativa de deixar os resultados repetíveis
    util_modelo.inicializa_seed(123) # para tentar repetir cálculo

    squad_dataset = squad_related.load_squad_dataset_1_1(parm_language=parm_language)

    model = trata_reader_pipe.Reader(parm_name_model, parm_dict_config_model)
    # print(f"model.info()\n {model.info}")


    resultado = squad_evaluate_v1_1.evaluate_learning_method_one_by_one_dataset(parm_dataset=squad_dataset,
                    parm_reader = model,
                    parm_dict_config_model=parm_dict_config_model,
                    parm_dict_config_eval=parm_dict_config_eval,
                    parm_if_trace=parm_if_trace,
                    parm_interval_print= parm_interval_print,
                    parm_if_record=parm_if_record)

    return resultado

dict_config_model = {
                # parâmetros complementares
                "learning_method":'context',
                "num_top_k": 3,
                "num_max_answer_length":80,
                # parâmetros transfer # deixar branco
                "num_doc_stride": '',
                "num_factor_multiply_top_k": '',
                "if_handle_impossible_answer": '',
                # parâmetros context
                'list_stop_words': ['.', '\n','!'],
                'cod_prompt_format': 102.,
                'if_do_sample': False,
                "val_length_penalty":0.,
                'val_temperature': 0.1,
}


results = []
# dict_config_eval = {}
dict_config_eval = {"num_question_max": 8}
name_model = "EleutherAI/gpt-j-6B"
sigla_linguagem = 'pt'

resultado = evaluate_context_method(sigla_linguagem,
            name_model,
            dict_config_model,
            dict_config_eval,
            parm_if_trace=False,
            parm_if_record=False)
results.append([name_model, sigla_linguagem, resultado])

sigla_linguagem = 'en'
resultado = evaluate_context_method(sigla_linguagem,
            name_model,
            dict_config_model,
            dict_config_eval,
            parm_if_trace=False,
            parm_if_record=False)
results.append([name_model, sigla_linguagem, resultado])

print(results)

exit()

results.append([name_model, sigla_linguagem, resultado])
name_model = "EleutherAI/gpt-neo-2.7B"
resultado = evaluate_context_method(sigla_linguagem,
            name_model,
            dict_config_model,
            dict_config_eval,
            parm_if_trace=False,
            parm_if_record=False)
results.append([name_model, sigla_linguagem, resultado])

name_model = "EleutherAI/gpt-neo-1.3B"
resultado = evaluate_context_method(sigla_linguagem,
            name_model,
            dict_config_model,
            dict_config_eval,
            parm_if_trace=False,
            parm_if_record=False)
results.append([name_model, sigla_linguagem, resultado])


print(results)


exit()
resultado = evaluate_context_method(sigla_linguagem, name_model, dict_config_model, dict_config_eval,parm_if_record=True)
print(resultado)

dict_config_model["num_max_answer_length"] = 160
resultado = evaluate_context_method('pt', name_model, dict_config_model, dict_config_eval,parm_if_record=True)
print(resultado)
resultado = evaluate_context_method(sigla_linguagem, name_model, dict_config_model, dict_config_eval,parm_if_record=True)
print(resultado)


resultado = evaluate_context_method(sigla_linguagem,
            name_model,
            dict_config_model,
            dict_config_eval,
            parm_if_trace=False,
            parm_if_record=False)

print(resultado)


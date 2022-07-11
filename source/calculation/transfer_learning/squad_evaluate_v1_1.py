import time
from typing import Dict

from source.calculation.metric.qa_metric import calculate_metrics
from source.calculation.transfer_learning.trata_reader import Reader
from source.data_related.squad_related import SquadDataset
from source.data_related import rastro_evaluation_qa
from source.calculation import util_modelo

example_dict_config_model = {"num_doc_stride":128,\
               "num_top_k":3, \
               "num_max_answer_length":30, \
               "if_handle_impossible_answer":False, \
               "num_factor_multiply_top_k":3}

def evaluate_transfer_nested(parm_dataset:SquadDataset, \
                      parm_dict_config_model:Dict, \
                      parm_dict_config_eval:Dict, \
                      parm_if_record:bool=True, \
                      parm_interval_print:int=100):
    """
    Evaluates specific models in transfer_learning using configuration in parm_config
    The models are:
        Portuguese
            pierreguillou/bert-large-cased-squad-v1.1-portuguese
            Modelo Q&A (1.24 gb) baseado no BERTimbau Base com refinamento no squad_v1_pt. (F1:84.43; EM=72.68)
        English
            distilbert-base-cased-distilled-squad
            Model (261 mb) is a fine-tune checkpoint of DistilBERT-base-cased, fine-tuned using (a second step of) knowledge distillation on SQuAD v1.1. This model reaches a F1 score of 87.1 on the dev set (for comparison, BERT bert-base-cased version reaches a F1 score of 88.7).

    Expected config parameters in parm_dict_config_model as defined in Reader()
    """
    msg_dif = util_modelo.compare_dicionarios_chaves(example_dict_config_model,
                parm_dict_config_model,'Esperado', 'Encontrado')

    assert msg_dif == "", f"Estrutura esperada de parm_dict_config_model não corresponde ao esperado {msg_dif}"

    assert parm_dict_config_model['num_top_k']>=3, f"To get EM@3 and F1@3 it must ask for at least 3 answers. It has {parm_dict_config_model['num_top_k']} answers asked."
    assert isinstance(parm_dict_config_eval,dict), f"parm_dict_config_eval must be a dict"

    if parm_dataset.language == 'en':
        name_model = 'distilbert-base-cased-distilled-squad'
        path_model = "models/transfer_learning/"+name_model
        model = Reader(pretrained_model_name_or_path=path_model, parm_dict_config=parm_dict_config_model)
    else:
        name_model = 'pierreguillou/bert-large-cased-squad-v1.1-portuguese'
        path_model = "models/transfer_learning/"+name_model
        model = Reader(pretrained_model_name_or_path=path_model, parm_dict_config=parm_dict_config_model)

    dict_evaluation = {
        # context
        'name_learning_method':'transfer',
        'ind_language':parm_dataset.language,
        'name_model':name_model,
        'name_device':model.name_device,
        'descr_filter':str(parm_dict_config_eval),
        # números gerais
        'datetime_execution': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_top_k':parm_dict_config_model['num_top_k'],
        "num_factor_multiply_top_k":parm_dict_config_model['num_factor_multiply_top_k'],
        # parâmetros (pode crescer)
        # Transfer Learning
        'num_doc_stride':parm_dict_config_model['num_doc_stride'],
        'num_max_answer_length':parm_dict_config_model['num_max_answer_length'],
        'if_handle_impossible_answer':parm_dict_config_model['if_handle_impossible_answer'],
        # Context learning
        'ind_format_prompt':'',
        'descr_task_prompt':'',
        'num_shot':99999  # usado valor numérico para não dar erro na carga depois
    }

    if parm_dict_config_eval and isinstance(parm_dict_config_eval, dict) and 'num_question_max' in parm_dict_config_eval:
        num_question_max = parm_dict_config_eval['num_question_max']
    else:
        num_question_max = 999999999

    num_question = 0
    f1_at_3 = f1 = exact_match = exact_match_at_3 =  0.
    print(f"Evalating in dataset {parm_dataset.name} model \n{model.info} ")
    tstart = time.time()
    for article in parm_dataset.nested_json:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                num_question += 1
                # calcular resposta
                resposta = model.answer(texto_contexto=paragraph['context'],\
                                        texto_pergunta=qa['question'])
                # print(f"resposta {resposta}")
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                metric_calculated = calculate_metrics(resposta, ground_truths)
                # print(f"metric_calculated {metric_calculated}")
                exact_match += metric_calculated['EM']
                f1 += metric_calculated['F1']
                exact_match_at_3 += metric_calculated['EM@3']
                f1_at_3 += metric_calculated['F1@3']

                if num_question % parm_interval_print == 0:
                    print(f"#{num_question} \
                    EM:{round(100.0 * exact_match / num_question,2)} \
                    F1:{round(100.0 * f1 / num_question,2)} \
                    EM@3:{round(100.0 * exact_match_at_3 / num_question,2)} \
                    F1@3:{round(100.0 * f1_at_3 / num_question,2)}")
                if num_question >= num_question_max:
                    break
            if num_question >= num_question_max:
                break
        if num_question >= num_question_max:
            break


    tend = time.time()
    exact_match = round(100.0 * exact_match / num_question,2)
    f1 = round(100.0 * f1 / num_question,2)
    exact_match_at_3 = round(100.0 * exact_match_at_3 / num_question,2)
    f1_at_3 = round(100.0 * f1_at_3 / num_question,2)


    dict_evaluation['num_question'] = num_question
    dict_evaluation['time_execution_total'] = int(round(tend - tstart, 0))
    dict_evaluation['time_execution_per_question'] = int(round(1000*(tend - tstart)/num_question, 0))
    evaluation = rastro_evaluation_qa.EvaluationQa(dict_evaluation)

    calculate_metric = rastro_evaluation_qa.CalculatedMetric({'cod_metric':'EM', 'value':round(exact_match,2)})
    evaluation.add_metric(calculate_metric)

    calculate_metric = rastro_evaluation_qa.CalculatedMetric({'cod_metric':'F1', 'value':round(f1,2)})
    evaluation.add_metric(calculate_metric)

    calculate_metric = rastro_evaluation_qa.CalculatedMetric({'cod_metric':'EM@3', 'value':round(exact_match_at_3,2)})
    evaluation.add_metric(calculate_metric)

    calculate_metric = rastro_evaluation_qa.CalculatedMetric({'cod_metric':'F1@3', 'value':round(f1_at_3,2)})
    evaluation.add_metric(calculate_metric)

    print(f"Evaluation result: {evaluation.info}")

    if parm_if_record:
        # atualizar dataframes
        rastro_evaluation_qa.persist_evaluation([evaluation])


    return evaluation.info


def evaluate_transfer_dataset(parm_dataset:SquadDataset, \
                      parm_dict_config_model:Dict, \
                      parm_dict_config_eval:Dict, \
                      parm_if_record:bool=True, \
                      parm_interval_print:int=100):
    """
    Evaluates specific models in transfer_learning using configuration in parm_config
    The models are:
        Portuguese
            pierreguillou/bert-large-cased-squad-v1.1-portuguese
            Modelo Q&A (1.24 gb) baseado no BERTimbau Base com refinamento no squad_v1_pt. (F1:84.43; EM=72.68)
        English
            distilbert-base-cased-distilled-squad
            Model (261 mb) is a fine-tune checkpoint of DistilBERT-base-cased, fine-tuned using (a second step of) knowledge distillation on SQuAD v1.1. This model reaches a F1 score of 87.1 on the dev set (for comparison, BERT bert-base-cased version reaches a F1 score of 88.7).

    Expected config parameters in parm_dict_config_model as defined in Reader()
    """
    msg_dif = util_modelo.compare_dicionarios_chaves(example_dict_config_model,
                parm_dict_config_model,'Esperado', 'Encontrado')

    assert msg_dif == "", f"Estrutura esperada de parm_dict_config_model não corresponde ao esperado {msg_dif}"

    assert parm_dict_config_model['num_top_k']>=3, f"To get EM@3 and F1@3 it must ask for at least 3 answers. It has {parm_dict_config_model['num_top_k']} answers asked."
    assert isinstance(parm_dict_config_eval,dict), f"parm_dict_config_eval must be a dict"

    if parm_dataset.language == 'en':
        name_model = 'distilbert-base-cased-distilled-squad'
        path_model = "models/transfer_learning/"+name_model
        model = Reader(pretrained_model_name_or_path=path_model, parm_dict_config=parm_dict_config_model)
    else:
        name_model = 'pierreguillou/bert-large-cased-squad-v1.1-portuguese'
        path_model = "models/transfer_learning/"+name_model
        model = Reader(pretrained_model_name_or_path=path_model, parm_dict_config=parm_dict_config_model)

    dict_evaluation = {
        # context
        'name_learning_method':'transfer',
        'ind_language':parm_dataset.language,
        'name_model':name_model,
        'name_device':model.name_device,
        'descr_filter':str(parm_dict_config_eval),
        # números gerais
        'datetime_execution': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_top_k':parm_dict_config_model['num_top_k'],
        "num_factor_multiply_top_k":parm_dict_config_model['num_factor_multiply_top_k'],
        # parâmetros (pode crescer)
        # Transfer Learning
        'num_doc_stride':parm_dict_config_model['num_doc_stride'],
        'num_max_answer_length':parm_dict_config_model['num_max_answer_length'],
        'if_handle_impossible_answer':parm_dict_config_model['if_handle_impossible_answer'],
        # Context learning
        'ind_format_prompt':'',
        'descr_task_prompt':'',
        'num_shot':99999  # usado valor numérico para não dar erro na carga depois
    }

    if parm_dict_config_eval and isinstance(parm_dict_config_eval, dict) and 'num_question_max' in parm_dict_config_eval:
        parm_dataset.dataset.select(range(parm_dict_config_eval['num_question_max']))

    num_question = 0
    f1_at_3 = f1 = exact_match = exact_match_at_3 =  0.
    print(f"Evalating in dataset {parm_dataset.name} model \n{model.info} ")
    tstart = time.time()
    for article in parm_dataset.nested_json:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                num_question += 1
                # calcular resposta
                resposta = model.answer(texto_contexto=paragraph['context'],\
                                        texto_pergunta=qa['question'])
                # print(f"resposta {resposta}")
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                metric_calculated = calculate_metrics(resposta, ground_truths)
                # print(f"metric_calculated {metric_calculated}")
                exact_match += metric_calculated['EM']
                f1 += metric_calculated['F1']
                exact_match_at_3 += metric_calculated['EM@3']
                f1_at_3 += metric_calculated['F1@3']

                if num_question % parm_interval_print == 0:
                    print(f"#{num_question} \
                    EM:{round(100.0 * exact_match / num_question,2)} \
                    F1:{round(100.0 * f1 / num_question,2)} \
                    EM@3:{round(100.0 * exact_match_at_3 / num_question,2)} \
                    F1@3:{round(100.0 * f1_at_3 / num_question,2)}")


    tend = time.time()
    exact_match = round(100.0 * exact_match / num_question,2)
    f1 = round(100.0 * f1 / num_question,2)
    exact_match_at_3 = round(100.0 * exact_match_at_3 / num_question,2)
    f1_at_3 = round(100.0 * f1_at_3 / num_question,2)


    dict_evaluation['num_question'] = num_question
    dict_evaluation['time_execution_total'] = int(round(tend - tstart, 0))
    dict_evaluation['time_execution_per_question'] = int(round(1000*(tend - tstart)/num_question, 0))
    evaluation = rastro_evaluation_qa.EvaluationQa(dict_evaluation)

    calculate_metric = rastro_evaluation_qa.CalculatedMetric({'cod_metric':'EM', 'value':round(exact_match,2)})
    evaluation.add_metric(calculate_metric)

    calculate_metric = rastro_evaluation_qa.CalculatedMetric({'cod_metric':'F1', 'value':round(f1,2)})
    evaluation.add_metric(calculate_metric)

    calculate_metric = rastro_evaluation_qa.CalculatedMetric({'cod_metric':'EM@3', 'value':round(exact_match_at_3,2)})
    evaluation.add_metric(calculate_metric)

    calculate_metric = rastro_evaluation_qa.CalculatedMetric({'cod_metric':'F1@3', 'value':round(f1_at_3,2)})
    evaluation.add_metric(calculate_metric)

    print(f"Evaluation result: {evaluation.info}")

    if parm_if_record:
        # atualizar dataframes
        rastro_evaluation_qa.persist_evaluation([evaluation])



    return evaluation.info



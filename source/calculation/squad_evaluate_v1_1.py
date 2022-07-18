import time
from typing import Dict

from source.calculation.metric.qa_metric import calculate_metrics, calculate_metrics_grouped
from source.data_related.squad_related import SquadDataset
from source.data_related import rastro_evaluation_qa
from source.calculation import util_modelo

example_dict_config_model = {
                # parâmetros transfer
               'learning_method':'transfer',
               "num_doc_stride":128,\
               "num_top_k":3, \
               "num_max_answer_length":30, \
               "if_handle_impossible_answer":False, \
               "num_factor_multiply_top_k":3,
                # parâmetros context
                'list_stop_words': ['.', '\n', '\n','!'],
                'if_do_sample': True,
                "val_length_penalty":0,
                'val_temperature': 0.1,
                'cod_prompt_format': 0}

def validate_config_eval(parm_dataset:SquadDataset,
                         parm_reader,
                         parm_dict_config_model:Dict,
                         parm_dict_config_eval:Dict):
    msg_dif = util_modelo.compare_dicionarios_chaves(example_dict_config_model,
                parm_dict_config_model,'Esperado', 'Encontrado')

    assert msg_dif == "", f"Estrutura esperada de parm_dict_config_model não corresponde ao esperado {msg_dif}"

    assert parm_dict_config_model['num_top_k']>=3, f"To get EM@3 and F1@3 it must ask for at least 3 answers. It has {parm_dict_config_model['num_top_k']} answers asked."
    assert isinstance(parm_dict_config_eval,dict), f"parm_dict_config_eval must be a dict"

    dict_evaluation = {
        # context
        'name_learning_method':parm_dict_config_model['learning_method'],
        'ind_language':parm_dataset.language,
        'name_model':parm_reader.name_model,
        'name_device':parm_reader.name_device,
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
        'list_stop_words':str(parm_dict_config_model['list_stop_words']),
        'if_do_sample':parm_dict_config_model['if_do_sample'],
        "val_length_penalty":parm_dict_config_model['val_length_penalty'],
        'val_temperature':parm_dict_config_model['val_temperature'],
        'cod_prompt_format':parm_dict_config_model['cod_prompt_format']
    }
    return dict_evaluation

def evaluate_learning_method_nested(parm_dataset:SquadDataset, \
                      parm_reader, \
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
    dict_evaluation = validate_config_eval(parm_dataset, parm_reader, parm_dict_config_model,parm_dict_config_eval)

    if parm_dict_config_eval and isinstance(parm_dict_config_eval, dict) and 'num_question_max' in parm_dict_config_eval:
        num_question_max = parm_dict_config_eval['num_question_max']
    else:
        num_question_max = 999999999

    metric_per_question = {}
    num_question = 0
    f1_at_3 = f1 = exact_match = exact_match_at_3 =  0.
    print(f"Evalating in dataset {parm_dataset.name} model \n{parm_reader.info} ")
    tstart = time.time()
    for article in parm_dataset.nested_json:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                num_question += 1
                # calcular resposta
                resposta = parm_reader.answer_one_question(texto_contexto=paragraph['context'],\
                                        texto_pergunta=qa['question'])
                # print(f"resposta {resposta}")
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                metric_calculated = calculate_metrics(resposta, ground_truths)
                # print(f"metric_calculated {metric_calculated}")
                exact_match += metric_calculated['EM']
                f1 += metric_calculated['F1']
                exact_match_at_3 += metric_calculated['EM@3']
                f1_at_3 += metric_calculated['F1@3']
                metric_per_question[qa['id']] = metric_calculated

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
        cod_evaluation = rastro_evaluation_qa.persist_evaluation([evaluation])
        rastro_evaluation_qa.persist_metric_per_question(parm_cod_evaluation=cod_evaluation,\
                                            parm_dict_metric_per_question=metric_per_question)


    return evaluation.info

def evaluate_learning_method_one_by_one_dataset(parm_dataset:SquadDataset, \
                      parm_reader, \
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
    dict_evaluation = validate_config_eval(parm_dataset, parm_reader, parm_dict_config_model,parm_dict_config_eval)

    target_dataset = parm_dataset.dataset
    if parm_dict_config_eval is not None and isinstance(parm_dict_config_eval, dict) and 'num_question_max' in parm_dict_config_eval:
        target_dataset = target_dataset.select(range(parm_dict_config_eval['num_question_max']))

    num_question = 0
    print(f"Evalating in dataset {parm_dataset.name} model \n{parm_reader.info} ")



    if parm_dict_config_eval and isinstance(parm_dict_config_eval, dict) and 'num_question_max' in parm_dict_config_eval:
        num_question_max = parm_dict_config_eval['num_question_max']
    else:
        num_question_max = 999999999

    metric_per_question = {}

    num_question = 0
    f1_at_3 = f1 = exact_match = exact_match_at_3 =  0.
    print(f"Evalating in dataset {parm_dataset.name} model \n{parm_reader.info} ")
    tstart = time.time()
    for qa in target_dataset:
        num_question += 1
        # calcular resposta
        resposta = parm_reader.answer_one_question(texto_contexto=qa['context'],\
                                texto_pergunta=qa['question'])
        # print(f"resposta {resposta}")
        list_ground_truth = qa['answer_text']
        metric_calculated = calculate_metrics(resposta, list_ground_truth)
        # print(f"metric_calculated {metric_calculated}")
        exact_match += metric_calculated['EM']
        f1 += metric_calculated['F1']
        exact_match_at_3 += metric_calculated['EM@3']
        f1_at_3 += metric_calculated['F1@3']
        metric_per_question[qa['id']] = metric_calculated


        if num_question % parm_interval_print == 0:
            print(f"#{num_question} \
            EM:{round(100.0 * exact_match / num_question,2)} \
            F1:{round(100.0 * f1 / num_question,2)} \
            EM@3:{round(100.0 * exact_match_at_3 / num_question,2)} \
            F1@3:{round(100.0 * f1_at_3 / num_question,2)}")
        if num_question >= num_question_max:
            break

    tend = time.time()
    exact_match = round(100.0 * exact_match / num_question,2)
    f1 = round(100.0 * f1 / num_question,2)
    exact_match_at_3 = round(100.0 * exact_match_at_3 / num_question,2)
    f1_at_3 = round(100.0 * f1_at_3 / num_question,2)

    tend = time.time()


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
        cod_evaluation = rastro_evaluation_qa.persist_evaluation([evaluation])
        rastro_evaluation_qa.persist_metric_per_question(parm_cod_evaluation=cod_evaluation,\
                                            parm_dict_metric_per_question=metric_per_question)

    return evaluation.info



def evaluate_learning_method_dataset(parm_dataset:SquadDataset, \
                      parm_reader, \
                      parm_dict_config_model:Dict, \
                      parm_dict_config_eval:Dict, \
                      parm_if_record:bool=True):
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
    dict_evaluation = validate_config_eval(parm_dataset, parm_reader, parm_dict_config_model,parm_dict_config_eval)

    target_dataset = parm_dataset.dataset
    if parm_dict_config_eval is not None and isinstance(parm_dict_config_eval, dict) and 'num_question_max' in parm_dict_config_eval:
        target_dataset = target_dataset.select(range(parm_dict_config_eval['num_question_max']))

    num_question = 0
    print(f"Evalating in dataset {parm_dataset.name} model \n{parm_reader.info} ")

    num_question = len(target_dataset)
    tstart = time.time()

    # calcular resposta
    lista_resposta = parm_reader.answer(parm_dataset=target_dataset)
    # print(f"resposta {resposta}")


    metric_calculated, metric_calculated_per_question = calculate_metrics_grouped(lista_resposta, target_dataset, num_question)

    tend = time.time()


    dict_evaluation['num_question'] = num_question
    dict_evaluation['time_execution_total'] = int(round(tend - tstart, 0))
    dict_evaluation['time_execution_per_question'] = int(round(1000*(tend - tstart)/num_question, 0))
    evaluation = rastro_evaluation_qa.EvaluationQa(dict_evaluation)

    for metric, metric_value in metric_calculated.items():
        calculate_metric = rastro_evaluation_qa.CalculatedMetric({'cod_metric':metric, 'value':metric_value})
        evaluation.add_metric(calculate_metric)

    print(f"Evaluation result: {evaluation.info}")

    if parm_if_record:
        # atualizar dataframes
        cod_evaluation = rastro_evaluation_qa.persist_evaluation([evaluation])
        rastro_evaluation_qa.persist_metric_per_question(parm_cod_evaluation=cod_evaluation,\
                                            parm_dict_metric_per_question=metric_calculated_per_question)




    return evaluation.info



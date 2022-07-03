import time
from typing import Dict

from source.calculation.metric.qa_metric import calculate_metrics
from source.calculation.transfer_learning.trata_reader import Reader
from source.data_related.squad_related import SquadDataset

def evaluate_transfer(parm_dataset:SquadDataset, \
                      parm_dict_config_model:Dict, \
                      parm_dict_config_eval:Dict, \
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
    assert parm_dict_config_model['top_k']>=3, f"To get EM@3 and F1@3 it must ask for at least 3 answers. It has {parm_dict_config_model['top_k']} answers asked."

    if parm_dataset.language == 'en':
        caminho_modelo = "models/transfer_learning/distilbert-base-cased-distilled-squad"
        model = Reader(pretrained_model_name_or_path=caminho_modelo, parm_dict_config=parm_dict_config_model)
    else:
        caminho_modelo = "models/transfer_learning/pierreguillou/bert-large-cased-squad-v1.1-portuguese"
        model = Reader(pretrained_model_name_or_path=caminho_modelo, parm_dict_config=parm_dict_config_model)

    num_question = 0
    f1_at_3 = f1 = exact_match = exact_match_at_3 =  0.
    print(f"Evalating in dataset {parm_dataset.name} model \n{model.info} ")
    tstart = time.time()
    for article in parm_dataset.data:
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
                if num_question > parm_dict_config_eval['num_question_max']:
                    break
            if num_question > parm_dict_config_eval['num_question_max']:
                break
        if num_question > parm_dict_config_eval['num_question_max']:
            break

    tend = time.time()
    exact_match = round(100.0 * exact_match / num_question,2)
    f1 = round(100.0 * f1 / num_question,2)
    exact_match_at_3 = round(100.0 * exact_match_at_3 / num_question,2)
    f1_at_3 = round(100.0 * f1_at_3 / num_question,2)
    dict_return = {'EM': round(exact_match,2),\
        'F1': round(f1,2), \
        'EM@3': round(exact_match_at_3,2),\
        'F1@3': round(f1_at_3,2), \
        'num_questions': num_question, \
        'duration total (s)': int(round(tend - tstart, 0)), \
        'duration per question (ms)': int(round(1000*(tend - tstart)/num_question, 0)) }

    print(f"Evaluation result: {dict_return}")
    return dict_return



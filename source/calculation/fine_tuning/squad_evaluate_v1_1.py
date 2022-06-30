import time
import copy


from source.calculation.metric import qa_metric
from source.calculation.fine_tuning import responder_extracao
from source.calculation.fine_tuning.modelo_reader import reader_en, reader_pt

def evaluate_finetuning(parm_dataset):
    json_fine_tuning = copy.deepcopy(responder_extracao.json_extracao_resposta)
    json_fine_tuning["tamanho_max_resposta"] = 30
    json_fine_tuning["top_k"] = 3
    if parm_dataset.language == 'en':
        model = reader_en
    else:
        model = reader_pt
    qtd_pergunta=0
    f1 = exact_match = total = 0
    tstart = time.time()
    print(model.info)
    for article in parm_dataset.data:
        for paragraph in article['paragraphs']:
            json_fine_tuning['texto_contexto'] = paragraph['context']
            for qa in paragraph['qas']:
                qtd_pergunta += 1
                total += 1
                # calcular resposta
                json_fine_tuning['texto_pergunta'] = qa['question']
                resposta = responder_extracao.responder(model, json_fine_tuning)
                # print(f"resposta {resposta}")
                resposta_predita = resposta[0]['texto_resposta']
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                exact_match += qa_metric.metric_max_over_ground_truths(
                    qa_metric.exact_match_score, resposta_predita, ground_truths)
                f1 += qa_metric.metric_max_over_ground_truths(
                    qa_metric.f1_score, resposta_predita, ground_truths)
    tend = time.time()
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    dict_return = {'exact_match': round(exact_match,4), 'f1': round(f1,4), \
        'duration total': round(tend - tstart, 4), \
        'duration per question': round((tend - tstart)/qtd_pergunta, 4) }
    print(f"{dict_return}")
    return dict_return

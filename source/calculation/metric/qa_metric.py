"""
QA Metrics

Fonte apoio:
 https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Metrics-for-QA

 Official evaluation script for v1.1 of the SQuAD dataset.
    wget https://github.com/allenai/bi-att-flow/archive/master.zip
        paste:  bi-att-flow-master\squad

Metrics for QA
There are two dominant metrics used by many question answering datasets, including SQuAD: exact match (EM) and F1 score. These scores are computed on individual question+answer pairs. When multiple correct answers are possible for a given question, the maximum score over all possible correct answers is computed. Overall EM and F1 scores are computed for a model by averaging over the individual example scores.

Exact Match
This metric is as simple as it sounds. For each question+answer pair, if the characters of the model's prediction exactly match the characters of (one of) the True Answer(s), EM = 1, otherwise EM = 0. This is a strict all-or-nothing metric; being off by a single character results in a score of 0. When assessing against a negative example, if the model predicts any text at all, it automatically receives a 0 for that example.

F1
F1 score is a common metric for classification problems, and widely used in QA. It is appropriate when we care equally about precision and recall. In this case, it's computed over the individual words in the prediction against those in the True Answer. The number of shared words between the prediction and the truth is the basis of the F1 score: precision is the ratio of the number of shared words to the total number of words in the prediction, and recall is the ratio of the number of shared words to the total number of words in the ground truth.


Let's see how these metrics work in practice. We'll load up a fine-tuned model (this one, to be precise) and its tokenizer, and compare our predictions against the True Answers.
"""
import string

exclude = set(string.punctuation)
import re
m = re.search('(?<=abc)def', 'abcdef')

def normalize_answer(s):

    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):

        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()



    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)
    num_same = len(common_tokens)
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = 1.0 * num_same / len(pred_tokens)
    rec = 1.0 * num_same / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def metric_score_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return scores_for_ground_truths

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def calculate_metrics(parm_list_answer, parm_list_ground_truths):
    """
    Return dict with metrics got considering
     parm_list_answer and parm_list_ground_truths
    """
    assert len(parm_list_answer)>=3, f"To get EM@3 and F1@3 it must have at least 3 answers. It has {len(parm_list_answer)} answers"
    predict_answer = parm_list_answer[0]['texto_resposta']
    em = metric_max_over_ground_truths(
        exact_match_score, predict_answer, parm_list_ground_truths)
    f1 = metric_max_over_ground_truths(
        f1_score, predict_answer, parm_list_ground_truths)

    # print(f"parm_list_ground_truths:\n {parm_list_ground_truths}")
    # print(f"parm_list_answer:\n {parm_list_answer[:3]}")
    # calculando @3 (para 3 primeiras respostas diferentes)
    em_at_3 = em
    f1_at_3 = f1
    # print(f"ndx=0 em_at_3:{em_at_3} f1_at_3:{f1_at_3}")
    for ndx_resposta in (1,2):
        predict_answer = parm_list_answer[ndx_resposta]['texto_resposta']
        em_ndx = metric_max_over_ground_truths(
            exact_match_score, predict_answer, parm_list_ground_truths)
        f1_ndx = metric_max_over_ground_truths(
            f1_score, predict_answer, parm_list_ground_truths)
        if em_ndx > em_at_3:
            em_at_3 = em_ndx
        if f1_ndx > f1_at_3:
            f1_at_3 = f1_ndx
        # print(f"ndx={ndx_resposta} em_at_3:{em_at_3} f1_at_3:{f1_at_3}")

    return {'EM':em, 'F1':f1, 'EM@3':em_at_3, 'F1@3':f1_at_3}


def calculate_metrics_grouped(lista_resposta, target_dataset, num_question:int):
    metric_per_question = {}
    f1_at_3 = f1 = exact_match = exact_match_at_3 =  0.
    for ndx in range(len(lista_resposta)):
        list_ground_truth = target_dataset[ndx]['answer_text']
        # ground_truths = list(map(lambda x: x['text'], list_ground_truth))
        metric_calculated = calculate_metrics(lista_resposta[ndx], list_ground_truth)
        metric_per_question[target_dataset[ndx]['id']] =  metric_calculated
        # print(f"metric_calculated {metric_calculated}")
        exact_match += metric_calculated['EM']
        f1 += metric_calculated['F1']
        exact_match_at_3 += metric_calculated['EM@3']
        f1_at_3 += metric_calculated['F1@3']

    exact_match = round(100.0 * exact_match / num_question,2)
    f1 = round(100.0 * f1 / num_question,2)
    exact_match_at_3 = round(100.0 * exact_match_at_3 / num_question,2)
    f1_at_3 = round(100.0 * f1_at_3 / num_question,2)

    return {'F1':f1, 'EM':exact_match, 'EM@3':exact_match_at_3, 'F1@3':f1_at_3 }, metric_per_question

"""
se envolver squad2.0
def get_gold_answers(example):
    #helper function that retrieves all possible true answers from a squad2.0 example

    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example -
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]

    return gold_answers
"""

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
print("nova importação")
m = re.search('(?<=abc)def', 'abcdef')
print(m.group(0))

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

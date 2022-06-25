print('oi')
import os, sys
print(sys.path)
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'.\.'))
import copy

from transformers.data.processors.squad import SquadV1Processor # SquadV2Processor

# this processor loads the SQuAD2.0 dev set examples
processor = SquadV1Processor()
examples = processor.get_dev_examples("data/dataset/squad/", filename="dev-v1.1.json")
print(len(examples))


def display_example(qid):
    from pprint import pprint

    idx = qid_to_example_index[qid]
    q = examples[idx].question_text
    c = examples[idx].context_text
    a = [answer['text'] for answer in examples[idx].answers]

    print(f'Example {idx} of {len(examples)}\n---------------------')
    print(f"Q: {q}\n")
    print("Context:")
    pprint(c)
    print(f"\nTrue Answers:\n{a}")


# generate some maps to help us identify examples of interest
qid_to_example_index = {example.qas_id: i for i, example in enumerate(examples)}
qid_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if has_answer]
no_answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if not has_answer]



"""


from transformers import AutoModelForQuestionAnswering


from source.fine_tuning.responder_extracao import responder_extracao, json_exemplo  # pylint: disable=wrong-import-position # precisa dos sys.path antes
from source.fine_tuning.modelo_reader import lista_modelos_reader

for modelo_reader in lista_modelos_reader:
    print(modelo_reader)
    # reader['modelo_reader'].score()

json_exemplo_pergunta_resposta = copy.deepcopy(json_exemplo)
json_exemplo_pergunta_resposta["tamanho_max_resposta"] = 30

resposta = responder_extracao(json_exemplo_pergunta_resposta)
print(f"resposta {resposta}")
"""
import json
import copy

class SquadDataset(object): # pylint: disable=missing-class-docstring # código de origem externa
    def __init__(self, data, file_name:str, language):
        self._data = data
        self._file_name = file_name
        self._language = language
        self._num_questions = 10570

    @property
    def data(self):
        return self._data

    @property
    def file_name(self):
        return self._file_name

    @property
    def language(self):
        return self._language

    @property
    def num_questions(self):
        return self._num_questions

def avaliar_dataset_squad_1_1(parm_dataset:SquadDataset, parm_lista_pergunta_imprimir=None):
    # parm_lista_pergunta_imprimir: exemplo repetidas ['571a52cb4faf5e1900b8a96b', '5730b9dc8ab72b1400f9c70f', '571a50df4faf5e1900b8a960']
    if not parm_lista_pergunta_imprimir:
        parm_lista_pergunta_imprimir = []
    print(f"Levantando quantitativo e verificando inconsistências no dataset {parm_dataset.file_name}")
    qtd_erro_fora_texto = qtd_exclusao = qtd_paragrafo = qtd_artigo = qtd_resposta = qtd_erro_fora_posicao = qtd_pergunta = qtd_pergunta_repetida = 0
    perguntas_repetidas = set()
    qtd_respostas_para_id_pergunta = {} # deixar os pares repetidos com maior número de respostas
    perguntas_unicas_id = set()
    for artigo in parm_dataset.data:
        qtd_artigo += 1
        for paragrafo in artigo['paragraphs']:
            contexto = paragrafo['context']
            qtd_paragrafo += 1
            for ndx_qas, pergunta_respostas in enumerate(paragrafo['qas']):
                qtd_respostas_para_id_pergunta[pergunta_respostas['id']] = len(pergunta_respostas['answers'])

                qtd_pergunta += 1
                if pergunta_respostas['id'] in parm_lista_pergunta_imprimir:
                    print(f"contexto: \n{contexto}")
                    print(f"investigar: \n{pergunta_respostas}")

                for resposta in pergunta_respostas['answers']:
                    qtd_resposta += 1
                    if resposta['text'] not in contexto:
                        print(f"Resposta não está no texto artigo {artigo['title']} id: {pergunta_respostas['id']}")
                        qtd_erro_fora_texto += 1
                    if resposta['text'] != contexto[resposta['answer_start']:resposta['answer_start']+len(resposta['text'])]:
                        qtd_erro_fora_posicao += 1

                if pergunta_respostas['id'] not in perguntas_unicas_id:
                    perguntas_unicas_id.add(pergunta_respostas['id'])
                else:
                    qtd_pergunta_repetida += 1
                    perguntas_repetidas.add(pergunta_respostas['id'])


    print(f"Totais artigos:{qtd_artigo} paragrafos:{qtd_paragrafo} perguntas:{qtd_pergunta} respostas:{qtd_resposta}")
    print(f"Erros perguntas: repetidas {qtd_pergunta_repetida}")
    print(f"Erro respostas fora do texto: {qtd_erro_fora_texto} fora da posição {qtd_erro_fora_posicao}")


def excluir_duplicata_dataset_squad_1_1(parm_dataset:SquadDataset):
    print(f"Levantando quantitativo e verificando inconsistências no dataset {parm_dataset.file_name}")
    qtd_erro_fora_texto = qtd_exclusao = qtd_paragrafo = qtd_artigo = qtd_resposta = qtd_erro_fora_posicao = qtd_pergunta = qtd_pergunta_repetida = 0
    perguntas_repetidas = set()
    qtd_respostas_para_id_pergunta = {} # deixar os pares repetidos com maior número de respostas
    perguntas_unicas_id = set()
    for artigo in parm_dataset.data:
        qtd_artigo += 1
        for paragrafo in artigo['paragraphs']:
            contexto = paragrafo['context']
            qtd_paragrafo += 1
            # qas_original = copy.copy(paragrafo['qas'])
            lista_id_a_excluir = []
            for ndx_qas, pergunta_respostas in enumerate(paragrafo['qas']):
                qtd_respostas_para_id_pergunta[pergunta_respostas['id']] = len(pergunta_respostas['answers'])
                qtd_pergunta += 1
                for resposta in pergunta_respostas['answers']:
                    qtd_resposta += 1
                    if resposta['text'] not in contexto:
                        print(f"Resposta não está no texto artigo {artigo['title']} id: {pergunta_respostas['id']}")
                        qtd_erro_fora_texto += 1
                    if resposta['text'] != contexto[resposta['answer_start']:resposta['answer_start']+len(resposta['text'])]:
                        qtd_erro_fora_posicao += 1

                if pergunta_respostas['id'] not in perguntas_unicas_id:
                    perguntas_unicas_id.add(pergunta_respostas['id'])
                else:
                    qtd_pergunta_repetida += 1
                    perguntas_repetidas.add(pergunta_respostas['id'])
                    if qtd_respostas_para_id_pergunta[pergunta_respostas['id']] >= len(pergunta_respostas['answers']):
                        lista_id_a_excluir.append(ndx_qas)
                    else:
                        print(f" qtd de repetição posterior é maior {len(pergunta_respostas['answers'])} > qtd_respostas_para_id_pergunta[pergunta_respostas['id']]")
                if ndx_qas == len(paragrafo['qas']) - 1: # só no final do conjunto de qas para não atrapalhar o for
                    qtd_exclusao_conjunto_qas = 0
                    for pos_qas_excluir in lista_id_a_excluir:
                        pos_ajustada_a_excluir = pos_qas_excluir - qtd_exclusao_conjunto_qas
                        # print(f"Deletado pos_qas_excluir {pos_qas_excluir} id_pergunta:{paragrafo['qas'][pos_ajustada_a_excluir]['id']}")
                        del paragrafo['qas'][pos_ajustada_a_excluir]
                        qtd_exclusao_conjunto_qas += 1
                        qtd_exclusao += 1


    print(f"Totais artigos:{qtd_artigo} paragrafos:{qtd_paragrafo} perguntas:{qtd_pergunta} respostas:{qtd_resposta}")
    print(f"Erros perguntas: repetidas {qtd_pergunta_repetida} excluídas {qtd_exclusao}")
    print(f"Erro respostas fora do texto: {qtd_erro_fora_texto} fora da posição {qtd_erro_fora_posicao}")



def imprimir_exemplo_dataset_squad_1_1(parm_dataset:SquadDataset):
    print(f"Dataset {parm_dataset.file_name} \n#artigos: {len(parm_dataset.data)}")
    for artigo in parm_dataset.data:
        print(f"\n****** artigo ****** \n  {artigo.keys()} \n  1o artigo \n    #paragrafos {len(artigo['paragraphs'])};\n    Artigo:\n      {artigo['title']}, ")
        for paragrafo in artigo['paragraphs']:
            print(f"\n    ****** paragrafo ******\n      {paragrafo.keys()} \n      1o paragrafo: \n      #perguntas: {len(paragrafo['qas'])} \n      Parágrafo: \n        {paragrafo['context']}")
            for pergunta_respostas in paragrafo['qas']:
                print(f"\n      ****** pergunta ****** \n        {pergunta_respostas.keys()} \n        1a pergunta: \n        Qtd respostas: {len(pergunta_respostas['answers'])}; \n        Pergunta: \n          {pergunta_respostas['question']}")
                for resposta in pergunta_respostas['answers']:
                    print(f"\n        ****** resposta ****** \n          {resposta.keys()} \n          1a resposta: \n          {resposta}")
                    break
                ground_truths = list(map(lambda x: x['text'], pergunta_respostas['answers']))
                print(f"          Exemplo 1o ground_truths de respostas: \n            {ground_truths}")
                break
            break
        break

def carregar_squad_1_1(parm_language:str):
    # path_project = '~/fontes/exqa-complearning/'
    path_project = '/home/borela/fontes/exqa-complearning/'
    assert parm_language in ['en', 'pt'], "parm_language must be in ['en', 'pt']"
    if parm_language == 'en':
        nome_dataset = 'dev-v1.1.json'
        path_dataset_file = path_project+"data/dataset/squad/"+nome_dataset
    else:
        nome_dataset = 'dev-v1.1-pt.json'
        path_dataset_file = path_project+"data/dataset/squad/"+nome_dataset
    with open(path_dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    return SquadDataset(data=dataset, file_name=nome_dataset, language = parm_language)

"""
May be used

from transformers.data.processors.squad import SquadV1Processor # SquadV2Processor

processor = SquadV1Processor()
examples = processor.get_dev_examples(path_project+"data/dataset/squad/", filename="dev-v1.1.json")
print(len(examples))


from datasets import load_dataset
dataset = load_dataset("squad_related")



# generate some maps to help us identify examples of interest
qid_to_example_index = {example.qas_id: i for i, example in enumerate(examples)}
qid_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if has_answer]
# no_answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if not has_answer]



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

"""
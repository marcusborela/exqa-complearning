"""
  reference about dataset
    https://huggingface.co/docs/datasets/v1.4.0/processing.html

"""
import json
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from source.calculation import util_modelo

PATH_DATASET = '/home/borela/fontes/exqa-complearning/data/dataset/squad/'

class SquadDataset(object):
    def __init__(self, nested_json, dataset, file_name:str, language):
        self._nested_json = nested_json
        self._dataset = dataset
        self._file_name = file_name
        self._language = language
        self._num_questions = 10570
        # path_flatten_json_file =  PATH_DATASET+'flatten-'+self._file_name
        # df_dataset = pd.read_json(path_flatten_json_file)
        # self._df = pd.json_normalize(df_dataset['data'])
        self._df = self._dataset.to_pandas()

    @property
    def nested_json(self):
        return self._nested_json

    @property
    def dataset(self):
        return self._dataset

    @property
    def name(self):
        return 'squad-dev-v1.1-'+self._language

    @property
    def file_name(self):
        return self._file_name

    @property
    def language(self):
        return self._language

    @property
    def num_questions(self):
        return self._num_questions

    @property
    def df(self):
        return self._df

    def print_total_per_question_type(self):
        counts = {}
        question_types = ["What ", "How ", "Is ", "Does ", "Do ", "Was ", "Where ", "Why ", "Which "]

        for question_type in question_types:
            counts[question_type] = self._df["question"].str.startswith(question_type).value_counts()[True]
            print(f"{question_type}: {counts[question_type]}")
            n_sample = min(2, counts[question_type] )
            for question in (
                self._df[self._df.question.str.startswith(question_type)]
                .sample(n=n_sample, random_state=42)['question']):
                print(question)

        pd.Series(counts).sort_values().plot.barh()
        plt.title("Frequency of Question Types")
        plt.show()

    def print_example(self):
        print(f"Dataset {self._file_name} \n#artigos: {len(self._nested_json)}")
        for artigo in self._nested_json:
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

    def print_histogram_num_tokens_given_tokenizer(self, parm_tokenizer):

        def compute_input_length(row):
            inputs = parm_tokenizer(row["question"], row["context"])
            return len(inputs["input_ids"])

        def compute_input_length_question(row):
            inputs = parm_tokenizer(row["question"])
            return len(inputs["input_ids"])


        df = pd.DataFrame()
        df["n_tokens_question_context"] = self._df.apply(compute_input_length, axis=1)
        df["n_tokens_question"] = self._df.apply(compute_input_length_question, axis=1)
        print("Number of tokens:")
        print(df.describe())

        fig, ax = plt.subplots()
        df["n_tokens_question_context"].hist(bins=100, grid=False, ec="C0", ax=ax)
        max_val = df["n_tokens_question_context"].max()
        plt.xlabel("Number of tokens in question-context pair")
        ax.axvline(x=max_val, ymin=0, ymax=1, linestyle="--", color="C1",
                label="Maximum sequence length")
        plt.legend()
        plt.ylabel("Count")
        plt.show()

        fig, ax = plt.subplots()
        max_val = df["n_tokens_question"].max()
        df["n_tokens_question"].hist(bins=100, grid=False, ec="C0", ax=ax)
        plt.xlabel("Number of tokens in question")
        ax.axvline(x=max_val, ymin=0, ymax=1, linestyle="--", color="C1",
                label="Maximum sequence length")
        plt.legend()
        plt.ylabel("Count")
        plt.show()

def evaluate_dataset_squad_1_1(parm_dataset:SquadDataset, parm_lista_pergunta_imprimir=None):
    # parm_lista_pergunta_imprimir: exemplo repetidas ['571a52cb4faf5e1900b8a96b', '5730b9dc8ab72b1400f9c70f', '571a50df4faf5e1900b8a960']
    if not parm_lista_pergunta_imprimir:
        parm_lista_pergunta_imprimir = []
    print(f"Levantando quantitativo e verificando inconsistências no dataset {parm_dataset.file_name}")
    qtd_erro_fora_texto = qtd_exclusao = qtd_paragrafo = qtd_artigo = qtd_resposta = qtd_erro_fora_posicao = qtd_pergunta = qtd_pergunta_repetida = 0
    perguntas_repetidas = set()
    qtd_respostas_para_id_pergunta = {} # deixar os pares repetidos com maior número de respostas
    perguntas_unicas_id = set()
    for artigo in parm_dataset.nested_json:
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

def _del_duplicate_dataset_squad_1_1(parm_dataset:SquadDataset):
    print(f"Levantando quantitativo e verificando inconsistências no dataset {parm_dataset.file_name}")
    qtd_erro_fora_texto = qtd_exclusao = qtd_paragrafo = qtd_artigo = qtd_resposta = qtd_erro_fora_posicao = qtd_pergunta = qtd_pergunta_repetida = 0
    perguntas_repetidas = set()
    qtd_respostas_para_id_pergunta = {} # deixar os pares repetidos com maior número de respostas
    perguntas_unicas_id = set()
    for artigo in parm_dataset.nested_json:
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

def load_squad_dataset_1_1(parm_language:str):
    # path_project = '~/fontes/exqa-complearning/'
    assert parm_language in ['en', 'pt'], "parm_language must be in ['en', 'pt']"
    if parm_language == 'en':
        nome_dataset = 'dev-v1.1.json'
        path_nested_json_file = PATH_DATASET+nome_dataset
        path_flatten_json_file =  PATH_DATASET+'flatten-'+nome_dataset
    else:
        nome_dataset = 'dev-v1.1-pt.json'
        path_nested_json_file = PATH_DATASET+nome_dataset
        path_flatten_json_file =  PATH_DATASET+'flatten-'+nome_dataset

    with open(path_nested_json_file) as dataset_file:
        nested_json = json.load(dataset_file)


    datasets = load_dataset('json',
                        data_files={'test': path_flatten_json_file, \
                                    },
                        field='data')
    return SquadDataset(nested_json= nested_json['data'], dataset= datasets['test'], file_name=nome_dataset, language = parm_language)

# Code below used just one time for correction/convertion

def _convert_json_in_dataset(parm_dataset:SquadDataset):

    # Iterating through the json list
    entry_list = list()

    for row in parm_dataset.nested_json:
        title = row['title']

        for paragraph in row['paragraphs']:
            context = paragraph['context']

            for qa in paragraph['qas']:
                entry = {}

                qa_id = qa['id']
                question = qa['question']
                answers = qa['answers']

                entry['id'] = qa_id
                entry['title'] = title.strip()
                entry['context'] = context.strip()
                entry['question'] = question.strip()

                answer_starts = [answer["answer_start"] for answer in answers]
                answer_texts = [answer["text"].strip() for answer in answers]

                entry['answer_start'] = answer_starts
                entry['answer_text'] = answer_texts

                entry_list.append(entry)

    new_dict = {}
    new_dict['data'] = entry_list
    file_name = 'flatten-' + str(parm_dataset.file_name)
    with open(PATH_DATASET+file_name, 'w') as json_file:
        json.dump(new_dict, json_file)

def _generate_dataset_from_json():
    # squad_dataset_en = load_squad_dataset_1_1(parm_language='en')
    squad_dataset_pt = load_squad_dataset_1_1(parm_language='pt')
    _convert_json_in_dataset(squad_dataset_pt)


def _correct_and_save_dataset_without_duplication(parm_dataset:SquadDataset, parm_file_name):
    """
    Salva arquivo sem duplicatas em path_project/data/dataset/squad/
    """
    _del_duplicate_dataset_squad_1_1(parm_dataset)
    # path_project = '~/fontes/exqa-complearning/'
    path_dataset_file = PATH_DATASET+parm_file_name
    dict_json = {"version": "1.1"}
    dict_json['data']=parm_dataset.nested_json
    with open(path_dataset_file,'w', encoding='utf-8') as dataset_file:
        json.dump(dict_json, dataset_file)

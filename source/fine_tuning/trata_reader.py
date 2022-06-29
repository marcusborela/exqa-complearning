"""
Código para criação dos modelos de reranking
"""
import os
import torch
from copy import deepcopy
from typing import  List, Dict
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline
from  source.fine_tuning.util_modelo import Text, Query

# import transformers
# print(f"transformers.__version__ {transformers.__version__}")

"""
Comando abaixo para tratar msg de advertência:
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
            - Avoid using `tokenizers` before the fork if possible
            - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Sobre paralelismo:
   https://docs.allennlp.org/main/api/data/data_loaders/multiprocess_data_loader/
"""
os.environ['TOKENIZERS_PARALLELISM'] = "0"

BATCH_SIZE_PADRAO = 16
# para debugar erros gpu
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def formatar_retirar_duplicatas(lista_resposta:List[Dict])-> List[Dict]:
    """
    Retira respostas duplicatas, acumula score e
        gera posições relativas ao início do documento
    Consideradas duplicadas as respostas que possuem mesmo texto de resposta
    Serão excluídas as que tiverem mesma posição (início e fim)
    E colocadas as referências diferentes em "lista_referencia".
    A lista é retornada em ordem decrescente de score.
    parametro:
    lista_resposta:
        [{'score': 9999,
         'start': 99,
         'end': 99,
         'answer': 'xxxx'}, ...]

    Return:
        [{'texto_resposta': 'xxxx'},
         'score': 9999,  # acumulado
         # marcadores de posição relativa dentro do documento
         'lista_referencia':[['uuid 2', 56, 65], ...]

    """
    # print(f"lista_resposta_ordenada: {lista_resposta_ordenada}")
    lista_referencia = {}
    score = {}
    set_posicao = set()  # pares de posição distintos já carregados
    for resposta in lista_resposta:
        if resposta['answer'] not in lista_referencia:
            set_posicao.add((resposta['start'], resposta['end']))
            lista_referencia[resposta['answer']] = [[resposta['start'], resposta['end']]]
            score[resposta['answer']] = resposta['score']
        else:
            score[resposta['answer']] += resposta['score']
            if (resposta['start'], resposta['end']) not in set_posicao:
                lista_referencia[resposta['answer']].append([resposta['start'], resposta['end']])

    lista_resposta_saida = []
    for answer in score.keys():
        resp = {}
        resp['texto_resposta']= answer
        resp['score'] = round(score[answer], 6) # limitando em 6 casas decimais. Até para os testes passarem.
        for referencia in lista_referencia[answer]:
            if 'lista_referencia' not in resp:
                resp['lista_referencia'] = [[referencia[0], referencia[1]]]
            else:
                resp['lista_referencia'].append([referencia[0], referencia[1]])
        lista_resposta_saida.append(resp)

    return sorted(lista_resposta_saida, key=lambda x: x['score'], reverse=True)


class Reader(): # pylint: disable=missing-class-docstring
    def __init__(self,
                 pretrained_model_name_or_path: str  = 'castorini/monot5-base-msmarco',
                 use_amp = False): # Automatic Mixed Precision
        self.model = self.get_model(pretrained_model_name_or_path)
        self.tokenizer = self.get_tokenizer(pretrained_model_name_or_path)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp
        self.pipe = pipeline("question-answering", model=self.model,\
                             tokenizer=self.tokenizer,\
                             device=0, framework='pt')
        self.doc_stride = 30
        self.handle_impossible_answer = False

    @staticmethod
    def get_model(pretrained_model_name_or_path: str, # pylint: disable=missing-function-docstring
                  *args, device: str = None, **kwargs) -> AutoModelForQuestionAnswering:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device) # pylint: disable=broad-except
        # Since we are using our model only for inference, switch to `eval` mode:
        return AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path,
                                                          *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str, # pylint: disable=missing-function-docstring
                      *args, batch_size: int = BATCH_SIZE_PADRAO, **kwargs) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs)


    def answer(self, texto_pergunta: str, texto_contexto: str, parm_topk:int=1, parm_max_answer_length:int=40) -> List[Dict]:

        respostas = self.pipe(question=texto_pergunta,
            context=texto_contexto,
            handle_impossible_answer=self.handle_impossible_answer,
            top_k=parm_topk,
            doc_stride = self.doc_stride,
            max_seq_len = self.model.config.max_position_embeddings,
            max_question_len = len(texto_pergunta), # número de tokens.. #chars>#tokens
            max_answer_len = parm_max_answer_length
            )
        if parm_topk == 1:
            lista_respostas = [respostas]
        else:
            lista_respostas = respostas

        lista_respostas = formatar_retirar_duplicatas(lista_respostas)

        return lista_respostas



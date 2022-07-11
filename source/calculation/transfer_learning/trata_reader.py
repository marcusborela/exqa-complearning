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
from source.calculation import util_modelo

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
         'lista_referencia':[[56, 65], ...]

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
    _dict_parameters_example = {"num_doc_stride":128,\
               "num_top_k":3, \
               "num_max_answer_length":30, \
               "if_handle_impossible_answer":False, \
               "num_factor_multiply_top_k":3}

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 parm_dict_config:Dict):
        if parm_dict_config is None:
            parm_dict_config = deepcopy(Reader._dict_parameters_example)
        msg_dif = util_modelo.compare_dicionarios_chaves(Reader._dict_parameters_example, parm_dict_config,
            'Esperado', 'Encontrado')
        assert msg_dif == "", f"Estrutura esperada de parm_dict_config não corresponde ao esperado {msg_dif}"

        self.name_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.name_device)
        # self.device = next(self.model.parameters(), None).device

        # Since we are using our model only for inference, switch to `eval` mode:
        self.model = self.get_model(pretrained_model_name_or_path).to(self.device).eval()

        self.tokenizer = self.get_tokenizer(pretrained_model_name_or_path)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # não usado # Automatic Mixed Precision self.use_amp = use_amp
        self.pipe = pipeline("question-answering", model=self.model,\
                             tokenizer=self.tokenizer,\
                             device=self.device, framework='pt')
        self.num_top_k = parm_dict_config["num_top_k"]
        self.num_doc_stride = parm_dict_config["num_doc_stride"]
        self.if_handle_impossible_answer = parm_dict_config["if_handle_impossible_answer"]
        self.num_max_answer_length = parm_dict_config["num_max_answer_length"]
        self.num_factor_multiply_top_k = parm_dict_config["num_factor_multiply_top_k"]

    @property
    def info(self):
        return {"name":self.pretrained_model_name_or_path,\
                "device": self.name_device,\
                "top_k": self.num_top_k,\
                "doc_stride": self.num_doc_stride,\
                "factor_multiply_top_k": self.num_factor_multiply_top_k,\
                "handle_impossible_answer":self.if_handle_impossible_answer,\
                "max_answer_length":self.num_max_answer_length,\
                "max_seq_len": self.model.config.max_position_embeddings}

    @staticmethod
    def get_model(pretrained_model_name_or_path: str,
                  *args, **kwargs) -> AutoModelForQuestionAnswering:
        return AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path,
                                                          *args, **kwargs)

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str,
                      *args, batch_size: int = 16, **kwargs) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs)

    def answer(self, texto_pergunta: str, texto_contexto: str) -> List[Dict]:

        respostas = self.pipe(question=texto_pergunta,
            context=texto_contexto,
            handle_impossible_answer=self.if_handle_impossible_answer,
            # como há casos de repostas idênticas em posições diferentes,
            # precisamos pegar mais respostas para devolver top_k
            top_k= self.num_top_k*self.num_factor_multiply_top_k,
            # precisa??? self.max_seq_len = self.model.config.max_position_embeddings
            doc_stride = self.num_doc_stride,
            max_question_len = len(texto_pergunta), # número de tokens.. #chars>#tokens
            max_answer_len = self.num_max_answer_length
            )

        if not isinstance(respostas, list):
            lista_respostas = [respostas]
        else:
            lista_respostas = respostas

        lista_respostas = formatar_retirar_duplicatas(lista_respostas)

        if len(lista_respostas) < self.num_top_k:
            print(f"Warning: #answers=len(lista_respostas)<self.num_top_k {len(lista_respostas)} < {self.num_top_k}. Use num_factor_multiply_top_k if it is not desired.")

        return lista_respostas[:self.num_top_k]



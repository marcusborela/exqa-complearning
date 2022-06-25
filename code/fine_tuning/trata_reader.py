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
from  code.fine_tuning.util_modelo import Text, Query

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


def formatar_retirar_duplicatas(lista_resposta:List[Dict], docto_to_pos_inicio)-> List[Dict]:
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
         'id_docto': 'id1'
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
            lista_referencia[resposta['answer']] = [[resposta['id_docto'], resposta['start'], resposta['end']]]
            score[resposta['answer']] = resposta['score']
        else:
            score[resposta['answer']] += resposta['score']
            if (resposta['start'], resposta['end']) not in set_posicao:
                lista_referencia[resposta['answer']].append([resposta['id_docto'], resposta['start'], resposta['end']])


    lista_resposta_saida = []
    for answer in score.keys():
        resp = {}
        resp['texto_resposta']= answer
        resp['score'] = score[answer]
        for referencia in lista_referencia[answer]:
            docto = referencia[0]
            ini_docto = docto_to_pos_inicio[docto]
            pos_inicio = referencia[1] - ini_docto
            pos_fim = referencia[2] - ini_docto
            if 'lista_referencia' not in resp:
                resp['lista_referencia'] = [[docto, pos_inicio, pos_fim]]
            else:
                resp['lista_referencia'].append([docto, pos_inicio, pos_fim])
        lista_resposta_saida.append(resp)

    return sorted(lista_resposta_saida, key=lambda x: x['score'], reverse=True)


def filtrar_resposta_de_unico_documento(lista_resposta:List[Dict], pos_to_docto:{})-> List[Dict]:
    """
    Retira de lista_resposta respostas que perpassam mais de um documento
    E para as respostas que ficam, adiciona o código do documento
    """
    lista_resposta_filtrada = []
    for resposta in lista_resposta:
        # tratando o caso de cair no espaço entre documentos
        if resposta['start'] not in pos_to_docto:
            resposta['start'] += 1
        if resposta['end'] not in pos_to_docto:
            resposta['end'] -= 1
        if resposta['start'] in pos_to_docto and \
           resposta['end'] in pos_to_docto:
            docto_inicio = pos_to_docto[resposta['start']]
            docto_final = pos_to_docto[resposta['end']]
            if docto_inicio == docto_final:
                resposta['id_docto'] = docto_inicio
                lista_resposta_filtrada.append(resposta)
    return lista_resposta_filtrada


def concatenar_taggear_texto(texts: List[Text])-> str:
    """
    retorna texto concatenado e 2 dicts:
    pos_to_docto{posx:'cod1',
              posx+1:'cod1'
              ...
                }

    intervalo{'cod1':[ini, fim],
              'cod2':[ini, fim],
              ...}

    docto_to_pos_inicio{'cod':ini, ...}

    Para depois retirar respostas cujo inicio e fim não
    estejam em um únicio documento
    E para encontrar o documento que contém a resposta
    """
    intervalo_pos_docto = {}

    ndx_atual=0 # posição começa com zero não com 1
    for documento in texts:
        id_docto = str(documento.id_docto)
        tamanho_texto = len(documento.text)
        intervalo_pos_docto[id_docto] = [ndx_atual, ndx_atual + tamanho_texto - 1] # -1 pois começa do zero
        if ndx_atual==0:
            texto_total = documento.text
        else:
            texto_total += " " + documento.text
        # +1 por que tem o branco no meio
        # a posição ndx_atual + tamanho_texto não terá modelo associado
        ndx_atual += tamanho_texto + 1

    pos_to_docto = {}
    docto_to_pos_inicio = {}
    for doc_id, par_posicao in intervalo_pos_docto.items():
        docto_to_pos_inicio[doc_id]=par_posicao[0]
        for ndx in range(par_posicao[0],par_posicao[1]+1): # +1 pois não inclusivo
            pos_to_docto[ndx] = doc_id
    # print(f"Texto total: {texto_total}")
    return texto_total, pos_to_docto, docto_to_pos_inicio


class Reader(): # pylint: disable=missing-class-docstring
    def __init__(self,
                 pretrained_model_name_or_path: str  = 'castorini/monot5-base-msmarco',
                 use_amp = False): # Automatic Mixed Precision
        self.model = self.get_model(pretrained_model_name_or_path)
        self.tokenizer = self.get_tokenizer(pretrained_model_name_or_path)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp
        self.pipe = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer,\
                             device=0, framework='pt')
        self.doc_stride = 30

    def ler(self, query: Query, texts: List[Text]) -> List[Text]:
        """Sorts a list of texts
        """
        return sorted(self.answer(query, texts), key=lambda x: x['score'], reverse=True)

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


    def answer(self, query: Query, texts: List[Text], parm_topk:int=1, parm_max_answer_length:int=40) -> List[Dict]:

        texts = deepcopy(texts)

        texto_total, pos_to_docto, docto_to_pos_inicio = concatenar_taggear_texto(texts)

        respostas = self.pipe(question=query.text,
            context=texto_total,
            handle_impossible_answer=True,
            topk=parm_topk,
            doc_stride = self.doc_stride,
            max_seq_len = self.model.config.max_position_embeddings,
            max_question_len = len(query.text), # número de tokens.. #chars>#tokens
            max_answer_len = parm_max_answer_length
            )
        if parm_topk == 1:
            lista_respostas = [respostas]
        else:
            lista_respostas = respostas

        lista_respostas = filtrar_resposta_de_unico_documento(lista_respostas, pos_to_docto)

        lista_respostas = formatar_retirar_duplicatas(lista_respostas,docto_to_pos_inicio)

        return lista_respostas


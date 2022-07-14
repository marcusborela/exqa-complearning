"""
Código para utilização dos modelos de context learning
"""
##from dataclasses import dataclass
import os
from typing import  List, Dict
from copy import deepcopy
import torch
from transformers import pipeline
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
##from source.calculation import util_modelo

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
               "num_batch_size":20, \
               "if_handle_impossible_answer":False, \
               "num_factor_multiply_top_k":3}

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 parm_dict_config:Dict):
        if parm_dict_config is None:
            parm_dict_config = deepcopy(Reader._dict_parameters_example)
        ##msg_dif = util_modelo.compare_dicionarios_chaves(Reader._dict_parameters_example, parm_dict_config,
        ##    'Esperado', 'Encontrado')
        ##assert msg_dif == "", f"Estrutura esperada de parm_dict_config não corresponde ao esperado {msg_dif}"

        self.name_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.name_device)
        # self.device = next(self.model.parameters(), None).device

        # Since we are using our model only for inference, switch to `eval` mode:
        self.model = self.get_model(pretrained_model_name_or_path).to(self.device).eval()

        self.tokenizer = self.get_tokenizer(pretrained_model_name_or_path)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        # não usado # Automatic Mixed Precision self.use_amp = use_amp
        self.prompt_format = parm_dict_config['prompt_format']
        self.answer_start  = parm_dict_config['answer_start']
        self.answer_stop   = parm_dict_config['answer_stop']
        self.answer_skips  = parm_dict_config['answer_skips']
        self.generator_sample  = parm_dict_config['generator_sample']
        self.generator_temperature  = parm_dict_config['generator_temperature']
        self.answer_max_length  = parm_dict_config['answer_max_length']

        ##self.num_batch_size = parm_dict_config["num_batch_size"]
        ##self.num_top_k = parm_dict_config["num_top_k"]
        ##self.num_doc_stride = parm_dict_config["num_doc_stride"]
        ##self.if_handle_impossible_answer = parm_dict_config["if_handle_impossible_answer"]
        ##self.num_max_answer_length = parm_dict_config["num_max_answer_length"]
        ##self.num_factor_multiply_top_k = parm_dict_config["num_factor_multiply_top_k"]
        ##self.max_seq_len = self.model.config.max_position_embeddings

        ##self.pipe = pipeline("question-answering", model=self.model,\
        ##                     tokenizer=self.tokenizer,\
        ##                     device=self.device)

    @property
    def info(self):
        return {"name":self.pretrained_model_name_or_path,\
                "device": self.name_device,\
                "top_k": self.num_top_k,\
                "batch_size": self.num_batch_size,\
                "doc_stride": self.num_doc_stride,\
                "factor_multiply_top_k": self.num_factor_multiply_top_k,\
                "handle_impossible_answer":self.if_handle_impossible_answer,\
                "max_answer_length":self.num_max_answer_length,\
                "max_seq_len": self.max_seq_len}

    @staticmethod
    def get_model(pretrained_model_name_or_path: str,
                  *args, **kwargs) -> GPTNeoForCausalLM:
        return GPTNeoForCausalLM.from_pretrained(pretrained_model_name_or_path,
                                             *args, **kwargs)

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str,
                      *args, batch_size: int = 128, **kwargs) -> GPT2Tokenizer:
        return GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs)

    @staticmethod
    def parse_answer(tokens, start_sequence, stop_sequence, skips):
        tokens = tokens.squeeze()
        check_index = 0
        status = 'ignoring'
        start_index = 0
        stop_index = len(tokens)
        for index, token in enumerate(tokens):
            if status in ['ignoring', 'starting']:
                if token == start_sequence[check_index]:
                    status = 'starting'
                    check_index += 1
                    if check_index == len(start_sequence):
                        check_index = 0
                        if skips <= 0:
                            status = 'getting'
                            start_index = index + 1
                        else:
                            status = 'ignoring'
                            skips -= 1
                else:
                    check_index = 0
            elif status in ['getting', 'stopping']:
                if token == stop_sequence[check_index]:
                    status = 'stopping'
                    check_index += 1
                    if check_index == len(stop_sequence):
                        status = 'stop'
                        stop_index = index + 1 - len(stop_sequence)
                        break
                else:
                    check_index = 0
            
        return tokens[start_index:stop_index].unsqueeze(dim=0)

    def answer_one_question(self, texto_pergunta: str, texto_contexto: str) -> List[Dict]:

        assert isinstance(texto_pergunta, str), f"Expected just one question, not {type(texto_pergunta)}"
        assert isinstance(texto_contexto, str), f"Expected just one context, not {type(texto_contexto)}"

        prompt = self.prompt_format.replace('{context}', texto_contexto).replace('{question}', texto_pergunta)
        #print(prompt)

        #print('Tokenizing...')
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        #print('Generating answer...')
        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=self.generator_sample,
                temperature=self.generator_temperature,
                max_length=self.answer_max_length,
            )
        #gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        #print(gen_text)

        #print('Extracting answer...')
        start_sequence = self.tokenizer(self.answer_start, return_tensors="pt").input_ids.squeeze()
        stop_sequence = self.tokenizer(self.answer_stop, return_tensors="pt").input_ids.squeeze()
        answer_tokens = self.parse_answer(gen_tokens, start_sequence, stop_sequence, self.answer_skips)
        #print(answer_tokens)

        #print('Decoding answer...')
        answer_text = self.tokenizer.batch_decode(answer_tokens)[0].strip()

        #print(answer_text)
        return [{'prompt': prompt, 'gen_tokens': gen_tokens, 'answer_tokens': answer_tokens, 'answer_text': answer_text}]


    def answer(self, parm_dataset) -> List[Dict]:
        list_column_drop = []
        for column_name in parm_dataset.column_names:
            if column_name not in ('context', 'question'):
                list_column_drop.append(column_name)

        dts = parm_dataset.remove_columns(list_column_drop)
        for column_name in ('context', 'question'):
            assert column_name in dts.column_names, f"Dataset must have column {column_name}. It has {dts.column_names}"

        respostas = self.pipe(
            data=dts,
            handle_impossible_answer=self.if_handle_impossible_answer,
            # como há casos de repostas idênticas em posições diferentes,
            # precisamos pegar mais respostas para devolver top_k
            top_k= self.num_top_k*self.num_factor_multiply_top_k,
            # precisa??? self.max_seq_len = self.model.config.max_position_embeddings
            doc_stride = self.num_doc_stride,
            max_question_len = 50, # número de tokens.. pt: max 44, en: max 46
            max_answer_len = self.num_max_answer_length,
            max_seq_len = self.max_seq_len
            )

        if not isinstance(respostas, list):
            lista_respostas = [respostas]
        else:
            lista_respostas = respostas

        lista_resposta_formatada = []
        for ndx_qas, respostas_qas in enumerate(lista_respostas):
            if len(respostas_qas) < self.num_top_k:
                print(f"Warning: #answers=len(lista_respostas)<self.num_top_k ndx_qas: {ndx_qas} {len(respostas_qas)} < {self.num_top_k}. Use num_factor_multiply_top_k if it is not desired.")
            lista_resposta_formatada.append(formatar_retirar_duplicatas(respostas_qas)[:self.num_top_k])

        return lista_resposta_formatada

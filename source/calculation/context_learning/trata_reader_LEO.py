"""
Código para utilização dos modelos de context learning
"""
##from dataclasses import dataclass
import os
from typing import  List, Dict
from copy import deepcopy
import torch
from transformers import pipeline
# from transformers import GPT2Tokenizer, GPTJForCausalLM
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
##from source.calculation import util_modelo
from transformers import StoppingCriteria, StoppingCriteriaList

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



class KeywordsStoppingCriteria(StoppingCriteria):
    """
      Fonte: https://stackoverflow.com/questions/69403613/how-to-early-stop-autoregressive-model-with-a-list-of-stop-words
    """
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


class Reader(): # pylint: disable=missing-class-docstring

    """
        Fonte dos parâmetros

            https://github.com/huggingface/transformers/blob/main/src/transformers/generation_utils.py
            linha: 920

                length_penalty (`float`, *optional*, defaults to 1.0):
                 Exponential penalty to the length. 1.0 means that the beam score is penalized by the sequence length.
                 0.0 means no penalty. Set to values < 0.0 in order to encourage the model to generate longer
                 sequences, to a value > 0.0 in order to encourage the model to produce shorter sequences.

                temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.

                num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
    """

    _dict_parameters_example = {
                    # parâmetros tarefa
                    "num_top_k": 2,
                    # "num_batch_size": 1,  assumido 1 para answer one question
                    "num_max_answer_length":150,
                    # parâmetros transfer - q&a
                    "num_doc_stride": 0,
                    "num_factor_multiply_top_k": 0,
                    "if_handle_impossible_answer":False,
                    # parâmetros context - text generation
                    'answer_stop': ['.', '\n', '\n'],
                    "val_length_penalty":2,
                    'generator_sample': False,
                    'generator_temperature': 1,
                    'prompt_format': ('''Instrução: Com base no texto abaixo, responda à pergunta:

    Texto: {context}

    Pergunta: {question}
    Resposta:''')}

    def __init__(self,
                 parm_name_model: str,
                 parm_dict_config:Dict):


        # para evitar estouro memória gpu
        torch.cuda.empty_cache()

        # Since we are using our model only for inference, switch to `eval` mode:
        self.name_model = parm_name_model
        ##self.path_model = "models/context_learning/"+self.name_model
        self.path_model = self.name_model
        #self.model = self.get_model(self.path_model).to(self.device).eval()

        self.name_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.name_device)
        self.num_batch_size = 1 #  parm_dict_config["num_batch_size"]


        self.pipe = pipeline("text-generation", model=self.get_model(self.path_model).to(self.device).eval(),\
                              tokenizer=self.get_tokenizer(self.path_model),\
                              device=0,framework='pt')


        self.atualiza_parametros_resposta(parm_dict_config)

        # Se desejar salvar o modelo, para ficar mais rápido a carga
        # torch.save(self.pipe.model, path_model+"/pytorch_model_saved.pt")
        #   gptj6b: muda de 4 minutos para 12 segundos
        self.max_seq_len = self.pipe.model.config.max_position_embeddings


    def atualiza_parametros_resposta(self, parm_dict_config:Dict):

        if parm_dict_config is None:
            parm_dict_config = deepcopy(Reader._dict_parameters_example)
##        msg_dif = util_modelo.compare_dicionarios_chaves(Reader._dict_parameters_example, parm_dict_config,
##            'Esperado', 'Encontrado')
##        assert msg_dif == "", f"Estrutura esperada de parm_dict_config não corresponde ao esperado {msg_dif}"


        #self.tokenizer = self.get_tokenizer(self.path_model)
        # não usado # Automatic Mixed Precision self.use_amp = use_amp

        self.num_top_k = parm_dict_config.get("num_top_k", None)
        self.num_doc_stride = parm_dict_config.get("num_doc_stride", None)
        self.prompt_format = parm_dict_config.get('prompt_format', None)
        self.if_handle_impossible_answer = parm_dict_config.get("if_handle_impossible_answer", None)
        self.num_max_answer_length = parm_dict_config.get("num_max_answer_length", None)
        self.num_factor_multiply_top_k = parm_dict_config.get("num_factor_multiply_top_k", None)
        self.if_do_sample = parm_dict_config.get('generator_sample', None)
        self.num_temperature = parm_dict_config.get('generator_temperature', None)
        self.answer_stop = parm_dict_config.get("answer_stop", None)
        self.val_length_penalty = parm_dict_config.get("val_length_penalty", None)

        stop_ids = [self.tokenizer.encode(w)[0] for w in self.answer_stop]
        self.stop_criteria = KeywordsStoppingCriteria(stop_ids)

    @property
    def info(self):
        return {"name":self.name_model,\
                "device": self.name_device,\
                # "top_k": self.num_top_k,\
                "num_return_sequences": self.num_top_k,\
                "batch_size": self.num_batch_size,\
                "length_penalty": self.val_length_penalty, \
                "doc_stride": self.num_doc_stride,\
                "factor_multiply_top_k": self.num_factor_multiply_top_k,\
                "handle_impossible_answer":self.if_handle_impossible_answer,\
                "max_answer_length":self.num_max_answer_length,\
                "max_seq_len": self.max_seq_len}

    @property
    def tokenizer(self):
        return self.pipe.tokenizer


    @property
    def model(self):
        return self.pipe.model

    @staticmethod
    def get_model(path_model: str,
                  *args, **kwargs):
        if os.path.isfile(path_model+"/pytorch_model_saved.pt"):
            return torch.load(path_model+"/pytorch_model_saved.pt")
        else:
            # return GPTJForCausalLM.from_pretrained(path_model, *args, **kwargs)
            # return GPTNeoForCausalLM.from_pretrained(path_model, *args, **kwargs)
            return GPTNeoForCausalLM.from_pretrained(
            path_model,
            ##revision="float16",
            ##torch_dtype=torch.float16,
            low_cpu_mem_usage=True)

    @staticmethod
    def get_tokenizer(path_model: str,
                      *args, batch_size: int = 128, **kwargs):
        return GPT2Tokenizer.from_pretrained(path_model, use_fast=False, *args, **kwargs)

    def answer_one_question(self, texto_pergunta: str, texto_contexto: str) -> List[Dict]:

        assert isinstance(texto_pergunta, str), f"Expected just one question, not {type(texto_pergunta)}"
        assert isinstance(texto_contexto, str), f"Expected just one context, not {type(texto_contexto)}"
        assert self.num_batch_size == 1, f"self.num_batch_size must be 1 for answer_one_question"
        prompt = self.prompt_format.replace('{context}', texto_contexto).replace('{question}', texto_pergunta)



        #print('Tokenizing...')
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = len(tokenized.input_ids[0])

        print(f"chars(prompt): {len(prompt)}")
        print(f"tokens(prompt): {prompt_len}")


        respostas = self.pipe(prompt,
            return_tensors=False,
            return_text=True,
            return_full_text=False,
            clean_up_tokenization_spaces=True,
            # max_question_len = len(texto_pergunta), # número de tokens.. #chars>#tokens
            # max_length= prompt_len + self.num_max_answer_length,
            max_new_tokens = self.num_max_answer_length,
            min_length = 2,
            num_return_sequences = self.num_top_k,
            num_beams =  self.num_top_k,
            do_sample = self.if_do_sample,
            temperature = self.num_temperature,
            length_penalty = self.val_length_penalty, # self.num_length_penalty \
            stopping_criteria=StoppingCriteriaList([self.stop_criteria]),
            )

        return respostas

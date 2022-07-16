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

    @property
    def info(self):
        return {}

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

        prompt_len = len(input_ids[0])
        
        #print('Generating answer...')
        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=self.generator_sample,
                temperature=self.generator_temperature,
                max_length=prompt_len + self.answer_max_length,
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

        #print('Decoding generation...')
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0].strip()

        #print(answer_text)
        return [{'prompt': prompt, 'gen_tokens': gen_tokens, 'gen_text': gen_text, 'answer_tokens': answer_tokens, 'answer_text': answer_text}]

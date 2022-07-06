"""
rotinas utilitárias
"""
import random
import torch
import numpy


def compare_dicionarios(dict_1, dict_2, dict_1_name, dict_2_name, path=""):
    """Compare two dictionaries recursively to find non mathcing elements (keys and values)

    Args:
        dict_1: dictionary 1
        dict_2: dictionary 2

    Returns: difference

    Fonte:  https://stackoverflow.com/questions/27265939/comparing-python-dictionaries-and-nested-dictionaries/27266178

    """
    err = ''
    key_err = ''
    value_err = ''
    old_path = path
    for k in dict_1.keys():  # antes old_path + "[%s]" % k
        path = old_path + f"[{k}]"
        if not k in dict_2:
            key_err += f"Key {dict_2_name}{path} not in {dict_2_name}\n"
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                err += compare_dicionarios(dict_1[k],
                                           dict_2[k], 'd1', 'd2', path)
            else:
                if dict_1[k] != dict_2[k]:
                    value_err += f"Value of {dict_1_name}{path} ({dict_1[k]}) not same as {dict_2_name}{path} ({dict_2[k]})\n"

    for k in dict_2.keys():
        path = old_path + f"[{k}]"
        if not k in dict_1:
            key_err += f"Key {dict_1_name}{path} not in {dict_2_name}\n"

    return key_err + value_err + err

def compare_dicionarios_chaves(dict_1, dict_2, dict_1_name, dict_2_name, path=""):
    """Compare two dictionaries recursively to find non mathcing keys

    Args:
        dict_1: dictionary 1
        dict_2: dictionary 2

    Returns: difference

    Fonte:  https://stackoverflow.com/questions/27265939/comparing-python-dictionaries-and-nested-dictionaries/27266178

    """
    err = ''
    key_err = ''
    old_path = path
    for k in dict_1.keys():
        path = old_path + f"[{k}]"
        if not k in dict_2:
            key_err += f"Key {dict_1_name}{path} not in {dict_2_name}\n"
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                err += compare_dicionarios_chaves(
                    dict_1[k], dict_2[k], 'd1', 'd2', path)

    for k in dict_2.keys():
        path = old_path + f"[{k}]"
        if not k in dict_1:
            key_err += f"Key {dict_2_name}{path} not in {dict_1_name}\n"
    return key_err + err

def inicializa_seed(num_semente:int=123):
  """
  É recomendado reiniciar as seeds antes de inicializar o modelo, pois assim
  garantimos que os pesos vao ser sempre os mesmos.
  fontes de apoio:
      http://nlp.seas.harvard.edu/2018/04/03/attention.html
      https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py#L15
  """
  random.seed(num_semente)
  numpy.random.seed(num_semente)
  torch.manual_seed(num_semente)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  #torch.cuda.manual_seed(num_semente)
  #Cuda algorithms
  #torch.backends.cudnn.deterministic = True


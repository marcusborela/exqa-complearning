import copy
import time
import pandas as pd
import numpy as np
from typing import Dict
from source.calculation import util_modelo

class CalculatedMetric(object):
    """
        See description in descr_tab_calculated_metric

    """

    dtype_evaluation = {
        'cod_evaluation':int,
        # context
        'cod_metric':str,
        'value':float,
    }

    """
    example = {
        # context
        'cod_metric':'EM', #:str="Metric code"
        'value':25.99, #:float::format 99.99="Value of metric in evaluation"
    }
    """

    def __init__(self, parm_calculated_metric:Dict):

        dtype_evaluation_without_code = copy.copy(CalculatedMetric.dtype_evaluation)
        del dtype_evaluation_without_code['cod_evaluation']
        msg_dif = util_modelo.compare_dicionarios_chaves(dtype_evaluation_without_code,
             parm_calculated_metric, 'Esperado', 'Encontrado')
        assert msg_dif == "", f"Estrutura esperada de parm_calculated_metric não corresponde ao esperado {msg_dif}"


        for property_name, property_type in dtype_evaluation_without_code.items():
            assert isinstance(parm_calculated_metric[property_name], property_type), f"evaluation_qa.{property_name} must be {property_type}"
        # assert isinstance(parm_calculated_metric['value'], float), f"parm_calculated_metric.value must be float"

        assert parm_calculated_metric['cod_metric'] in ('EM', 'F1', 'EM@3', 'F1@3'), "parm_calculated_metric.cod_metric must be in ('EM', 'F1', 'EM@3', 'F1@3')"
        assert parm_calculated_metric['value'] >= 1, "parm_calculated_metric.value must be >= 1 (83.38%) "


        self._data = parm_calculated_metric
        self._data['value'] = float(self._data['value'],2) # 2 decimal places

    @property
    def data(self):
        return self._data

    @property
    def metric(self):
        return self._data['cod_metric']

    @property
    def value(self):
        return self._data['value']

class EvaluationQa(object):
    """
        See description in descr_tab_evaluation
    """
    dtype_evaluation = {
        'cod':int,
        # context
        'name_learning_method':str,
        'ind_language':str,
        'name_model':str,
        'name_device':str,
        'descr_filter':str,
        # números gerais
        'num_question':int,
        'num_top_k':int,
        'num_factor_multiply_top_k':int,
        'datetime_execution': str,
        'time_execution_total':float,
        'time_execution_per_question':float,
        # parâmetros (pode crescer)
        # Transfer Learning
        'num_doc_stride':int,
        'num_max_answer_length':int,
        'if_handle_impossible_answer':bool,
        # Context learning
        'ind_format_prompt':str,
        'descr_task_prompt':str,
        'num_shot':int
    }
    """
    example = {
        # context
        'name_learning_method':'transfer', #:str="Name of the learning method. Must be in ('context','transfer')"
        'ind_language':'en', #:str="Indicate the language. Must be one of: ('en', 'pt')"
        'name_model':'modelxxxx', #:str="Name of the machine learning model"
        'name_device':'gpu:0', #:str="Name of device used. Rule: in ('cpu', 'gpu:0')"
        'descr_filter':'', #:str="Description of the filter, if applied,  in the dataset for qa selection. Ex.: 'len(contexto) + len(pergunta) < 1024'"
        # números gerais
        'num_question':999999, #:int="Number of questions evaluated"
        'datetime_execution':'yyyy-yy-dd hh24:mi:ss', #:datetime::format yyyy-yy-dd hh24:mi:ss="Date and time of execution start"
        'time_execution_total':2164, #:float::format 99999="Total execution time of the evaluation (in seconds)"
        'time_execution_per_question':222, #:float::format 9999="Average evaluation execution time per question (in miliseconds)"
        # parâmetros (pode crescer)
        # Transfer Learning
        'num_doc_stride':9999, #:int="Indicates the number of chars to be repeated in the division of large contexts, if necessary"
        'num_max_answer_length':99999, #:int="Maximum number of characters accepted in the response"
        'if_handle_impossible_answer':False, #:bool::format 9="Indicates whether the model was considering the possibility of not having an answer to the question in the context"
        # Context learning
        'ind_format_prompt':'1_1', #:str="Indicates the format of the prompt in context learning. Must be in: (descr: "description; zero shot", descr_1_n: "description; shot: 1 context:n qas",descr_1_1:"description; shot: 1 context:1 qas")
        'descr_task_prompt':'Answer question below', #:str="Description of the task in the prompt"
        'num_shot':0, #:int="Number of examples used in the prompt"
    }
    """

    def __init__(self, parm_evaluation_qa_data:Dict):
        dtype_evaluation_without_code = copy.copy(EvaluationQa.dtype_evaluation)
        del dtype_evaluation_without_code['cod']
        msg_dif = util_modelo.compare_dicionarios_chaves(dtype_evaluation_without_code,
             parm_evaluation_qa_data, 'Esperado', 'Encontrado')
        assert msg_dif == "", f"Estrutura esperada de parm_evaluation_qa_data não corresponde ao esperado {msg_dif}"

        for property_name, property_type in dtype_evaluation_without_code.items():
            assert isinstance(parm_evaluation_qa_data[property_name], property_type), f"evaluation_qa.{property_name} must be {property_type}"

        assert parm_evaluation_qa_data['name_learning_method'] in ('transfer', 'context'), f"evaluation_qa.name_learning_method must be in ('transfer', 'context')"
        assert parm_evaluation_qa_data['ind_language'] in ('en','pt'), f"evaluation_qa.ind_language must be in ('en','pt')"
        assert parm_evaluation_qa_data['name_device'] in ('cpu', 'gpu:0'), f"evaluation_qa.name_device must be in ('cpu', 'gpu:0')"
        assert parm_evaluation_qa_data['ind_format_prompt'] in ('descr','descr_1_n', 'descr_1_1'), f"evaluation_qa.ind_format_prompt must be in ('descr','descr_1_n', 'descr_1_1')"

        """
        assert isinstance(parm_evaluation_qa_data['datetime_execution'], str), f"evaluation_qa.datetime_execution must be str"
        assert isinstance(parm_evaluation_qa_data['time_execution_total'], int), f"evaluation_qa.time_execution_total must be int"
        assert isinstance(parm_evaluation_qa_data['time_execution_per_question'], int), f"evaluation_qa.time_execution_per_question must be int"
        assert isinstance(parm_evaluation_qa_data['num_doc_stride'], int), f"evaluation_qa.num_doc_stride must be int"
        assert isinstance(parm_evaluation_qa_data['num_max_answer_length'], int), f"evaluation_qa.num_max_answer_length must be int"
        assert isinstance(parm_evaluation_qa_data['if_handle_impossible_answer'], bool), f"evaluation_qa.if_handle_impossible_answer must be bool"
        assert isinstance(parm_evaluation_qa_data['name_model'], str), f"evaluation_qa.name_model must be str"
        assert isinstance(parm_evaluation_qa_data['descr_filter'], str), f"evaluation_qa.descr_filter must be str"
        assert isinstance(parm_evaluation_qa_data['num_question'], int), f"evaluation_qa.num_question must be int"
        assert isinstance(parm_evaluation_qa_data['descr_task_prompt'], str), f"evaluation_qa.descr_task_prompt must be str"
        assert isinstance(parm_evaluation_qa_data['num_shot'], int), f"evaluation_qa.num_shot must be int"
        """

        self._data = parm_evaluation_qa_data
        self._data['calculated_metric'] = []

    def add_metric(self, parm_calculated_metric:CalculatedMetric):
        self._data['calculated_metric'].append(parm_calculated_metric)

    @property
    def data(self):
        return self._data

class RastroEvaluationQa(object):
    """
    evaluateQa
    init:
        carrega arquivo

    add:
        adiciona ao final do arquivo

    commit:
        persiste no arquivo
    """

    def __init__(self):

        # Reads data from tab_calculated_metric.csv in dataframe
        df_calculated_metric = pd.read_csv('data/tab_calculated_metric.csv', sep = ',',
            header=0,
            dtype= CalculatedMetric.dtype_evaluation,
            index_col=False)

        if df_calculated_metric.shape[0]>0:
            for property_name, property_type in CalculatedMetric.dtype_evaluation.items():
                if property_type in (int, float, bool):
                    df_calculated_metric[property_name] = df_calculated_metric[property_name].astype(property_type)
        self._df_calculated_metric = df_calculated_metric




        print("EvaluationQa.dtype_evaluation: \n {EvaluationQa.dtype_evaluation.items}")
        # Reads data from tab_evaluation.csv in dataframe
        df_evaluation = pd.read_csv('data/tab_evaluation.csv', sep = ',',
            header=0, index_col=False,
            dtype= EvaluationQa.dtype_evaluation)

        if df_evaluation.shape[0]>0:
            for property_name, property_type in EvaluationQa.dtype_evaluation.items():
                if property_type in (int, float, bool):
                    df_evaluation[property_name] = df_evaluation[property_name].astype(property_type)
        self._df_evaluation = df_evaluation
        self._last_code = df_evaluation.shape[0]



    def add(self, parm_evaluation:EvaluationQa):
        self._last_code += 1
        eval_base = copy.copy(parm_evaluation.data)
        eval_metric = copy.copy(eval_base['calculated_metric'])
        del eval_base['calculated_metric']
        #append row to the dataframe
        eval_base['cod'] = self._last_code
        self._df_evaluation = self._df_evaluation.append(eval_base, ignore_index=True)
        for calculated_metric in eval_metric:
            new_calculated_metric = calculated_metric.data
            new_calculated_metric['cod_evaluation'] = self._last_code
            self._df_calculated_metric = self._df_calculated_metric.append(new_calculated_metric, ignore_index=True)

    def save(self):
        self._df_evaluation.to_csv('data/tab_evaluation.csv', sep = ',', index=False)
        self._df_calculated_metric.to_csv('data/tab_calculated_metric.csv', sep = ',', index=False)

    @property
    def count(self):
        return self._last_code

    def imprime(self):
        print(f"count: \n{self.count}")
        print(f"df_evaluation: \n{self.df_evaluation}")
        print(f"df_calculated_metric: \n{self._df_calculated_metric}")


    @property
    def df_calculated_metric(self):
        return self._df_calculated_metric

    @property
    def df_evaluation(self):
        return self._df_evaluation


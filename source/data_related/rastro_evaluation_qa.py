import copy
import pandas as pd
import numpy as np
from typing import  List
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

    def __init__(self, parm_calculated_metric:Dict):

        dtype_evaluation_without_code = copy.copy(CalculatedMetric.dtype_evaluation)
        del dtype_evaluation_without_code['cod_evaluation']
        msg_dif = util_modelo.compare_dicionarios_chaves(dtype_evaluation_without_code,
             parm_calculated_metric, 'Esperado', 'Encontrado')
        assert msg_dif == "", f"Estrutura esperada de parm_calculated_metric não corresponde ao esperado {msg_dif}"


        for property_name, property_type in dtype_evaluation_without_code.items():
            assert isinstance(parm_calculated_metric[property_name], property_type), f"evaluation_qa.{property_name} must be {property_type}"
        # assert isinstance(parm_calculated_metric['value'], float), f"parm_calculated_metric.value must be float"

        assert parm_calculated_metric['cod_metric'] in ('EM', 'F1', 'EM@3', 'F1@3'), "parm_calculated_metric.cod_metric must be in ('EM', 'F1', 'EM@3', 'F1@3'). Found: {parm_calculated_metric['cod_metric']}"
        # assert parm_calculated_metric['value'] >= 1, "parm_calculated_metric.value must be >= 1 (83.38%) . Found: {parm_calculated_metric['value']}"


        self._data = parm_calculated_metric
        self._data['value'] = round(self._data['value'],2) # 2 decimal places

    @property
    def data(self):
        return self._data

    @property
    def metric(self):
        return self._data['cod_metric']

    @property
    def value(self):
        return self._data['value']

    @property
    def info(self):
        return f"{self._data['cod_metric']}:{self._data['value']}"

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
        'num_batch_size':int,
        'num_top_k':int,
        'num_factor_multiply_top_k':int,
        'datetime_execution': str,
        'time_execution_total':int,
        'time_execution_per_question':int,
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

    def __init__(self, parm_evaluation_qa_data:Dict):
        dtype_evaluation_without_code = copy.copy(EvaluationQa.dtype_evaluation)
        del dtype_evaluation_without_code['cod']
        msg_dif = util_modelo.compare_dicionarios_chaves(dtype_evaluation_without_code,
             parm_evaluation_qa_data, 'Esperado', 'Encontrado')
        assert msg_dif == "", f"Estrutura esperada de parm_evaluation_qa_data não corresponde ao esperado {msg_dif}"

        for property_name, property_type in dtype_evaluation_without_code.items():
            assert isinstance(parm_evaluation_qa_data[property_name], property_type), f"evaluation_qa.{property_name} must be {property_type}"

        assert parm_evaluation_qa_data['name_learning_method'] in ('transfer', 'context'), f"evaluation_qa.name_learning_method must be in ('transfer', 'context'). Found: {parm_evaluation_qa_data['name_learning_method']}"
        assert parm_evaluation_qa_data['ind_language'] in ('en','pt'), f"evaluation_qa.ind_language must be in ('en','pt'). Found: {parm_evaluation_qa_data['ind_language']}"
        assert parm_evaluation_qa_data['name_device'] in ('cpu', 'cuda:0'), f"evaluation_qa.name_device must be in ('cpu', 'cuda:0'). Found: {parm_evaluation_qa_data['name_device']}"
        assert parm_evaluation_qa_data['ind_format_prompt'] in ('', 'descr','descr_1_n', 'descr_1_1'), f"evaluation_qa.ind_format_prompt must be in ('descr','descr_1_n', 'descr_1_1'). Found: {parm_evaluation_qa_data['ind_format_prompt']}"

        self._data = parm_evaluation_qa_data
        self._data['calculated_metric'] = []

    def add_metric(self, parm_calculated_metric:CalculatedMetric):
        self._data['calculated_metric'].append(parm_calculated_metric)

    @property
    def info(self):
        values = {x: self._data[x] for x in self._data if x != 'calculated_metric'}
        descr = f"Evaluation properties: {str(values)}"
        descr += f"\nEvaluation metrics: {[x.info for x in self._data['calculated_metric']]})"
        return descr

    def imprime(self):
        print(self.info)

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

def persist_evaluation(parm_list_evaluation:List[EvaluationQa], parm_if_print:bool=False):
    rastro_eval_qa = RastroEvaluationQa()
    for evaluation in parm_list_evaluation:
        rastro_eval_qa.add(evaluation)
    rastro_eval_qa.save()
    if parm_if_print:
        rastro_eval_qa.imprime()
    return rastro_eval_qa._last_code+1 # código gerado

dtype_calculated_metric_per_question = {
    'cod_evaluation':int,
    'cod_question':str,
    'cod_metric':str,
    'value':float,
}

def persist_metric_per_question(parm_cod_evaluation, parm_dict_metric_per_question:Dict):
    """
    Expects a dict with:
        id_question: {} {'EM':xxxx, 'F1':xxxx, 'EM@3':xxxx, 'F1@3':xxxx}
    """
    # Reads data from tab_calculated_metric_per_question.csv in dataframe
    df_calculated_metric_per_question = pd.read_csv('data/tab_calculated_metric_per_question.csv', sep = ',',
        header=0,
        dtype= dtype_calculated_metric_per_question,
        index_col=False)

    if df_calculated_metric_per_question.shape[0]>0:
        for property_name, property_type in dtype_calculated_metric_per_question.items():
            if property_type in (int, float, bool):
                df_calculated_metric_per_question[property_name] = df_calculated_metric_per_question[property_name].astype(property_type)

    eval_dict = {}
    eval_dict['cod_evaluation'] = parm_cod_evaluation
    for question_id in parm_dict_metric_per_question:
        eval_dict['cod_question'] = question_id
        for metric, metric_value in parm_dict_metric_per_question[question_id].items():
            eval_dict['cod_metric'] = metric
            eval_dict['value'] = metric_value
            #append row to the dataframe
            df_calculated_metric_per_question = df_calculated_metric_per_question.append(eval_dict, ignore_index=True)

    df_calculated_metric_per_question.to_csv('data/tab_calculated_metric_per_question.csv', sep = ',', index=False)

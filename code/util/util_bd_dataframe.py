""" 
Utility functions for dealing with data (files with name starting with tab_*) in dataframe

"""
import time
import pandas as pd
import numpy as np
import copy
from util import util_noise_functions as util_noise

const_number_of_queries = 54
const_cod_metric_dcg10='DCG@10'
const_cod_metric_ndcg10='nDCG@10'
const_cod_metric_dg_ndcg10='DG:nDCG@10'

# trec20 FULL in english
const_cod_search_context_rerank_trec20 = 1
const_cod_search_context_bm25_trec20 = 3

# trec20 with judment in english
const_cod_search_context_dpr_trec20_judment_en = 2
const_cod_search_context_bm25_trec20_judment_en = 4
const_cod_search_context_rerank_trec20_judment_en = 5

list_search_context_en_trec20_full = [const_cod_search_context_rerank_trec20,
                                       const_cod_search_context_bm25_trec20]

list_search_context_en_trec20_judment = [const_cod_search_context_dpr_trec20_judment_en,
                                               const_cod_search_context_bm25_trec20_judment_en,
                                               const_cod_search_context_rerank_trec20_judment_en]

list_search_context_en = [const_cod_search_context_rerank_trec20,
                                const_cod_search_context_bm25_trec20,
                                const_cod_search_context_dpr_trec20_judment_en,
                                const_cod_search_context_bm25_trec20_judment_en,
                                const_cod_search_context_rerank_trec20_judment_en]

# trec20 with judment in portuguese
const_cod_search_context_dpr_trec20_judment_pt = 7
const_cod_search_context_bm25_trec20_judment_pt = 6
const_cod_search_context_rerank_trec20_judment_model_multi_pt_msmarco = 8
const_cod_search_context_rerank_trec20_judment_model_en_pt_msmarco = 9
const_cod_search_context_rerank_trec20_judment_model_small_pt = 10
const_cod_search_context_rerank_trec20_judment_model_base_pt = 11
const_cod_search_context_rerank_trec20_judment_model_mono_ptt5_unicamp_base_pt_msmarco_100k = 12
const_cod_search_context_rerank_trec20_judment_model_mono_ptt5_unicamp_base_t5_vocab = 13
const_cod_search_context_dpr_trec20_judment_pt_void = 14

list_search_context_pt = [const_cod_search_context_bm25_trec20_judment_pt,
                                const_cod_search_context_dpr_trec20_judment_pt,
                                const_cod_search_context_rerank_trec20_judment_model_multi_pt_msmarco,
                                const_cod_search_context_rerank_trec20_judment_model_en_pt_msmarco,
                                const_cod_search_context_rerank_trec20_judment_model_small_pt,
                                const_cod_search_context_rerank_trec20_judment_model_base_pt,
                                const_cod_search_context_rerank_trec20_judment_model_mono_ptt5_unicamp_base_pt_msmarco_100k,
                                const_cod_search_context_rerank_trec20_judment_model_mono_ptt5_unicamp_base_t5_vocab,
                                const_cod_search_context_dpr_trec20_judment_pt_void
                                ]

list_search_context_pt_rerank = [const_cod_search_context_rerank_trec20_judment_model_multi_pt_msmarco,
                                       const_cod_search_context_rerank_trec20_judment_model_en_pt_msmarco,
                                       const_cod_search_context_rerank_trec20_judment_model_small_pt,
                                       const_cod_search_context_rerank_trec20_judment_model_base_pt,
                                       const_cod_search_context_rerank_trec20_judment_model_mono_ptt5_unicamp_base_pt_msmarco_100k,
                                       const_cod_search_context_rerank_trec20_judment_model_mono_ptt5_unicamp_base_t5_vocab
                                       ]

list_search_context_pt_dpr = [
                                        const_cod_search_context_dpr_trec20_judment_pt,
                                        const_cod_search_context_dpr_trec20_judment_pt_void
                                       ]


def imprime_resumo_df(df):
    print(f"keys: {df.keys()}")
    print(f"dtypes: {df.dtypes}")
    print(f"head: {df.head}(5)")
    print(f"shape: {df.shape}[0]")

def count_tokens(parm_text:str )-> str:
    return len(util_noise.return_tokens(parm_text))

def count_tokens_missing(parm_texto_a:str,parm_texto_b:str)->int:
    lista_palavra_a = [token.lower() for token in util_noise.return_tokens(parm_texto_a)] 
    lista_palavra_b = [token.lower() for token in util_noise.return_tokens(parm_texto_b)] 
    
    qtd = 0
    for palavra in lista_palavra_a:
        if palavra not in lista_palavra_b:
            qtd += 1
    return qtd

def read_df_original_query_and_dict_val_idg():
    """Reads data from tab_original_query.csv in dataframe 
        Returns dataframe with original queries 
            and a dict dict_val_idcg10 with
                key: cod_original_query
                value: val_idcg10
    """
    df = pd.read_csv('data/tab_original_query.csv', sep = ';', 
        header=0, dtype= {'cod':np.int64, 'language':str, 'text':str, 'val_idcg10': str})
    df['val_idcg10'] = df['val_idcg10'].astype(float)        

    dict_val_idcg10 = {}
    for index, row in df.iterrows():
        dict_val_idcg10[(row['cod'],row['language'])] = row['val_idcg10']
    # print(dict_val_idcg10)
    #imprime_resumo_df(df)
    return df, dict_val_idcg10

def save_df_original_query(df):
    """Reads data from tab_original_query.csv in dataframe 
    """
    assert df.shape[0] == const_number_of_queries, f"Error: expected {const_number_of_queries} queries to match number of original queries"
    df.to_csv('data/tab_original_query.csv', sep = ';', index=False) 


def read_df_passage_with_judment():
    """Reads data from tab_noisy_query.csv in dataframe 
    """
    dict_passage = {'cod_docto':[], 'language':[], 'text':[]}
    for i, line in enumerate(open('data/tab_passage_with_judment.csv',  encoding="utf8")):
        if i == 0:  # header
            continue 
        ndx_cod_docto = line.index(';')
        cod_docto = line[:ndx_cod_docto]
        language = line[ndx_cod_docto+1:ndx_cod_docto+3]
        text = line[ndx_cod_docto+4:]
        dict_passage['cod_docto'].append(cod_docto)
        dict_passage['language'].append(language)
        dict_passage['text'].append(text)
    df = pd.DataFrame.from_dict(dict_passage)  
    df['cod_docto'] = df['cod_docto'].astype(str)      
    return df 


def read_df_noisy_query():
    """Reads data from tab_noisy_query.csv in dataframe 
    """
    df = pd.read_csv('data/tab_noisy_query.csv', sep = ';', 
        header=0, dtype= {'cod_original_query':np.int64, 'language':str, 'cod_noise_kind':np.int64, 'text':str})
    #imprime_resumo_df(df)
    return df 

def read_df_noisy_query_with_extra_columns():
    """Reads data from tab_noisy_query.csv in dataframe 
    """
    df = pd.read_csv('data/tab_noisy_query.csv', sep = ';', 
        header=0, dtype= {'cod_original_query':np.int64, 'language':str, 'cod_noise_kind':np.int64, 'text':str})
    #imprime_resumo_df(df)
    # load qtd of tokens
    df['qtd_tokens'] = df.apply(lambda row:count_tokens(row.text), axis = 1)

    # load diff of tokens
    df_original_query, _ = read_df_original_query_and_dict_val_idg()
    
    dict_text_original_query = {}
    for index, row in df_original_query.iterrows():
        dict_text_original_query[(row['cod'],row['language'])] = row['text']
    
    df['qtd_tokens_passing'] = df.apply(lambda row:count_tokens_missing(row.text, dict_text_original_query[(row['cod_original_query'],row['language'])]), axis = 1)
    df['qtd_tokens_missing'] = df.apply(lambda row:count_tokens_missing(dict_text_original_query[(row['cod_original_query'],row['language'])], row.text), axis = 1)

    return df 


def read_df_judment_and_dict_relevance():
    """Reads data from tab_judment.csv in dataframe 
        scale = {3:'perfectly relevant', 2:'highly relevant', 1:'related', 0:'Irrelevant'}
       In the dict_relevance, cod_docto is a str, as it is saved in the text base. 
    """
    df = pd.read_csv('data/tab_judment.csv', sep = ';', 
        header=0, dtype= {'cod_query':np.int64,'cod_docto':np.int64,'val_relevance':np.int64})   
    dict_relevance = {}
    for index, row in df.iterrows():
        dict_relevance[(row['cod_query'],str(row['cod_docto']))] = row['val_relevance']
    return df, dict_relevance


def read_df_noise_kind():
    """Reads data from tab_noise_kind.csv in dataframe 
    """
    df = pd.read_csv('data/tab_noise_kind.csv', sep = ',', 
        header=0, dtype= {'cod':np.int64, 'descr':str, 'abbreviation':str})   
    # imprime_resumo_df(df)
    return df

def read_df_search_context():
    """Reads data from tab_search_context.csv in dataframe 
    """
    df = pd.read_csv('data/tab_search_context.csv', sep = ',', 
        header=0, dtype= {'cod':np.int64,'abbreviation_ranking_function':str, 'abbreviation_text_base':str, 'abbreviation_text_search_engine':str, 'abbreviation_model':str}) 
    df['abbreviation'] = np.where(df['abbreviation_ranking_function']!='BM25', df['abbreviation_text_base'] + '-' + df['abbreviation_ranking_function'] + ':' + df['abbreviation_model'], df['abbreviation_text_base'] + '-' + df['abbreviation_ranking_function'] )   

    #df[df['abbreviation_ranking_function']!='BM25']['abbreviation'] =  df[df['abbreviation_ranking_function']!='BM25']['abbreviation_text_base'] + '-' + df[df['abbreviation_ranking_function']!='BM25']['abbreviation_ranking_function'] + ':' + df[df['abbreviation_ranking_function']!='BM25']['abbreviation_model']
    #df[df['abbreviation_ranking_function']=='BM25']['abbreviation'] =  df[df['abbreviation_ranking_function']=='BM25']['abbreviation_text_base'] + '-' + df[df['abbreviation_ranking_function']=='BM25']['abbreviation_ranking_function'] 
    # imprime_resumo_df(df)
    return df

def read_df_search_context_given_trec20_judment_pt():
    """Reads data from tab_search_context.csv in dataframe 
    """
    df = pd.read_csv('data/tab_search_context.csv', sep = ',', 
        header=0, dtype= {'cod':np.int64,'abbreviation_ranking_function':str, 'abbreviation_text_base':str, 'abbreviation_text_search_engine':str, 'abbreviation_model':str}) 
    df['abbreviation'] = np.where(df['abbreviation_ranking_function']!='BM25', df['abbreviation_text_base'] + '-' + df['abbreviation_ranking_function'] + ':' + df['abbreviation_model'], df['abbreviation_text_base'] + '-' + df['abbreviation_ranking_function'] )   

    #df[df['abbreviation_ranking_function']!='BM25']['abbreviation'] =  df[df['abbreviation_ranking_function']!='BM25']['abbreviation_text_base'] + '-' + df[df['abbreviation_ranking_function']!='BM25']['abbreviation_ranking_function'] + ':' + df[df['abbreviation_ranking_function']!='BM25']['abbreviation_model']
    #df[df['abbreviation_ranking_function']=='BM25']['abbreviation'] =  df[df['abbreviation_ranking_function']=='BM25']['abbreviation_text_base'] + '-' + df[df['abbreviation_ranking_function']=='BM25']['abbreviation_ranking_function'] 
    # imprime_resumo_df(df)
    return df[df.cod.isin(const_list_search_context_trec20_judment_pt)]

def save_noisy_query(parm_dict_noisy_text:dict, parm_cod_noise_kind:int, parm_language:str):
    """Appends data passed in tab_noise_query.csv

    Args:
        parm_dict_noisy_text (dict): {'cod_original_query': [], 'text': []}
        parm_cod_noise_kind (int): cod of noise kind (can not exists in tab_noise_kind.csv)
    """
    assert parm_language in ['pt','en'], f"parm_language {parm_language} is not correct. Expected value in ['pt','en']."
    assert 'cod_original_query' in parm_dict_noisy_text, f"Error: expected 'cod_original_query' in parm_dict_noisy_text"
    assert 'text' in parm_dict_noisy_text, f"Error: expected 'cod_original_query' in parm_dict_noisy_text"
    assert len(parm_dict_noisy_text.keys()) == 2, f"Error: expected only 2 keys in parm_dict_noisy_text"
    assert len(parm_dict_noisy_text['cod_original_query' ]) == len(parm_dict_noisy_text['text' ]), f"Error: expected same number of records in the lists in parm_dict_noisy_text"
    assert len(parm_dict_noisy_text['cod_original_query' ]) == const_number_of_queries, f"Error: expected {const_number_of_queries} queries to match number of original queries"

    # assert parm_cod_noise_kind exists in tab_noise_kind.csv
    df_noise_kind = read_df_noise_kind()
    assert parm_cod_noise_kind in list(df_noise_kind['cod']), f"Error: {parm_cod_noise_kind} must exists in tab_noise_kind.csv"


    # noisy_query: ler arquivo
    df_noisy_query = read_df_noisy_query() 
        
    # assert not exists noisy query
    list_uniques_noisy_queries = []
    for par in df_noisy_query[['cod_original_query', 'cod_noise_kind', 'language']].value_counts().index.values:
        list_uniques_noisy_queries.append([par[0], par[1], par[2]])
    
    for cod_original_query in parm_dict_noisy_text['cod_original_query']:
        assert [cod_original_query, parm_cod_noise_kind, parm_language] not in list_uniques_noisy_queries, f"Error: (cod_original_query, cod_noise_kind, language) ({cod_original_query}, {cod_noise_kind}, {parm_language}) must not exists in tab_noisy_query.csv"
    

    # noisy_query: adiciona registros no arquivo
    parm_dict_noisy_text['cod_noise_kind'] = [parm_cod_noise_kind] * const_number_of_queries     
    parm_dict_noisy_text['language'] = [parm_language] * const_number_of_queries     
    temp_df = pd.DataFrame.from_dict(parm_dict_noisy_text)
    
    #print(temp_df.head(3))
    df_noisy_query = df_noisy_query.append(temp_df, ignore_index = True, sort=True)
    
    # noisy_query: salva arquivo
    df_noisy_query[['cod_original_query', 'language', 'cod_noise_kind', 'text']].to_csv('data/tab_noisy_query.csv', sep = ';', index=False)   


# calculated_metric
def read_df_calculated_metric():
    """Reads data from tab_calculated_metric.csv in dataframe 
    """
    df = pd.read_csv('data/tab_calculated_metric.csv', sep = ',', 
        header=0, 
        dtype= {'date_time_execution':str,'cod_metric':str,'cod_original_query':np.int64,'cod_noise_kind':np.int64,'cod_search_context':np.int64,'value':str, 'qtd_judment_assumed_zero_relevance':np.int64, 'language':str})        
    if df.shape[0]>0:
        df['value'] = df['value'].astype(float)        
    # imprime_resumo_df(df)
    return df 

def read_df_calculated_metric_with_label(parm_list_search_context:list=None):
    """Reads data from tab_calculated_metric.csv in dataframe 
    """
    df = pd.read_csv('data/tab_calculated_metric.csv', sep = ',', 
        header=0, 
        dtype= {'date_time_execution':str,'cod_metric':str,'cod_original_query':np.int64,'cod_noise_kind':np.int64,'cod_search_context':np.int64,'value':str, 'qtd_judment_assumed_zero_relevance':np.int64, 'language':str})
    if df.shape[0]>0:
        df['value'] = df['value'].astype(float)        

    if parm_list_search_context:
        df = df.query('cod_search_context in ' + str(parm_list_search_context))

    df_noise_kind = read_df_noise_kind()
    df = pd.merge(df, df_noise_kind, left_on='cod_noise_kind', right_on='cod',suffixes=(None,'_noise_kind'))
    df = df.drop(["cod"], axis = 1)
    df = df.rename(columns={"abbreviation": "abbrev_noise_kind"}, errors="raise")
    df_search_context = read_df_search_context()
    df = pd.merge(df, df_search_context, left_on='cod_search_context', right_on='cod',suffixes=(None,'_search_context'))
    df = df.rename(columns={"descr": "noise_kind", "qtd_judment_assumed_zero_relevance": "qtd_judment_assumed", "abbreviation":"search_context", "abbreviation_text_search_engine":"search_engine"}, errors="raise")
    df = df.drop(["cod", 'abbreviation_ranking_function', 'abbreviation_text_base', 'abbreviation_model'], axis = 1)

    # carregar noisy queries to calculate variables
    df_noisy_query = read_df_noisy_query_with_extra_columns()

    df = pd.merge(df, df_noisy_query, left_on=['cod_original_query', 'language', 'cod_noise_kind'], right_on=['cod_original_query', 'language', 'cod_noise_kind'],suffixes=(None,'_noise_kind'))

    return df

def save_calculated_metric(dict_val:dict, cod_search_context:int, cod_noise_kind:int, parm_language:str):
    print(f"saving calculated metric cod_search_context:{cod_search_context}, cod_noise_kind:{cod_noise_kind}")
    assert parm_language in ['pt','en'], f"parm_language {parm_language} is not correct. Expected value in ['pt','en']."
    assert 'cod_original_query' in dict_val.keys(), f"Parameter dict_val.keys without cod_original_query"
    assert 'dcg10' in dict_val.keys(), f"Parameter dict_val.keys without dcg10"
    assert 'ndcg10' in dict_val.keys(), f"Parameter dict_val.keys without ndcg10"
    assert 'qtd_judment_assumed_zero_relevance' in dict_val.keys(), f"Parameter dict_val.keys without qtd_judment_assumed_zero_relevance"
    assert len(dict_val.keys())==4, f"Parameter dict_val.keys with unknown key {dict_val.keys()}"
    assert len(dict_val['qtd_judment_assumed_zero_relevance' ]) == const_number_of_queries, f"Error: expected {const_number_of_queries} records to match number of original queries. Found {len(dict_val['qtd_judment_assumed_zero_relevance' ])} in qtd_judment_assumed_zero_relevance"
    assert len(dict_val['dcg10' ]) == const_number_of_queries, f"Error: expected {const_number_of_queries} records to match number of original queries. Found {len(dict_val['dcg10' ])} in dcg10"
    assert len(dict_val['ndcg10' ]) == const_number_of_queries, f"Error: expected {const_number_of_queries} records to match number of original queries. Found {len(dict_val['ndcg10' ])} in ndcg10"
    assert len(dict_val['cod_original_query' ]) == const_number_of_queries, f"Error: expected {const_number_of_queries} records to match number of original queries. Found {len(dict_val['cod_original_query' ])} in cod_original_query"


    # print(df_noise_kind.keys())
    df_noise_kind = read_df_noise_kind()
    assert cod_noise_kind in list(df_noise_kind['cod']), f"Error: {parm_cod_noise_kind} must exists in tab_noise_kind.csv"


    # assert cod_search_context not exists in tab_noise_kind.csv
    df_search_context = read_df_search_context()
    assert cod_search_context in list(df_search_context['cod']), f"Error: {cod_search_context} must exists in tab_search_context.csv"


    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    df_calculated_metric =  read_df_calculated_metric()
    save_dict_val = {}
    save_dict_val['cod_original_query'] = dict_val['cod_original_query']
    save_dict_val['qtd_judment_assumed_zero_relevance'] = dict_val['qtd_judment_assumed_zero_relevance']


    save_dict_val['date_time_execution'] = [current_time] * const_number_of_queries    
    save_dict_val['language'] = [parm_language] * const_number_of_queries    
    save_dict_val['cod_search_context'] = [cod_search_context] * const_number_of_queries    
    save_dict_val['cod_noise_kind'] = [cod_noise_kind] * const_number_of_queries    


    save_dict_val['value'] = dict_val['dcg10']
    save_dict_val['cod_metric'] = [const_cod_metric_dcg10] * const_number_of_queries
    temp_df = pd.DataFrame.from_dict(save_dict_val)    
    # print('dcg10', temp_df.head(5))
    df_calculated_metric = df_calculated_metric.append(temp_df, ignore_index = True, sort=True)


    save_dict_val['value'] = dict_val['ndcg10']
    save_dict_val['cod_metric'] = [const_cod_metric_ndcg10] * const_number_of_queries
    temp_df = pd.DataFrame.from_dict(save_dict_val)    

    # assert not exists calculated metric
    for index, row in temp_df.iterrows():           
        condition = f"cod_original_query == {row['cod_original_query']} &  cod_noise_kind == {row['cod_noise_kind']} & cod_search_context == {row['cod_search_context']} & cod_metric == '{const_cod_metric_dg_ndcg10}' & language == '{row['language']}'"
        assert df_calculated_metric.query(condition).shape[0] == 0, f"Can not save again calculus already done for condition {condition}"
    
    # print('dcg10', temp_df.head(5))
    df_calculated_metric = df_calculated_metric.append(temp_df, ignore_index = True, sort=True)


    # noisy_query: salva arquivo
    df_calculated_metric[['date_time_execution','cod_metric','cod_original_query','cod_noise_kind','cod_search_context','value','qtd_judment_assumed_zero_relevance','language']].to_csv('data/tab_calculated_metric.csv', sep = ',', index=False)   

def save_dg_metric(df_dg):
    # print(f"saving calculated metric")
    # 'cod_original_query','cod_noise_kind','cod_search_context','value','qtd_judment_assumed_zero_relevance' 
    assert 'language' in df_dg.keys(), f"Parameter df_dg.keys without language"
    assert 'cod_original_query' in df_dg.keys(), f"Parameter df_dg.keys without cod_original_query"
    assert 'cod_noise_kind' in df_dg.keys(), f"Parameter df_dg.keys without cod_noise_kind"
    assert 'cod_search_context' in df_dg.keys(), f"Parameter df_dg.keys without cod_search_context"
    assert 'value' in df_dg.keys(), f"Parameter df_dg.keys without value"
    assert 'qtd_judment_assumed_zero_relevance' in df_dg.keys(), f"Parameter df_dg.keys without qtd_judment_assumed_zero_relevance"
    assert len(df_dg.keys())==6, f"Parameter df_dg.keys with unknown key {df_dg.keys()}"  
    assert (df_dg.shape[0] % const_number_of_queries) == 0, f"Error: expected multiple of {const_number_of_queries} records to match number of original queries calculated. Found {df_dg.shape[0]}."

    # assert cod_search_context exists in tab_search_context.csv
    df_search_context = read_df_search_context()
    qtd_search_context = 0
    for cod_search_context in df_dg['cod_search_context'].unique():
        qtd_search_context += 1
        assert cod_search_context in list(df_search_context['cod']), f"Error: {cod_search_context} must exists in tab_search_context.csv"

    # assert cod_noise_kind exists in tab_noise_kind.csv
    df_noise_kind = read_df_noise_kind()
    qtd_noise_kind = 0
    for cod_noise_kind in df_dg['cod_noise_kind'].unique():
        qtd_noise_kind += 1
        assert cod_noise_kind in list(df_noise_kind['cod']), f"Error: {cod_noise_kind} must exists in tab_noise_kind.csv"

    qtd_language = 0
    for language in df_dg['language'].unique():
        qtd_language += 1

    # assert cod_original_query exists in tab_original_query.csv
    df_original_query, _ = read_df_original_query_and_dict_val_idg()
    qtd_original_query = 0
    list_uniques_original_queries_with_language = []
    for par in df_original_query[['cod', 'language']].value_counts().index.values:
        list_uniques_original_queries_with_language.append([par[0], par[1]])
    
    for cod_original_query, language in df_dg[['cod_original_query', 'language']].value_counts().index.values:
        qtd_original_query += 1
        assert [cod_original_query, language] in list_uniques_original_queries_with_language, f"Error: (cod_original_query, language) ({cod_original_query}, {language}) must exists in tab_original_query.csv"

    # assert df_dg.shape[0] == (qtd_original_query * qtd_noise_kind * qtd_search_context * qtd_language), f"Error: expected {qtd_original_query * qtd_noise_kind * qtd_search_context} records to match number product: qtd_original_query * qtd_noise_kind * qtd_search_context. Found {df_dg.shape[0]}."

    df_dg['date_time_execution'] = time.strftime('%Y-%m-%d %H:%M:%S')
    df_dg['cod_metric'] = const_cod_metric_dg_ndcg10

    df_calculated_metric =  read_df_calculated_metric()
    # assert not exists calculated metric
    for index, row in df_dg.iterrows():           
        condition = f"cod_original_query == {row['cod_original_query']} &  cod_noise_kind == {row['cod_noise_kind']} & cod_search_context == {row['cod_search_context']} & cod_metric == 'DG:nDCG@10' & language == '{row['language']}'"
        assert df_calculated_metric.query(condition).shape[0] == 0, f"Can not save again calculus already done for condition {condition}"
    
    df_calculated_metric = df_calculated_metric.append(df_dg, ignore_index = True, sort=True)

    # noisy_query: salva arquivo
    df_calculated_metric[['date_time_execution','cod_metric','cod_original_query','cod_noise_kind','cod_search_context','value','qtd_judment_assumed_zero_relevance', 'language']].to_csv('data/tab_calculated_metric.csv', sep = ',', index=False)   



"""
Old version:

def load_dict_judment_and_scale()->dict:
    
        Gets data from data/originals/2020qrels-pass.txt 
            (got from https://trec.nist.gov/data/deep/2020qrels-pass.txt)

        Returns 2 dicts:
            dict_judment:
                key: tuple formed by (query_nr, passage_id)
                value: relevance registered for the tuple
            dict_scale_relevance:
                key: relevance (number between 0 and 3)
                value: label associated
    Obs.:
        **Passages** were judged on a four-point scale of:
        * Not Relevant (0), 
        * Related (1), 
        * Highly Relevant (2), and 
        * Perfect (3), 

        where 'Related' is actually NOT Relevant---it means that the passage was on the same general topic, but did not answer the question. 

        Thus, for Passage Ranking task runs (only), to compute evaluation measures that use binary relevance judgments using 
        trec_eval, you either need to use 
        trec_eval's -l option [trec_eval -l 2 qrelsfile runfile]
        or modify the qrels file to change all 1 judgments to 0.
        

    scale = {3:'perfectly relevant', 2:'highly relevant', 1:'related', 0:'Irrelevant'}
    dict_judment = {}
    for i, line in enumerate(open('data/originals/2020qrels-pass.txt')):
        query_nr, _, pid, eval = line.rstrip().split()
        query_nr = int(query_nr)
        dict_judment[(query_nr, pid)] = int(eval)

    print(f"\nLoaded len: {len(dict_judment)} judments; dict_judment[0] {list(dict_judment)[0], dict_judment[list(dict_judment)[0]], scale[dict_judment[list(dict_judment)[0]]]}")
    assert len(dict_judment)==11386, f" len(dict_judment) {len(dict_judment)} was expected to be 11386 as it is in https://trec.nist.gov/data/deep/2020qrels-pass.txt"
    return dict_judment, scale

"""
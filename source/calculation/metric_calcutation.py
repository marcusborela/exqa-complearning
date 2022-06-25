""" 
Utility functions for calculatint Informational Retrieval Metrics
"""
import math
import time
import sys
from tqdm import tqdm
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from util import util_bd_dataframe as util_bd_pandas


def calculate_metric_in_es(parm_type_retrieval:str, parm_language:str):
    """ Calcuate metrics in base of elastic search 
        Considers search mechanism specified in parm_type_retrieval

    Args:
        parm_type_retrieval (str): dpr, bm25, rerank


    """
    assert parm_type_retrieval in ("dpr","bm25","rerank"), f"parm_type_retrieval: {parm_type_retrieval} expected to be in ('dpr','bm25')"
    assert parm_language in ['en','pt'], f"parm_language: {parm_language} is not valid: in ['en','pt']"

    _, dict_val_idcg10 = util_bd_pandas.read_df_original_query_and_dict_val_idg()
    df_noisy_query = util_bd_pandas.read_df_noisy_query()
    _, dict_judment = util_bd_pandas.read_df_judment_and_dict_relevance()

    doc_store = util_es.return_doc_store(parm_language)
    if parm_type_retrieval == 'bm25':
        if parm_language == 'en':
            search_context = util_bd_pandas.const_cod_search_context_bm25_trec20_judment_en        
        else:
            search_context = util_bd_pandas.const_cod_search_context_bm25_trec20_judment_pt 
        retriever = util_ret.init_retriever_es_bm25(doc_store, parm_language)

    elif parm_type_retrieval == 'dpr':
        retriever = util_ret.init_retriever_es_dpr(doc_store, parm_language)
        if parm_language == 'en':
            search_context = util_bd_pandas.const_cod_search_context_dpr_trec20_judment_en     
        else:
            # search_context = util_bd_pandas.const_cod_search_context_dpr_trec20_judment_pt  
            search_context =  util_bd_pandas.const_cod_search_context_dpr_trec20_judment_pt_void       
    else:   
        retriever = util_ret.init_retriever_es_bm25(doc_store)
        if parm_language == 'en':
            reranker = MonoT5(pretrained_model_name_or_path='castorini/monot5-base-msmarco') # default: 'castorini/monot5-base-msmarco'        
            search_context = util_bd_pandas.const_cod_search_context_rerank_trec20_judment_en
        else:  
            # try out unicamp-dl/mt5-base-en-pt-msmarco  
            # results expected worse: https://arxiv.org/pdf/2108.13897

            # search_context = util_bd_pandas.const_cod_search_context_rerank_trec20_judment_model_en_pt_msmarco
            # reranker = MonoT5(pretrained_model_name_or_path='unicamp-dl/mt5-base-en-pt-msmarco')        

            #search_context = util_bd_pandas.const_cod_search_context_rerank_trec20_judment_model_multi_pt_msmarco
            #reranker = MonoT5(pretrained_model_name_or_path='unicamp-dl/mt5-base-multi-msmarco')        

            # search_context = util_bd_pandas.const_cod_search_context_rerank_trec20_judment_model_small_pt
            # reranker = MonoT5(pretrained_model_name_or_path='unicamp-dl/ptt5-small-portuguese-vocab', token_false= '▁não', token_true='▁sim')        

            """
            Deu erro:
            search_context = util_bd_pandas.const_cod_search_context_rerank_trec20_judment_model_minilm_multi_msmarco
            reranker = MonoT5(pretrained_model_name_or_path='unicamp-dl/multilingual-MiniLM-L6-v2-multi-msmarco', token_false= '▁false', token_true='▁true')        

            search_context = util_bd_pandas.const_cod_search_context_rerank_trec20_judment_model_minilm_multi_msmarco
            reranker = MonoT5(pretrained_model_name_or_path='unicamp-dl/multilingual-MiniLM-L6-v2-pt-msmarco', token_false= '▁não', token_true='▁sim')        


            Unrecognized configuration class <class 'transformers.models.xlm_roberta.configuration_xlm_roberta.XLMRobertaConfig'> for this kind of AutoModel: AutoModelForSeq2SeqLM.
            Model type should be one of BigBirdPegasusConfig, M2M100Config, LEDConfig, BlenderbotSmallConfig, MT5Config, T5Config, PegasusConfig, MarianConfig, MBartConfig, BlenderbotConfig, BartConfig, FSMTConfig, EncoderDecoderConfig, XLMProphetNetConfig, ProphetNetConfig.
            """

            # search_context = util_bd_pandas.const_cod_search_context_rerank_trec20_judment_model_base_pt
            # reranker = MonoT5(pretrained_model_name_or_path='unicamp-dl/ptt5-base-portuguese-vocab', token_false= '▁não', token_true='▁sim')        

            # search_context = util_bd_pandas.const_cod_search_context_rerank_trec20_judment_model_base_pt_dif_token
            # reranker = MonoT5(pretrained_model_name_or_path='unicamp-dl/ptt5-base-portuguese-vocab', token_false= '▁false', token_true='▁true')        

            # falta rodar:
            # search_context = util_bd_pandas.const_cod_search_context_rerank_trec20_judment_model_mono_ptt5_unicamp_base_pt_msmarco_100k
            # reranker = MonoT5(pretrained_model_name_or_path='unicamp-dl/ptt5-base-pt-msmarco-100k', token_false= '▁não', token_true='▁sim')        

            search_context = util_bd_pandas.const_cod_search_context_rerank_trec20_judment_model_mono_ptt5_unicamp_base_t5_vocab
            reranker = MonoT5(pretrained_model_name_or_path='unicamp-dl/ptt5-base-t5-vocab', token_false= '▁false', token_true='▁true')        

    # Calculate dcg10 and ndcg10 of noisy queries
    for noise_kind in tqdm(df_noisy_query[df_noisy_query['language']==parm_language]['cod_noise_kind'].unique(), "Progress bar - Noise kind"):
        calculated_metric = {'cod_original_query':[],'dcg10':[],'ndcg10':[], 'qtd_judment_assumed_zero_relevance':[]}
        for index, row in tqdm(df_noisy_query[(df_noisy_query['cod_noise_kind']==noise_kind) & (df_noisy_query['language']==parm_language)].iterrows(), "Progress bar - query"):       
            cod_original_query = row["cod_original_query"]
            idcg10 = dict_val_idcg10[(cod_original_query, parm_language)]
            query_text = row["text"]

            # retrieve documents
            if parm_type_retrieval in ('bm25','dpr'):
                list_docs = retriever.retrieve(query_text, top_k=10)
            else: # 'rerank'
                doctos_returned_bm25 = retriever.retrieve(query_text, top_k=100)
                list_doctos = []
                for docto in doctos_returned_bm25:
                    dt = class_docto_encontrado(id=docto.id, text=docto.content)
                    list_doctos.append(dt)
                query_busca = Query(query_text)
                list_docs = reranker.rerank(query_busca, list_doctos)[:10]

            # calculating dcg10 for query
            dcg10 = 0
            qtd_judment_assumed_zero_relevance = 0
            for i, docto_encontrado in enumerate(list_docs):
                if i >= 10:
                    raise 'Error: more than 10 retrieved!'
                docid = docto_encontrado.id
                if (cod_original_query, docid) not in dict_judment:
                    # print(f'Query and passage_id not in judment:{(cod_original_query, docid)}. Assumed zero relevance. ')
                    qtd_judment_assumed_zero_relevance += 1
                eval = dict_judment.get((cod_original_query, docid), 0)
                dcg10 += (2**int(eval)-1) / math.log2(i+2)

            # calculate ndcg10 for query
            ndcg10 = dcg10 / idcg10

            # add values to lists
            calculated_metric['cod_original_query'].append(cod_original_query)
            calculated_metric['dcg10'].append(dcg10)
            calculated_metric['ndcg10'].append(ndcg10)
            calculated_metric['qtd_judment_assumed_zero_relevance'].append(qtd_judment_assumed_zero_relevance)
        util_bd_pandas.save_calculated_metric(calculated_metric, cod_search_context=search_context, cod_noise_kind=noise_kind,parm_language=parm_language)    
        print(f" Done for noise_kind: {noise_kind} em {time.strftime('%Y-%m-%d %H:%M:%S')}")        


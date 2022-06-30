"""
Ordena busca

"""

from copy import deepcopy
import os, sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'.\.'))

json_extracao_resposta =  {
   "texto_pergunta": "Qual o melhor time do Brasil?",
   "top_k":3,
   "tamanho_max_resposta" : 40,
   "texto_contexto": "Por causa da chuva, o Flamengo, melhor \
time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo \
Flamengo, o melhor time, na minha Família. Além da bicicleta, em 1990, eu tinha a honra \
de ser flamenguista.  Ontem, quando reli os documentos do tribunal, \n\
 descobri que em 1990 quando tomei posse, minha declaração de bens só continha uma bicicleta."
    }

def responder(parm_reader, parm_json:dict):
    """
    Calcula

    Retorna dicionário com 4 chaves:
    sigla_modelo_reader
    lista_sugestao_resposta: [
        [{'texto_resposta': 'xxxx'},
         'score': 9999,  # acumulado
    ]
    contexto: ['xxxx']
    """
    if 'texto_pergunta' not in parm_json:
        raise Exception(f' json sem chave texto_pergunta. Apenas: {parm_json.keys()}')
    if "tamanho_max_resposta" not in parm_json:
      parm_json["tamanho_max_resposta"] = 40
    if parm_json['top_k'] == 0:
        parm_json['top_k'] = 2

    resultado = {}
    resultado = parm_reader.answer(texto_pergunta=parm_json['texto_pergunta'], \
            texto_contexto=parm_json['texto_contexto'], parm_topk=parm_json['top_k'], \
            parm_max_answer_length=parm_json['tamanho_max_resposta'])

    return resultado

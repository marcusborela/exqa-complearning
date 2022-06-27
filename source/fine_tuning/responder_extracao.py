"""
Ordena busca

"""

from copy import deepcopy
import os, sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'.\.'))
from  source.fine_tuning import modelo_reader, util_modelo

__esquema_json ={
    "definitions": {},
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://example.com/object1639600315.json",
  "title": "Root",
  "type": "object",
  "required": [
    "sigla_modelo_reader",
    "texto_pergunta",
    "top_k_reader",
    "lista_documento"
  ],
  "properties": {
    "sigla_modelo_reader": {
      "$id": "#root/sigla_modelo",
      "title": "sigla_modelo",
      "type": "string",
      "default": "",
      "examples": [
        "unicamp-dl/mt5-base-mmarco-v2"
      ],
      "pattern": "^.*$"
    },
    "texto_pergunta": {
      "$id": "#root/texto_pergunta",
      "title": "texto_pergunta",
      "type": "string",
      "default": "",
      "examples": [
        "Palavras faltando: faltando 1, faltando 2"
      ],
      "pattern": ".*"
    },
    "top_k_reader": {
      "$id": "#root/top_k_reader",
      "title": "top_k_reader",
      "type": "number",
      "multipleOf" : 1,
      "minimum": 1,
      "default": "",
      "examples": [
        3
      ],
      "pattern": "^[0-9]+$"
    },
    "tamanho_max_resposta": {
      "$id": "#root/tamanho_max_resposta",
      "title": "tamanho_max_resposta",
      "type": "number",
      "minimum": 1,
      "multipleOf" : 1,
      "default": "",
      "examples": [
        50
      ],
      "pattern": "^[0-9]+$"
    },
    "lista_documento": {
      "$id": "#root/lista_documento",
      "title": "lista_documento",
      "type": "array",
      "default": [],
      "items":{
        "$id": "#root/lista_documento/items",
        "title": "Items",
        "type": "object",
        "required": [
          "cod",
          "texto"
        ],
        "properties": {
          "cod": {
            "$id": "#root/lista_documento/items/cod",
            "title": "Cod",
            "type": "string",
            "default": "",
            "examples": [
              "uuid texto 1"
            ],
            "pattern": "^.*$"
          },
          "texto": {
            "$id": "#root/lista_documento/items/texto",
            "title": "texto",
            "type": "string",
            "default": "",
            "examples": [
              "nome - texto 1"
            ],
            "pattern": ".*"
          }
        }
      }

    }
  }
}

__esquema_json['properties']['sigla_modelo_reader']['pattern'] = modelo_reader.regex_lista_modelo_reader_valida_json

json_exemplo =  {
   "texto_pergunta": "Qual o melhor time do Brasil?",
   "top_k_reader":3,
   "tamanho_max_resposta" : 40,
   "texto_contexto": "Por causa da chuva, o Flamengo, melhor \
time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo \
Flamengo, o melhor time, na minha Família. Além da bicicleta, em 1990, eu tinha a honra \
de ser flamenguista.  Ontem, quando reli os documentos do tribunal, \n\
 descobri que em 1990 quando tomei posse, minha declaração de bens só continha uma bicicleta."
    }

json_exemplo['sigla_modelo_reader'] = modelo_reader.lista_modelos_reader[0]


def responder_extracao(parm_json:dict):
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
    if parm_json['top_k_reader'] == 0:
        parm_json['top_k_reader'] = 2

    resultado = {}
    resultado = modelo_reader.reader[parm_json['sigla_modelo_reader']].answer(texto_pergunta=parm_json['texto_pergunta'], \
            texto_contexto=parm_json['texto_contexto'], parm_topk=parm_json['top_k_reader'], \
            parm_max_answer_length=parm_json['tamanho_max_resposta'])

    return resultado

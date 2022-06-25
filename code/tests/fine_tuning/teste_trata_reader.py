"""
print('oi')
Desejado comportamento semelhante a https://countwordsfree.com/comparetexts

"""
from operator import pos
import warnings
import unittest
import os
import sys
from time import strftime

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'.\.'))

import trata_reader as alvo # pylint: disable=wrong-import-position # precisa dos sys.path antes
import util_modelo


lista_json_exemplo =  [{
   "texto_pergunta": "Qual o melhor time do Brasil?",
   "top_k_reader":3,
   "tamanho_max_resposta" : 40,
   "lista_documento": [  # na ordem de relevância com a pergunta
    {
       "cod": "uuid 2",
       "texto": "Por causa da chuva, o Flamengo, melhor time do Brasil, ficou sem jogar a final da Libertadores."},
    {
       "cod": "uuid 3",
       "texto": "Há muitos torcedores pelo Flamengo, o melhor time, na minha Família. Além da bicicleta, em 1990, eu tinha a honra de ser flamenguista. "},
    {
       "cod": "uuid 1",
       "texto": "Ontem, quando reli os documentos do tribunal, \n descobri que em 1990 quando tomei posse, minha declaração de bens só continha uma bicicleta."},
             ]
    }
]


lista_texto_concatenado_esperado = ["Por causa da chuva, o Flamengo, melhor \
time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo \
Flamengo, o melhor time, na minha Família. Além da bicicleta, em 1990, eu tinha a honra \
de ser flamenguista.  Ontem, quando reli os documentos do tribunal, \n\
 descobri que em 1990 quando tomei posse, minha declaração de bens só continha uma bicicleta."]

lista_pos_to_docto_esperado = [{0: 'uuid 2',
 1: 'uuid 2', 2: 'uuid 2', 3: 'uuid 2', 4: 'uuid 2', 5: 'uuid 2', 6: 'uuid 2', 7: 'uuid 2', 8: 'uuid 2', 9: 'uuid 2', 10: 'uuid 2',
 11: 'uuid 2', 12: 'uuid 2', 13: 'uuid 2', 14: 'uuid 2', 15: 'uuid 2', 16: 'uuid 2', 17: 'uuid 2', 18: 'uuid 2', 19: 'uuid 2', 20: 'uuid 2',
 21: 'uuid 2', 22: 'uuid 2', 23: 'uuid 2', 24: 'uuid 2', 25: 'uuid 2', 26: 'uuid 2', 27: 'uuid 2', 28: 'uuid 2', 29: 'uuid 2', 30: 'uuid 2',
 31: 'uuid 2', 32: 'uuid 2', 33: 'uuid 2', 34: 'uuid 2', 35: 'uuid 2', 36: 'uuid 2', 37: 'uuid 2', 38: 'uuid 2', 39: 'uuid 2', 40: 'uuid 2',
 41: 'uuid 2', 42: 'uuid 2', 43: 'uuid 2', 44: 'uuid 2', 45: 'uuid 2', 46: 'uuid 2', 47: 'uuid 2', 48: 'uuid 2', 49: 'uuid 2', 50: 'uuid 2',
 51: 'uuid 2', 52: 'uuid 2', 53: 'uuid 2', 54: 'uuid 2', 55: 'uuid 2', 56: 'uuid 2', 57: 'uuid 2', 58: 'uuid 2', 59: 'uuid 2', 60: 'uuid 2',
 61: 'uuid 2', 62: 'uuid 2', 63: 'uuid 2', 64: 'uuid 2', 65: 'uuid 2', 66: 'uuid 2', 67: 'uuid 2', 68: 'uuid 2', 69: 'uuid 2', 70: 'uuid 2',
 71: 'uuid 2', 72: 'uuid 2', 73: 'uuid 2', 74: 'uuid 2', 75: 'uuid 2', 76: 'uuid 2', 77: 'uuid 2', 78: 'uuid 2', 79: 'uuid 2', 80: 'uuid 2',
 81: 'uuid 2', 82: 'uuid 2', 83: 'uuid 2', 84: 'uuid 2', 85: 'uuid 2', 86: 'uuid 2', 87: 'uuid 2', 88: 'uuid 2', 89: 'uuid 2', 90: 'uuid 2',
 91: 'uuid 2', 92: 'uuid 2', 93: 'uuid 2', 94: 'uuid 2', 96: 'uuid 3', 97: 'uuid 3', 98: 'uuid 3', 99: 'uuid 3', 100: 'uuid 3',
 101: 'uuid 3', 102: 'uuid 3', 103: 'uuid 3', 104: 'uuid 3', 105: 'uuid 3', 106: 'uuid 3', 107: 'uuid 3', 108: 'uuid 3', 109: 'uuid 3', 110: 'uuid 3',
 111: 'uuid 3', 112: 'uuid 3', 113: 'uuid 3', 114: 'uuid 3', 115: 'uuid 3', 116: 'uuid 3', 117: 'uuid 3', 118: 'uuid 3', 119: 'uuid 3', 120: 'uuid 3',
 121: 'uuid 3', 122: 'uuid 3', 123: 'uuid 3', 124: 'uuid 3', 125: 'uuid 3', 126: 'uuid 3', 127: 'uuid 3', 128: 'uuid 3', 129: 'uuid 3', 130: 'uuid 3',
 131: 'uuid 3', 132: 'uuid 3', 133: 'uuid 3', 134: 'uuid 3', 135: 'uuid 3', 136: 'uuid 3', 137: 'uuid 3', 138: 'uuid 3', 139: 'uuid 3', 140: 'uuid 3',
 141: 'uuid 3', 142: 'uuid 3', 143: 'uuid 3', 144: 'uuid 3', 145: 'uuid 3', 146: 'uuid 3', 147: 'uuid 3', 148: 'uuid 3', 149: 'uuid 3', 150: 'uuid 3',
 151: 'uuid 3', 152: 'uuid 3', 153: 'uuid 3', 154: 'uuid 3', 155: 'uuid 3', 156: 'uuid 3', 157: 'uuid 3', 158: 'uuid 3', 159: 'uuid 3', 160: 'uuid 3',
 161: 'uuid 3', 162: 'uuid 3', 163: 'uuid 3', 164: 'uuid 3', 165: 'uuid 3', 166: 'uuid 3', 167: 'uuid 3', 168: 'uuid 3', 169: 'uuid 3', 170: 'uuid 3',
 171: 'uuid 3', 172: 'uuid 3', 173: 'uuid 3', 174: 'uuid 3', 175: 'uuid 3', 176: 'uuid 3', 177: 'uuid 3', 178: 'uuid 3', 179: 'uuid 3', 180: 'uuid 3',
 181: 'uuid 3', 182: 'uuid 3', 183: 'uuid 3', 184: 'uuid 3', 185: 'uuid 3', 186: 'uuid 3', 187: 'uuid 3', 188: 'uuid 3', 189: 'uuid 3', 190: 'uuid 3',
 191: 'uuid 3', 192: 'uuid 3', 193: 'uuid 3', 194: 'uuid 3', 195: 'uuid 3', 196: 'uuid 3', 197: 'uuid 3', 198: 'uuid 3', 199: 'uuid 3', 200: 'uuid 3',
 201: 'uuid 3', 202: 'uuid 3', 203: 'uuid 3', 204: 'uuid 3', 205: 'uuid 3', 206: 'uuid 3', 207: 'uuid 3', 208: 'uuid 3', 209: 'uuid 3', 210: 'uuid 3',
 211: 'uuid 3', 212: 'uuid 3', 213: 'uuid 3', 214: 'uuid 3', 215: 'uuid 3', 216: 'uuid 3', 217: 'uuid 3', 218: 'uuid 3', 219: 'uuid 3', 220: 'uuid 3',
 221: 'uuid 3', 222: 'uuid 3', 223: 'uuid 3', 224: 'uuid 3', 225: 'uuid 3', 226: 'uuid 3', 227: 'uuid 3', 228: 'uuid 3', 229: 'uuid 3', 230: 'uuid 3',
 232: 'uuid 1', 233: 'uuid 1', 234: 'uuid 1', 235: 'uuid 1', 236: 'uuid 1', 237: 'uuid 1', 238: 'uuid 1', 239: 'uuid 1', 240: 'uuid 1',
 241: 'uuid 1', 242: 'uuid 1', 243: 'uuid 1', 244: 'uuid 1', 245: 'uuid 1', 246: 'uuid 1', 247: 'uuid 1', 248: 'uuid 1', 249: 'uuid 1', 250: 'uuid 1',
 251: 'uuid 1', 252: 'uuid 1', 253: 'uuid 1', 254: 'uuid 1', 255: 'uuid 1', 256: 'uuid 1', 257: 'uuid 1', 258: 'uuid 1', 259: 'uuid 1', 260: 'uuid 1',
 261: 'uuid 1', 262: 'uuid 1', 263: 'uuid 1', 264: 'uuid 1', 265: 'uuid 1', 266: 'uuid 1', 267: 'uuid 1', 268: 'uuid 1', 269: 'uuid 1', 270: 'uuid 1',
 271: 'uuid 1', 272: 'uuid 1', 273: 'uuid 1', 274: 'uuid 1', 275: 'uuid 1', 276: 'uuid 1', 277: 'uuid 1', 278: 'uuid 1', 279: 'uuid 1', 280: 'uuid 1',
 281: 'uuid 1', 282: 'uuid 1', 283: 'uuid 1', 284: 'uuid 1', 285: 'uuid 1', 286: 'uuid 1', 287: 'uuid 1', 288: 'uuid 1', 289: 'uuid 1', 290: 'uuid 1',
 291: 'uuid 1', 292: 'uuid 1', 293: 'uuid 1', 294: 'uuid 1', 295: 'uuid 1', 296: 'uuid 1', 297: 'uuid 1', 298: 'uuid 1', 299: 'uuid 1', 300: 'uuid 1',
 301: 'uuid 1', 302: 'uuid 1', 303: 'uuid 1', 304: 'uuid 1', 305: 'uuid 1', 306: 'uuid 1', 307: 'uuid 1', 308: 'uuid 1', 309: 'uuid 1', 310: 'uuid 1',
 311: 'uuid 1', 312: 'uuid 1', 313: 'uuid 1', 314: 'uuid 1', 315: 'uuid 1', 316: 'uuid 1', 317: 'uuid 1', 318: 'uuid 1', 319: 'uuid 1', 320: 'uuid 1',
 321: 'uuid 1', 322: 'uuid 1', 323: 'uuid 1', 324: 'uuid 1', 325: 'uuid 1', 326: 'uuid 1', 327: 'uuid 1', 328: 'uuid 1', 329: 'uuid 1', 330: 'uuid 1',
 331: 'uuid 1', 332: 'uuid 1', 333: 'uuid 1', 334: 'uuid 1', 335: 'uuid 1', 336: 'uuid 1', 337: 'uuid 1', 338: 'uuid 1', 339: 'uuid 1', 340: 'uuid 1',
 341: 'uuid 1', 342: 'uuid 1', 343: 'uuid 1', 344: 'uuid 1', 345: 'uuid 1', 346: 'uuid 1', 347: 'uuid 1', 348: 'uuid 1', 349: 'uuid 1', 350: 'uuid 1',
 351: 'uuid 1', 352: 'uuid 1', 353: 'uuid 1', 354: 'uuid 1', 355: 'uuid 1', 356: 'uuid 1', 357: 'uuid 1', 358: 'uuid 1', 359: 'uuid 1', 360: 'uuid 1',
 361: 'uuid 1', 362: 'uuid 1', 363: 'uuid 1', 364: 'uuid 1', 365: 'uuid 1', 366: 'uuid 1', 367: 'uuid 1', 368: 'uuid 1', 369: 'uuid 1', 370: 'uuid 1',
 371: 'uuid 1'}]

lista_pos_to_docto_entrada_filtro = [{0: 'uuid 2',
 1: 'uuid 2', 2: 'uuid 2', 3: 'uuid 2', 4: 'uuid 2', 5: 'uuid 2', 6: 'uuid 2', 7: 'uuid 2', 8: 'uuid 2', 9: 'uuid 2', 10: 'uuid 2',
 11: 'uuid 2', 12: 'uuid 2', 13: 'uuid 2', 14: 'uuid 2', 15: 'uuid 2', 16: 'uuid 2', 17: 'uuid 2', 18: 'uuid 2', 19: 'uuid 2', 20: 'uuid 2',
 21: 'uuid 2', 22: 'uuid 2', 23: 'uuid 2', 24: 'uuid 2', 25: 'uuid 2', 26: 'uuid 2', 27: 'uuid 2', 28: 'uuid 2', 29: 'uuid 2', 30: 'uuid 2',
 31: 'uuid 2', 32: 'uuid 2', 33: 'uuid 2', 34: 'uuid 2', 35: 'uuid 2', 36: 'uuid 2', 37: 'uuid 2', 38: 'uuid 2', 39: 'uuid 2', 40: 'uuid 2',
 41: 'uuid 2', 42: 'uuid 2', 43: 'uuid 2', 44: 'uuid 2', 45: 'uuid 2', 46: 'uuid 2', 47: 'uuid 2', 48: 'uuid 2', 49: 'uuid 2', 50: 'uuid 2',
 51: 'uuid 2', 52: 'uuid 2', 53: 'uuid 2', 54: 'uuid 2', 55: 'uuid 2', 56: 'uuid 2', 57: 'uuid 2', 58: 'uuid 2', 59: 'uuid 2', 60: 'uuid 2',
 61: 'uuid 2', 62: 'uuid 2', 63: 'uuid 2', 64: 'uuid 2', 65: 'uuid 2', 66: 'uuid 2', 67: 'uuid 2', 68: 'uuid 2', 69: 'uuid 2', 70: 'uuid 2',
 71: 'uuid 2', 72: 'uuid 2', 73: 'uuid 2', 74: 'uuid 2', 75: 'uuid 2', 76: 'uuid 2', 77: 'uuid 2', 78: 'uuid 2', 79: 'uuid 2', 80: 'uuid 2',
 81: 'uuid 2', 82: 'uuid 2', 83: 'uuid 2', 84: 'uuid 2', 85: 'uuid 2', 86: 'uuid 2', 87: 'uuid 2', 88: 'uuid 2', 89: 'uuid 2', 90: 'uuid 2',
 91: 'uuid 2', 92: 'uuid 2', 93: 'uuid 2', 94: 'uuid 2', 96: 'uuid 3', 97: 'uuid 3', 98: 'uuid 3', 99: 'uuid 3', 100: 'uuid 3',
 101: 'uuid 3', 102: 'uuid 3', 103: 'uuid 3', 104: 'uuid 3', 105: 'uuid 3', 106: 'uuid 3', 107: 'uuid 3', 108: 'uuid 3', 109: 'uuid 3', 110: 'uuid 3',
 111: 'uuid 3', 112: 'uuid 3', 113: 'uuid 3', 114: 'uuid 3', 115: 'uuid 3', 116: 'uuid 3', 117: 'uuid 3', 118: 'uuid 3', 119: 'uuid 3', 120: 'uuid 3',
 121: 'uuid 3', 122: 'uuid 3', 123: 'uuid 3', 124: 'uuid 3', 125: 'uuid 3', 126: 'uuid 3', 127: 'uuid 3', 128: 'uuid 3', 129: 'uuid 3', 130: 'uuid 3',
 131: 'uuid 3', 132: 'uuid 3', 133: 'uuid 3', 134: 'uuid 3', 135: 'uuid 3', 136: 'uuid 3', 137: 'uuid 3', 138: 'uuid 3', 139: 'uuid 3', 140: 'uuid 3',
 141: 'uuid 3', 142: 'uuid 3', 143: 'uuid 3', 144: 'uuid 3', 145: 'uuid 3', 146: 'uuid 3', 147: 'uuid 3', 148: 'uuid 3', 149: 'uuid 3', 150: 'uuid 3',
 151: 'uuid 3', 152: 'uuid 3', 153: 'uuid 3', 154: 'uuid 3', 155: 'uuid 3', 156: 'uuid 3', 157: 'uuid 3', 158: 'uuid 3', 159: 'uuid 3', 160: 'uuid 3',
 161: 'uuid 3', 162: 'uuid 3', 163: 'uuid 3', 164: 'uuid 3', 165: 'uuid 3', 166: 'uuid 3', 167: 'uuid 3', 168: 'uuid 3', 169: 'uuid 3', 170: 'uuid 3',
 171: 'uuid 3', 172: 'uuid 3', 173: 'uuid 3', 174: 'uuid 3', 175: 'uuid 3', 176: 'uuid 3', 177: 'uuid 3', 178: 'uuid 3', 179: 'uuid 3', 180: 'uuid 3',
 181: 'uuid 3', 182: 'uuid 3', 183: 'uuid 3', 184: 'uuid 3', 185: 'uuid 3', 186: 'uuid 3', 187: 'uuid 3', 188: 'uuid 3', 189: 'uuid 3', 190: 'uuid 3',
 191: 'uuid 3', 192: 'uuid 3', 193: 'uuid 3', 194: 'uuid 3', 195: 'uuid 3', 196: 'uuid 3', 197: 'uuid 3', 198: 'uuid 3', 199: 'uuid 3', 200: 'uuid 3',
 201: 'uuid 3', 202: 'uuid 3', 203: 'uuid 3', 204: 'uuid 3', 205: 'uuid 3', 206: 'uuid 3', 207: 'uuid 3', 208: 'uuid 3', 209: 'uuid 3', 210: 'uuid 3',
 211: 'uuid 3', 212: 'uuid 3', 213: 'uuid 3', 214: 'uuid 3', 215: 'uuid 3', 216: 'uuid 3', 217: 'uuid 3', 218: 'uuid 3', 219: 'uuid 3', 220: 'uuid 3',
 221: 'uuid 3', 222: 'uuid 3', 223: 'uuid 3', 224: 'uuid 3', 225: 'uuid 3', 226: 'uuid 3', 227: 'uuid 3', 228: 'uuid 3', 229: 'uuid 3', 230: 'uuid 3',
 232: 'uuid 1', 233: 'uuid 1', 234: 'uuid 1', 235: 'uuid 1', 236: 'uuid 1', 237: 'uuid 1', 238: 'uuid 1', 239: 'uuid 1', 240: 'uuid 1',
 241: 'uuid 1', 242: 'uuid 1', 243: 'uuid 1', 244: 'uuid 1', 245: 'uuid 1', 246: 'uuid 1', 247: 'uuid 1', 248: 'uuid 1', 249: 'uuid 1', 250: 'uuid 1',
 251: 'uuid 1', 252: 'uuid 1', 253: 'uuid 1', 254: 'uuid 1', 255: 'uuid 1', 256: 'uuid 1', 257: 'uuid 1', 258: 'uuid 1', 259: 'uuid 1', 260: 'uuid 1',
 261: 'uuid 1', 262: 'uuid 1', 263: 'uuid 1', 264: 'uuid 1', 265: 'uuid 1', 266: 'uuid 1', 267: 'uuid 1', 268: 'uuid 1', 269: 'uuid 1', 270: 'uuid 1',
 271: 'uuid 1', 272: 'uuid 1', 273: 'uuid 1', 274: 'uuid 1', 275: 'uuid 1', 276: 'uuid 1', 277: 'uuid 1', 278: 'uuid 1', 279: 'uuid 1', 280: 'uuid 1',
 281: 'uuid 1', 282: 'uuid 1', 283: 'uuid 1', 284: 'uuid 1', 285: 'uuid 1', 286: 'uuid 1', 287: 'uuid 1', 288: 'uuid 1', 289: 'uuid 1', 290: 'uuid 1',
 291: 'uuid 1', 292: 'uuid 1', 293: 'uuid 1', 294: 'uuid 1', 295: 'uuid 1', 296: 'uuid 1', 297: 'uuid 1', 298: 'uuid 1', 299: 'uuid 1', 300: 'uuid 1',
 301: 'uuid 1', 302: 'uuid 1', 303: 'uuid 1', 304: 'uuid 1', 305: 'uuid 1', 306: 'uuid 1', 307: 'uuid 1', 308: 'uuid 1', 309: 'uuid 1', 310: 'uuid 1',
 311: 'uuid 1', 312: 'uuid 1', 313: 'uuid 1', 314: 'uuid 1', 315: 'uuid 1', 316: 'uuid 1', 317: 'uuid 1', 318: 'uuid 1', 319: 'uuid 1', 320: 'uuid 1',
 321: 'uuid 1', 322: 'uuid 1', 323: 'uuid 1', 324: 'uuid 1', 325: 'uuid 1', 326: 'uuid 1', 327: 'uuid 1', 328: 'uuid 1', 329: 'uuid 1', 330: 'uuid 1',
 331: 'uuid 1', 332: 'uuid 1', 333: 'uuid 1', 334: 'uuid 1', 335: 'uuid 1', 336: 'uuid 1', 337: 'uuid 1', 338: 'uuid 1', 339: 'uuid 1', 340: 'uuid 1',
 341: 'uuid 1', 342: 'uuid 1', 343: 'uuid 1', 344: 'uuid 1', 345: 'uuid 1', 346: 'uuid 1', 347: 'uuid 1', 348: 'uuid 1', 349: 'uuid 1', 350: 'uuid 1',
 351: 'uuid 1', 352: 'uuid 1', 353: 'uuid 1', 354: 'uuid 1', 355: 'uuid 1', 356: 'uuid 1', 357: 'uuid 1', 358: 'uuid 1', 359: 'uuid 1', 360: 'uuid 1',
 361: 'uuid 1', 362: 'uuid 1', 363: 'uuid 1', 364: 'uuid 1', 365: 'uuid 1', 366: 'uuid 1', 367: 'uuid 1', 368: 'uuid 1', 369: 'uuid 1', 370: 'uuid 1',
 371: 'uuid 1'}]

lista_docto_to_pos_inicio_esperado= [{'uuid 2': 0,
                               'uuid 3': 96,
                               'uuid 1': 232}]

lista_resposta_entrada_filtro = [
[
  { #uuid2
    "score": 0.8311,
    "start": 72,
    "end": 81,
    "answer": "uuid 2",
  },
  { # excluir pois final tá no meio de outro docto
    "score": 0.1520,
    "start": 72,
    "end": 231,
    "answer": "exclui",
  },
  { # uuid2 com final no espaço final do próprio doct
    "score": 0.0010,
    "start": 72,
    "end": 95,
    "answer": "uuid 2",
  },
  { # uuid3 com início no espaço do inicio do próprio doct
    "score": 0.0056,
    "start": 95,
    "end": 99,
    "answer": "uuid 3",
  },
  {  # excluir pois início no espaço do inicio de outro doct
    "score": 0.0003,
    "start": 95,
    "end": 235,
    "answer": "exclui",
  },
  { # uuid3 com início no espaço do inicio do próprio doct
    "score": 0.0056,
    "start": 232,
    "end": 371,
    "answer": "uuid 1",
  },
  {  # excluir pois início negativo
    "score": 0.0003,
    "start": -1,
    "end": 235,
    "answer": "exclui",
  },
  {  # uuid2 pois início fora do texto é somado de 1
    "score": 0.0003,
    "start": -1,
    "end": 70,
    "answer": "uuid 2",
  },
  {  # uuid1 pois final fora do texto é subtraido de 1
    "score": 0.0003,
    "start": 235,
    "end": 372,
    "answer": "uuid 1",
  },
  {  # excluir pois final além do texto
    "score": 0.0003,
    "start": 235,
    "end": 373,
    "answer": "exclui",
  }
]]

lista_resposta_saida_filtro = [
[
  { #uuid2
    "score": 0.8311,
    "start": 72,
    "end": 81,
    "answer": "uuid 2",
    "id_docto": "uuid 2",
  },
  { # uuid2 com final no espaço final do próprio doct
    "score": 0.0010,
    "start": 72,
    "end": 94,
    "answer": "uuid 2",
    "id_docto": "uuid 2",
  },
  { # uuid3 com início no espaço do inicio do próprio doct
    "score": 0.0056,
    "start": 96,
    "end": 99,
    "answer": "uuid 3",
    "id_docto": "uuid 3",
  },
  { # uuid3 com início no espaço do inicio do próprio doct
    "score": 0.0056,
    "start": 232,
    "end": 371,
    "answer": "uuid 1",
    "id_docto": "uuid 1",
  },
  {  # uuid2 pois início fora do texto é somado de 1
    "score": 0.0003,
    "start": 0,
    "end": 70,
    "answer": "uuid 2",
    "id_docto": "uuid 2",
  },
  {  # uuid3 pois final fora do texto é subtraido de 1
    "score": 0.0003,
    "start": 235,
    "end": 371,
    "answer": "uuid 1",
    "id_docto": "uuid 1",
  },
]]


"""
lista_resposta_saida_formatacao = [
  [{'texto_resposta': 'uuid 2', 'score': 0.8323999999999999, 'lista_referencia': [['uuid 3', 0, 3]]}]
]
"""
lista_resposta_saida_formatacao = [[
  {'texto_resposta': 'uuid 2',
  'score': 0.8323999999999999,
     'lista_referencia': [
       ['uuid 2', 72, 81],
       ['uuid 2', 72, 94],
       ['uuid 2', 0, 70]]},
  {'texto_resposta': 'uuid 1',
  'score': 0.0059,
     'lista_referencia': [
       ['uuid 1', 0, 139],
       ['uuid 1', 3, 139]]},
  {'texto_resposta': 'uuid 3',
  'score': 0.0056,
     'lista_referencia': [
       ['uuid 3', 0, 3]]}]]

def teste_formatar_retirar_duplicatas(parm_teste_objeto):
    """
        Teste de retirada de duplicatas
    """
    caso_teste = 'Testar retirada de duplicatas'
    for ind_conjunto_resposta, conjunto_resposta in enumerate(lista_resposta_saida_filtro):
        conjunto_saida = alvo.formatar_retirar_duplicatas(conjunto_resposta, lista_docto_to_pos_inicio_esperado[ind_conjunto_resposta])
        print(conjunto_saida)
        if str(conjunto_saida) != str(lista_resposta_saida_formatacao[ind_conjunto_resposta]):
            parm_teste_objeto.lista_verification_errors.append([caso_teste, \
                              f'\nretornado: {conjunto_saida} \
                                \nesperado: {lista_resposta_saida_formatacao[ind_conjunto_resposta]}'])

def teste_concatenar_taggear_texto(parm_teste_objeto):
  for ndx_exemplo, json_exemplo in enumerate(lista_json_exemplo):
    list_doctos = []
    for docto in json_exemplo['lista_documento']:
      documento = util_modelo.ClassDocto(id_docto=docto['cod'], text=docto['texto'])
      list_doctos.append(documento)
    texto_retornado, pos_to_docto, docto_to_pos_inicio = alvo.concatenar_taggear_texto(list_doctos)
    # print(pos_to_docto)
    # print(texto_retornado)
    # print(docto_to_pos_inicio)
    caso_teste = 'Testar concatenação e taggeamento de texto'
    if texto_retornado != lista_texto_concatenado_esperado[ndx_exemplo]:
      parm_teste_objeto.lista_verification_errors.append([caso_teste, \
                              f'\nretornado: {texto_retornado} \
                                \nesperado: {lista_texto_concatenado_esperado[ndx_exemplo]}'])

    caso_teste = 'Testar correção indicação de doc-id para cada posição'
    if pos_to_docto != lista_pos_to_docto_esperado[ndx_exemplo]:
      parm_teste_objeto.lista_verification_errors.append([caso_teste, \
                              f'\nretornado: {pos_to_docto} \
                                \nesperado: {lista_pos_to_docto_esperado[ndx_exemplo]}'])

    caso_teste = 'Testar correção indicação de início de posição no texto para cada docto-id'
    if docto_to_pos_inicio != lista_docto_to_pos_inicio_esperado[ndx_exemplo]:
      parm_teste_objeto.lista_verification_errors.append([caso_teste, \
                              f'\nretornado: {docto_to_pos_inicio} \
                                \nesperado: {lista_docto_to_pos_inicio_esperado[ndx_exemplo]}'])

def teste_filtrar_resposta_de_unico_documento(parm_teste_objeto):
    caso_teste = 'Testar filtrar resposta unico documento'
    for ind_conjunto_resposta, conjunto_resposta_entrada in enumerate(lista_resposta_entrada_filtro):
        # print(f"ind_conjunto_resposta: {ind_conjunto_resposta}")
        conjunto_saida = alvo.filtrar_resposta_de_unico_documento(conjunto_resposta_entrada, lista_pos_to_docto_entrada_filtro[0])
        # print(ind_conjunto_resposta, conjunto_saida)
        if str(lista_resposta_saida_filtro[ind_conjunto_resposta]) != str(conjunto_saida):
          # print(conjunto_saida)
          # print(lista_resposta_saida_filtro[ind_conjunto_resposta])
          parm_teste_objeto.lista_verification_errors.append([caso_teste, \
                              f'\nretornado: {conjunto_saida} \
                                \nesperado: {lista_resposta_saida_filtro[ind_conjunto_resposta]}'])
          parm_teste_objeto.lista_verification_errors.append([caso_teste, \
                         f'Dif: ver dif'])
        # se_iguais, texto_dif = are_two_lists_equals(conjunto_saida, lista_resposta_saida_filtro)
        #if not se_iguais:


class TestEndPoint(unittest.TestCase):
    """
    Classe de testes - unittest
    """
    maxDiff = None

    def setUp(self):
        warnings.filterwarnings("ignore", category=ResourceWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.lista_verification_errors = []
        print(
            f"\n********* Testes iniciados de filtrar-busca em {strftime('[%Y-%b-%d %H:%M:%S]')} ")

    def tearDown(self):
        """
        ações a serem feitas no final da instância da classe: disparar erros encontrados
        """
        self.assertEqual([], self.lista_verification_errors)
        # self.assertAlmostEqual()

    def teste_a_executar(self):
        """
        casos de testes a realizar
        """
        teste_filtrar_resposta_de_unico_documento(self)
        teste_concatenar_taggear_texto(self)
        teste_formatar_retirar_duplicatas(self)
        return


if __name__ == "__main__":
    unittest.main(verbosity=2)
    # unittest.main()


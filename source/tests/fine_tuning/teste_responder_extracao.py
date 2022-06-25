"""
print('oi')
Desejado comportamento semelhante a https://countwordsfree.com/comparetexts

"""

import warnings
import unittest
import os
import sys
from time import strftime
# para execução direta
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'.\.'))
from source.fine_tuning import responder_extracao  # pylint: disable=wrong-import-position # precisa dos sys.path antes
from source.fine_tuning.modelo_reader import lista_modelos_reader  # pylint: disable=wrong-import-position # precisa dos sys.path antes
from source.fine_tuning import trata_reader as alvo # pylint: disable=wrong-import-position # precisa dos sys.path antes

from source.fine_tuning import util_modelo


json_resposta_esperada_topk_1 = {'texto_resposta': 'Flamengo,',
'score': 0.8342542052268982,
'lista_referencia': [[22, 31]]}

"""
json_resposta_esperada_topk_2 = [{'texto_resposta': 'Flamengo,',
  'score': 0.8342542052268982,
  'lista_referencia': [[22, 31]]}]
"""

json_resposta_esperada_topk_2 = [{'texto_resposta': 'Flamengo,',
 'score': 0.8342542052268982, 'lista_referencia': [[22, 31]]},
{'texto_resposta': 'Flamengo, melhor time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo Flamengo,',
 'score': 0.064724400639534, 'lista_referencia': [[22, 131]]}]

json_resposta_esperada_topk_9 = [
 {'texto_resposta': 'Flamengo,',
 'score': 0.8461036747321486,
 'lista_referencia': [[22, 31], [122, 131]]},
 {'texto_resposta': 'Flamengo, melhor time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo Flamengo,',
 'score': 0.06712984014302492,
 'lista_referencia': [[22, 131]]},
 {'texto_resposta': 'o Flamengo,',
 'score': 0.005499169230461121,
 'lista_referencia': [[20, 31]]},
 {'texto_resposta': 'Flamengo, melhor time do Brasil,',
 'score': 0.0019257370731793344,
 'lista_referencia': [[22, 54]]},
 {'texto_resposta': 'Flamengo, melhor time do Brasil, ficou sem jogar a final da Libertadores.',
 'score': 0.0011339493794366717,
 'lista_referencia': [[22, 95]]}
 ]

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

json_exemplo['sigla_modelo_reader'] = lista_modelos_reader[0]

def are_two_lists_equals(list1: list, list2: list) -> bool:
    """
    Compare two lists and logs the difference.
    :param list1: first list.
    :param list2: second list.
    :return:      if there is difference between both lists.
    """
    diff = [i for i in list1 + list2 if i not in list1 or i not in list2]
    result_bool = len(diff) == 0
    text_dif = f'There are {len(diff)} differences:\n{diff[:5]}'
    return result_bool, text_dif

def teste_responder_extracao(parm_teste_objeto, parm_sigla_modelo_reader: str):
    """
    teste de resposta por extração
    """

    json_exemplo["sigla_modelo"] = parm_sigla_modelo_reader

    json_exemplo["top_k_reader"] = 1
    caso_teste = 'teste trazer 1 resposta'
    resposta =  responder_extracao.responder_extracao( json_exemplo)
    # print(f" Modelo: {parm_sigla_modelo_reader}; resultado: {resposta}")
    caso_teste = 'teste responder multiplos documentos - trazer todos top_k_reader=3'
    if len(resposta['lista_sugestao_resposta']) != 1:
        parm_teste_objeto.lista_verification_errors.append([caso_teste, 'Retornadas: '+ str(len(resposta['lista_sugestao_resposta'])) + ' respostas',
                                            'Esperados: 1 resposta '])

    caso_teste = 'teste estrutura resposta ok'
    msg_dif = util_modelo.compare_dicionarios_chaves(json_resposta_esperada_topk_1, resposta['lista_sugestao_resposta'][0],"Esperado", "Encontrado")
    if msg_dif != "":
        parm_teste_objeto.lista_verification_errors.append([caso_teste, 'Retornada estrutura dif do esperado: '+ msg_dif])

    caso_teste = 'teste valor resposta ok'
    msg_dif = util_modelo.compare_dicionarios(json_resposta_esperada_topk_1, resposta['lista_sugestao_resposta'][0],"Esperado", "Encontrado")
    if msg_dif != "":
        parm_teste_objeto.lista_verification_errors.append([caso_teste,
            'Resposta (conteúdo) difere do esperado: '+ msg_dif])


    caso_teste = 'teste responder multiplos respostas concatenadas na referência- quantidade 2'
    json_exemplo["top_k_reader"] = 2
    resposta =  responder_extracao.responder_extracao( json_exemplo)
    # print(resposta)

    # msg_dif = util_modelo.compare_dicionarios(json_resposta_esperada_topk_2, resposta['lista_sugestao_resposta'][0],"Esperado", "Encontrado")
    se_iguais, msg_dif = are_two_lists_equals(json_resposta_esperada_topk_2, resposta['lista_sugestao_resposta'],)
    if not se_iguais:
        parm_teste_objeto.lista_verification_errors.append([caso_teste,
            'Resposta (conteúdo) difere do esperado: '+ msg_dif])

    caso_teste = 'teste responder multiplos documentos - quantidade 9'
    json_exemplo["top_k_reader"] = 9
    resposta =  responder_extracao.responder_extracao( json_exemplo)
    # print(resposta)
    se_iguais, msg_dif = are_two_lists_equals(json_resposta_esperada_topk_9, resposta['lista_sugestao_resposta'],)
    if not se_iguais:
        parm_teste_objeto.lista_verification_errors.append([caso_teste,
            'Resposta (conteúdo) difere do esperado: '+ msg_dif])


def teste_limite_topk_reader(parm_teste_objeto, parm_sigla_modelo_reader: str):
    """
        https://www.google.com/search?q=error+huggingface+pipeline+for+question-answering+keyerror+%22for+s%2C+e%2C+score+in+zip%28starts%2C+ends%2C+scores%29%22&rlz=1C1GCEA_enBR958BR958&biw=955&bih=1397&sxsrf=ALiCzsa6x9jMs-XL_CZ2PNIg9lSClBEOMA%3A1654871221457&ei=tVSjYpi0G4mi1sQP45uhyAo&ved=0ahUKEwjY4rbai6P4AhUJkZUCHeNNCKkQ4dUDCA4&uact=5&oq=error+huggingface+pipeline+for+question-answering+keyerror+%22for+s%2C+e%2C+score+in+zip%28starts%2C+ends%2C+scores%29%22&gs_lcp=Cgdnd3Mtd2l6EAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsAMyBwgAEEcQsANKBAhBGABKBAhGGABQ1qIBWMTWAWCF2AFoAnABeACAAQCIAQCSAQCYAQCgAQHIAQjAAQE&sclient=gws-wiz
        https://stackoverflow.com/questions/62300836/keyerror-when-using-non-default-models-in-huggingface-transformers-pipeline

        Alguns modelos respondem fora do limite:
                deepset/xlm-roberta-base-squad2-distilled
    """


    json_exemplo["sigla_modelo"] = parm_sigla_modelo_reader


    for limite in range(1, 1000, 5):
        # caso_teste = f'{limite} teste responder multiplos documentos - quantidade enorme'
        json_exemplo["top_k_reader"] = limite
        resposta =  responder_extracao.responder_extracao( json_exemplo)
        print(f"limite {limite} len(resposta['lista_sugestao_resposta']) {len(resposta['lista_sugestao_resposta'])}")


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
            f"\n********* Testes iniciados de responder-extracao em {strftime('[%Y-%b-%d %H:%M:%S]')} ")

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
        teste_realizado=False
        for sigla_modelo in lista_modelos_reader:
            if sigla_modelo == "pierreguillou/bert-large-cased-squad-v1.1-portuguese":
                teste_realizado = True
                # teste_limite_topk_reader(self, sigla_modelo )
                teste_responder_extracao(self, sigla_modelo)
                break
        if not teste_realizado:
            raise Exception("Sem modelo com respostas conhecidas para testes")



if __name__ == "__main__":
    unittest.main(verbosity=2)
    # unittest.main()


print('oi')
from transformers import AutoModelForQuestionAnswering

import os, sys
print(sys.path)
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(r'.\.'))
import copy
from code.fine_tuning.responder_extracao import responder_extracao, json_exemplo  # pylint: disable=wrong-import-position # precisa dos sys.path antes
from code.fine_tuning.modelo_reader import lista_modelos_reader

for modelo_reader in lista_modelos_reader:
    print(modelo_reader)
    # reader['modelo_reader'].score()

json_exemplo_pergunta_resposta = copy.deepcopy(json_exemplo)
json_exemplo_pergunta_resposta["tamanho_max_resposta"] = 30

resposta = responder_extracao(json_exemplo_pergunta_resposta)
print(f"resposta {resposta}")

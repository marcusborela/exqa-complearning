# para inglês, usar código acima de 100.
# 'Instruction: Based on the text below, answer the question:\n\nText:{context}\n\nQuestion:{question}\nResponse:')}
# 'prompt_format': ('Instrução: Com base no texto abaixo, responda à pergunta:\n\nTexto: {context}\n\nPergunta: {question}\nResposta:'

instrucao_0shot_pt = 'Instrução: Com base no texto abaixo, responda à pergunta:\n\n'
instrucao_1shot_pt = 'Instrução: Conforme o exemplo a seguir, com base no texto, responda à pergunta:\n\n'
instrucao_fewshot_pt = 'Instrução: Conforme os exemplos a seguir, com base no texto, responda à pergunta:\n\n'
instrucao_0shot_en = 'Instruction: Based on the text below, answer the question:\n\n'
instrucao_1shot_en = 'Instruction: According to the following example, based on the text, answer the question:\n\n'
instrucao_fewshot_en = 'Instruction: According to the following examples, based on the text, answer the question:\n\n'

exemplo1_pt = 'Exemplo:\nTexto: João nasceu em 1980 e trabalha no TCU desde 2005\nPergunta: Quem nasceu em 1980?\nResposta:João\n\n'
exemplo1_en = 'Example:\nText: John was born in 1980 and has worked at TCU since 2005\nQuestion: Who was born in 1980?\nAnswer:John\n\n'

texto_questao_resposta_pt = 'Texto:{context}\n\nPergunta:{question}\nResposta:'
texto_questao_resposta_en = 'Text:{context}\n\nQuestion:{question}\nAnswer:'

dict_prompt_format ={
1: {"prompt": instrucao_0shot_pt + texto_questao_resposta_pt, "num_shot":0},
101: {"prompt": instrucao_0shot_en + texto_questao_resposta_en, "num_shot":0},
2: {"prompt": instrucao_1shot_pt + exemplo1_pt + texto_questao_resposta_pt, "num_shot":1},
102: {"prompt": instrucao_1shot_en + exemplo1_en + texto_questao_resposta_en, "num_shot":1},
}

for cod, body in list(dict_prompt_format.items()):
    print('Codigo:', cod, 'Shots:', body['num_shot'])
    print(body['prompt'])
    print('=============================================================')
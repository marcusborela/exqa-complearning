# para inglês, usar código acima de 100.
# 'Instruction: Based on the text below, answer the question:\n\nText:{context}\n\nQuestion:{question}\nResponse:')}
# 'prompt_format': ('Instrução: Com base no texto abaixo, responda à pergunta:\n\nTexto: {context}\n\nPergunta: {question}\nResposta:'

instrucao_pt = 'Instrução: Com base no texto abaixo, responda de forma sucinta à pergunta, evitando repetir palavras da pergunta:\n\n'
instrucao_en = 'Instruction: Based on the text below, answer the question succinctly, avoiding repeating words from the question:\n\n'

exemplo1_pt = 'Exemplo:\n\nTexto: Marcus nasceu em 1980 e trabalha no TCU desde 2005.\n\nPergunta: Quem nasceu em 1980?\nResposta: Marcus\n\n'
exemplo1_en = 'Example:\n\nText: Marcus was born in 1980 and has worked at TCU since 2005.\n\nQuestion: Who was born in 1980?\nAnswer: Marcus\n\n'

exemplo2_pt_tptp = 'Exemplo:\n\nTexto: Marcus nasceu em 1980 e trabalha no TCU desde 2005.\n\nPergunta: Quem nasceu em 1980?\nResposta: Marcus\n\nTexto: Marcus nasceu em 1980 e trabalha no TCU desde 2005.\n\nPergunta: Em qual ano Marcus nasceu?\nResposta: 1980\n\n'
exemplo2_en_tptp = 'Example:\n\nText: Marcus was born in 1980 and has worked at TCU since 2005.\n\nQuestion: Who was born in 1980?\nAnswer: Marcus\n\nText: Marcus was born in 1980 and has worked at TCU since 2005.\n\nQuestion: In which year was Marcus born?\nAnswer: 1980\n\n'

exemplo2_pt_tpp = 'Exemplo:\n\nTexto: Marcus nasceu em 1980 e trabalha no TCU desde 2005.\n\nPergunta: Quem nasceu em 1980?\nResposta: Marcus\nPergunta: Em qual ano Marcus nasceu?\nResposta: 1980\n\n'
exemplo2_en_tpp = 'Example:\n\nText: Marcus was born in 1980 and has worked at TCU since 2005.\n\nQuestion: Who was born in 1980?\nAnswer: Marcus\nQuestion: In which year was Marcus born?\nAnswer: 1980\n\n'

texto_questao_resposta_pt = 'Texto:{context}\n\nPergunta:{question}\nResposta:'
texto_questao_resposta_en = 'Text:{context}\n\nQuestion:{question}\nAnswer:'

dict_prompt_format ={
1: {"prompt": instrucao_pt + texto_questao_resposta_pt, "num_shot":0},
101: {"prompt": instrucao_en + texto_questao_resposta_en, "num_shot":0},
2: {"prompt": instrucao_pt + exemplo1_pt + texto_questao_resposta_pt, "num_shot":1},
102: {"prompt": instrucao_en + exemplo1_en + texto_questao_resposta_en, "num_shot":1},
3: {"prompt": instrucao_pt + exemplo2_pt_tptp + texto_questao_resposta_pt, "num_shot":2, "format_example": "tptp"},
103: {"prompt": instrucao_en + exemplo2_en_tptp + texto_questao_resposta_en, "num_shot":2, "format_example": "tptp"},
4: {"prompt": instrucao_pt + exemplo2_pt_tpp + texto_questao_resposta_pt, "num_shot":2, "format_example": "tpp"},
104: {"prompt": instrucao_en + exemplo2_en_tpp + texto_questao_resposta_en, "num_shot":2, "format_example": "tpp"},
}

for cod, body in list(dict_prompt_format.items()):
    print('Codigo:', cod, 'Shots:', body['num_shot'])
    if "format_example" in body:
        print("format_example", body["format_example"])
    print(body['prompt'])
    print('================')
# para inglês, usar código acima de 100.
# 'Instruction: Based on the text below, answer the question:\n\nText:{context}\n\nQuestion:{question}\nResponse:')}
# 'prompt_format': ('Instrução: Com base no texto abaixo, responda à pergunta:\n\nTexto: {context}\n\nPergunta: {question}\nResposta:'
dict_prompt_format ={
1: {"prompt": 'Instrução: Com base no texto abaixo, responda à pergunta:\n\nTexto: {context}\n\nPergunta: {question}\nResposta:',
"num_shot":0},
101: {"prompt": 'Instruction: Based on the text below, answer the question:\n\nText:{context}\n\nQuestion:{question}\nResponse:',
    "num_shot":0},
2: {"prompt": 'Instrução: Com base no texto abaixo, assim como feito no exemplo, responda à pergunta:\n\nExemplo:\n\nTexto: João nasceu em 1980 e trabalha no TCU desde 2005\n\nPergunta: Quem nasceu em 1980?\n\nResposta:João\n\nTexto: {context}\n\nPergunta: {question}\nResposta:',
    "num_shot":1},
}


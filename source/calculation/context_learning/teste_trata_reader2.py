from source.calculation.context_learning.trata_reader2 import Reader

parm_dict_config = {'answer_start': 'Resposta:',
                    'answer_stop': ['.', '\n', '\n'],
                    'answer_skips': 0,
                    'generator_sample': False,
                    'generator_temperature': 1,
                    'answer_max_length': 2048,
                    'prompt_format': ('''Instrução: Com base no texto abaixo, responda à pergunta:

Texto: {context}

Pergunta: {question}
Resposta:''')}

def teste_context_reader():
    # reader = Reader('EleutherAI/gpt-neo-1.3B', parm_dict_config)
    reader = Reader('EleutherAI/gpt-j-6B', parm_dict_config)
    ##print(reader.info())
    print('Reader instanciated. Answering...')
    response = reader.answer_one_question('Qual o melhor time do Brasil?', 'Por causa da chuva, o Flamengo, melhor time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo Flamengo, o melhor time, na minha Família. Além da bicicleta, em 1990, eu tinha a honra de ser flamenguista.  Ontem, quando reli os documentos do tribunal, descobri que em 1990 quando tomei posse, minha declaração de bens só continha uma bicicleta.')
    print('\n----------------- Prompt -----------------')
    print(response[0]['prompt'])
    print('\n----------------- Answer -----------------')
    print(response[0]['answer_text'])

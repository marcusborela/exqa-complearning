from trata_reader_LEO import Reader

parm_dict_config = {'answer_start': 'Resposta:',
                    'answer_stop': ['.', '\n', '\n'],
                    'answer_skips': 0,
                    'generator_sample': False,
                    'generator_temperature': 1,
                    'answer_max_length': 200,
                    'prompt_format': ('''Instrução: Com base no texto abaixo, responda à pergunta:

Texto:{context}

Pergunta:{question}
Resposta:''')}

def teste_context_reader():
    reader = Reader('EleutherAI/gpt-neo-1.3B', parm_dict_config)
    print('Reader instanciated. Answering...')
    responses = reader.answer_one_question('Qual o melhor time do Brasil?', 'Por causa da chuva, o Flamengo, melhor time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo Flamengo, o melhor time, na minha Família. Além da bicicleta, em 1990, eu tinha a honra de ser flamenguista.  Ontem, quando reli os documentos do tribunal, descobri que em 1990 quando tomei posse, minha declaração de bens só continha uma bicicleta.')
    print('\n----------------- Prompt -----------------')
    print(responses[0]['prompt'])
    for i, response in enumerate(responses):
        print('\n----------------- Answer', i + 1, '-----------------')
        print(response['answer_text'])
        #print('\n----------------- Generation -----------------')
        #print(response[0]['gen_text'])

teste_context_reader()
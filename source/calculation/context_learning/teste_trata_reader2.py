from trata_reader import Reader


parm_dict_config = {
                # parâmetros complementares
                "num_top_k": 2,
                "num_batch_size": 1,
                "num_doc_stride": 0,
                "num_factor_multiply_top_k": 0,
                "if_handle_impossible_answer":False,
                "num_max_answer_length":150,
                # parâmetros context
                    'answer_start': 'Resposta:',
                    'answer_stop': ['.', '\n', '\n'],
                    'answer_skips': 0,
                    'generator_sample': False,
                    'generator_temperature': 1,
                    'answer_max_length': 150,
                    'prompt_format': ('''Instrução: Com base no texto abaixo, responda à pergunta:

Texto: {context}

Pergunta: {question}
Resposta:''')}

def teste_context_reader(name_model:str):
    # reader = Reader(, parm_dict_config)
    reader = Reader(name_model, parm_dict_config)
    print(f"reader.info()\n {reader.info}")
    print('Reader instanciated. Answering...')
    response = reader.answer_one_question('Qual o melhor time do Brasil?', 'Por causa da chuva, o Flamengo, melhor time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo Flamengo, o melhor time, na minha Família. Além da bicicleta, em 1990, eu tinha a honra de ser flamenguista.  Ontem, quando reli os documentos do tribunal, descobri que em 1990 quando tomei posse, minha declaração de bens só continha uma bicicleta.')
    print('\n----------------- Prompt -----------------')
    print(response[0]['prompt'])
    for ndx, resp in enumerate(response):
        print('\n----------------- Answer -----------------')
        print(f"{ndx}: {resp['generated_text']}")

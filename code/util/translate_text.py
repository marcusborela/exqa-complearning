"""
just to explore coding
"""
print('oi')

# import googletrans
# print(googletrans.LANGUAGES)

from googletrans import Translator

translator = Translator()
texto_origem = 'eu te amo'
lingua_origem = 'pt'
lingua_destino = 'en'
texto_traduzido = translator.translate('eu te amo',dest=lingua_destino,src=lingua_origem)
print(texto_traduzido.text)

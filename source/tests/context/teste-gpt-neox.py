import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

torch.cuda.empty_cache()
print('torch.cuda.is_available():', torch.cuda.is_available())

print('Loading model...')
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")

print('Loading tokenizer...')
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

print('Transfering model to GPU...')
device = torch.device("cuda")
model.to(device)

print('Generating prompt...')
prompt = ('''Instrução: Dado o contexto, responda à pergunta:

Contexto: Por causa da chuva, o Flamengo, melhor time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo Flamengo, o melhor time, na minha Família. Além da bicicleta, em 1990, eu tinha a honra de ser flamenguista.  Ontem, quando reli os documentos do tribunal, descobri que em 1990 quando tomei posse, minha declaração de bens só continha uma bicicleta.
Pergunta: De onde os documentos foram relidos?
Resposta: Do tribunal

Contexto: Por causa da chuva, o Flamengo, melhor time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo Flamengo, o melhor time, na minha Família. Além da bicicleta, em 1990, eu tinha a honra de ser flamenguista.  Ontem, quando reli os documentos do tribunal, descobri que em 1990 quando tomei posse, minha declaração de bens só continha uma bicicleta.
Pergunta: Quando tomei posse?
Resposta: 1990

Contexto: Por causa da chuva, o Flamengo, melhor time do Brasil, ficou sem jogar a final da Libertadores. Há muitos torcedores pelo Flamengo, o melhor time, na minha Família. Além da bicicleta, em 1990, eu tinha a honra de ser flamenguista.  Ontem, quando reli os documentos do tribunal, descobri que em 1990 quando tomei posse, minha declaração de bens só continha uma bicicleta.
Pergunta: Qual o melhor time do Brasil?
Resposta:''') 

print('Tokenizing...')
tokenized = tokenizer(prompt, return_tensors="pt")
input_ids = tokenized.input_ids.to(device)
attention_mask = tokenized.attention_mask.to(device)

print('Generating answer...')
with torch.no_grad():
    gen_tokens = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=1,
        max_length=2048,
    )

print('Decoding answer...')
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
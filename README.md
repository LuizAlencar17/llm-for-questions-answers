#### 1° OBS: Para obter o modelo trainado, por favor. Acesse o arquivo https://huggingface.co/luizalencar/NLP-TPFINAL-LLM/blob/main/tpfinal-model_checkpoint.zip, e extraia os arquivos

#### 2° OBS: Para obter uma melhor experiência utilize o Google Colab, não esqueça de fazer upload de todos os arquivos da pasta "sample_data". Isso é fundamental! ;)


## 1. Relatório de Pré-processamento 

1.1. Descrição detalhada das etapas de download, extração e pré-processamento dos textos das legislações.

- O download dos arquivos foi feito de forma manual, utilizando o navegador Google Chrome.

- A extração dos textos foi feita através da biblioteca “pdfplumber” conforme o código abaixo:

Código:
```python 
def load_data(file_path):
    return json.load(open(file_path))

def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def has_text_in_pdf(pdf):
    return bool(pdf) and bool(re.search('[a-zA-Z]', pdf)) and len(pdf) > MIN_PDF_SIZE

def list_pdf_files(directory):
    pdf_files = [f"{directory}/{f}" for f in os.listdir(directory) if f.endswith('.pdf')]
    return pdf_files

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

RAW_TEXT_PATH = "./sample_data/raw_texts.json"
PDFS_PATH = "./sample_data"
RAW_PDFS_TEXTS = []
ALLOWED_PDFS = []
MIN_PDF_SIZE = 100

for pdf in tqdm(list_pdf_files(PDFS_PATH)):
    text = extract_text_from_pdf(pdf)
    if has_text_in_pdf(text):
        RAW_PDFS_TEXTS.append(text)
        ALLOWED_PDFS.append(pdf)
save_data(RAW_PDFS_TEXTS, RAW_TEXT_PATH)
RAW_PDFS_TEXTS = load_data(RAW_TEXT_PATH)
RAW_PDFS_TEXTS
```

Saída:
```python 
 'MINISTÉRIO DA EDUCAÇÃO\nCONSELHO NACIONAL DE EDUCAÇÃO\nCONSELHO PLENO\nRESOLUÇÃO Nº 2, DE 1º DE JULHO DE 2015 (*) (**) (***)\nDefine as Diretrizes Curriculares Nacionais para a\nformação inicial em nível superior (cursos de\nlicenciatura, cursos de formação pedagógica para\ngraduados e cursos de segunda licenciatura) e para\na formação continuada.\nO Presidente do Conselho Nacional de Educação, no uso de suas atribuições\nlegais e tendo em vista o disposto na Lei nº 9.131, de 24 de novembro de 1995, Lei nº 9.394,\nde 20 de dezembro de 1996, Lei nº 11.494, de 20 de junho de 2007, Lei nº 11.502, de 11 de\njulho de 2007, Lei nº 11.738, de 16 de julho de 2008, Lei nº 12.796, de 4 de abril de 2013, Lei\nnº 13.005, de 25 de junho de 2014, observados os preceitos dos artigos 61 até 67 e do artigo\n87 da Lei nº 9.394, de 1996, que dispõem sobre a formação de profissionais do magistério, e\nconsiderando o Decreto nº 6.755, de 29 de janeiro de 2009, as Resoluções CNE/CP nº 1, de\n18 de fevereiro de 2002, CNE/CP nº 2, de 19 de fevereiro de 2002, CNE/CP nº 1, de 15 de\nmaio de 2006, CNE/CP nº 1, de 11 de fevereiro de 2009, CNE/CP nº 3, de 15 de junho de\n2012, e as Resoluções CNE/CEB nº 2, de 19 de abril de 1999, e CNE/CEB nº 2, de 25 de\nfevereiro de 2009, as Diretrizes Curriculares Nacionais da Educação Básica, bem como o\nParecer CNE/CP nº 2, de 9 de junho de 2015, homologado por Despacho do Ministro de\nEstado da Educação publicado no Diário Oficial do União de 25 de junho de 2015, e\nCONSIDERANDO que a consolidação das normas nacionais para a formação\nde profissionais do magistério para a educação básica é indispensável para o projeto nacional\nda educação brasileira, em seus níveis e suas modalidades da educação, tendo em vista a\nabrangência e a complexidade da educação de modo geral e, em especial, a educação escolar\ninscrita na sociedade;\nCONSIDERANDO que a concepção sobre conhecimento, educação e ensino é\nbasilar para garantir o projeto da educação nacional, superar a fragmentação das políticas\npúblicas e a desarticulação institucional por meio da instituição do Sistema Nacional de\nEducação, sob relações de cooperação e colaboração entre entes federados e sistemas\neducacionais;\nCONSIDERANDO que a igualdade de condições para o acesso e a\npermanência na escola; a liberdade de aprender, ensinar, pesquisar e divulgar a cultura, o\npensamento, a arte e o saber; o pluralismo de ideias e de concepções pedagógicas; o respeito à\nliberdade e o apreço à tolerância; a valorização do profissional da educação; a gestão\ndemocrática do ensino público; a garantia de um padrão de qualidade; a valorização da\nexperiência extraescolar; a vinculação entre a educação escolar, o trabalho e as práticas\nsociais; o respeito e a valorização da diversidade étnico-racial, entre outros, constituem\nprincípios vitais para a melhoria e democratização da gestão e do ensino;\nCONSIDERANDO que as instituições de educação básica, seus processos de\norganização e gestão e projetos pedagógicos cumprem, sob a legislação vigente, um papel\n(*) Resolução CNE/CP 2/2015. Diário Oficial da União, Brasília, 2 de julho de 2015 – Seção 1 – pp. 8-12.\n(**) Retificação publicada no DOU de 3/7/2015, Seção 1, p. 28: Na Resolução CNE/CP nº 2, de 1º de julho de\n2015, publicada no Diário Oficial da União de 2/7/2015, Seção 1, pp. 8-12, no Art. 17, § 1º, p. 11, onde se lê: "II\n- atividades ou cursos de extensão, oferecida por atividades formativas diversas, em consonância com o projeto\nde extensão aprovado pela instituição de educação superior formadora;", leia-se: "III - atividades ou cursos de\nextensão, oferecida por atividades formativas diversas, em consonância com o projeto de extensão aprovado pela\ninstituição de educação superior formadora;".\n(***) Alterada pela Resolução CNE/CP nº 1, de 9 de agosto de 2017.\nestratégico na formação requerida nas diferentes etapas (educação infantil, ensino\nfundamental e ensino médio) e modalidades da educação básica;\nCONSIDERANDO a necessidade de articular as Diretrizes Curriculares\nNacionais para a Formação Inicial e Continuada, em Nível Superior, e as Diretrizes\nCurriculares Nacionais para a Educação Básica;\nCONSIDERANDO os princípios que norteiam a base comum nacional para a\nformação inicial e continuada, tais como: a) sólida formação teórica e interdisciplinar; b)\nunidade teoria-prática; c) trabalho coletivo e interdisciplinar; d) compromisso social e\nvalorização do profissional da educação; e) gestão democrática; f) avaliação e regulação dos\ncursos de formação;\nCONSIDERANDO a articulação entre graduação e pós-graduação e entre\npesquisa e extensão como princípio pedagógico essencial ao exercício e aprimoramento do\nprofissional do magistério e da prática educativa;\nCONSIDERANDO a docência ....',

```

- Para o pré-processamento dos textos, foi feita a remoção de caracteres e símbolos indesejados; a conversão dos textos em minúsculas; a remoção de espaços em branco extras; a remoção de stopwords (palavras comuns que podem não ser úteis para análise); e por fim, uma etapa de lematização de palavras. Conforme a figura abaixo:

Código:
```python 
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(words)
    return cleaned_text

CLEANED_TEXT_PATH = "./sample_data/cleaned_texts.json"
CLEANED_PDFS_TEXTS = []

for text in tqdm(RAW_PDFS_TEXTS):
    CLEANED_PDFS_TEXTS.append(preprocess_text(text))
save_data(CLEANED_PDFS_TEXTS, CLEANED_TEXT_PATH)
CLEANED_PDFS_TEXTS = load_data(CLEANED_TEXT_PATH)
CLEANED_PDFS_TEXTS[-3:]

```

Saída:
```python 
 'poder executivo ministrio educao universidade federal amazona conselho ensino pesquisa extenso consepe cmara ensino graduao ceg resoluo n 472014 dispe sobre norma internas relativas processo seletivo extramacro pse prreitor ensino graduao presidente cmara ensino graduao conselho ensino pesquisa extenso universidade federal amazona uso atribuies estatutrias considerando necessidade atualizao norma ingresso modalidades transferncia facultativa interna externa portador diploma curso superior transferncia curso reopo universidade federal amazona modo ajustlas legislao vigente considerando dispe lei 939496 demais legislao vigor tocante transferncia facultativa considerando disposto caput artigos 41 42 estatuto artigos 73 74 regimento geral universidade federal amazona bem inciso ii art 44 inciso iv art 53 lei 939496 relativamente preenchimento fixao vagas cursos graduao considerando disposto resoluo n 01794consepe define reas conhecimento universidade amazona resoluo ...
```

1.2. Ferramentas utilizadas e desafios enfrentados durante o processo. 

- Como mencionado anteriormente, a ferramenta utilizada para extração de textos foi a “pdfplumber” e o principal desafio foi extrair pdfs cujo os textos faziam parte de imagens, tornando inviável processar alguns desses arquivos.

1.3. Base de dados

- A base de dados está no arquivo “sample_data/cleaned_texts.json”


## 2. Base de Dados Sintética

2.1. Arquivo contendo os 1000 exemplos de perguntas e respostas gerados. 

- A base de dados sintética com os 1000 exemplos está no arquivo “sample_data/instructions_and_responses.json”


2.2. Metodologia utilizada para a geração dos exemplos.

- Para gerar a base de dados foi utilizado o modelo Gemini do Google, utilizando um prompt de comando passando o texto extraído do PDF e pedido para o modelo criar os exemplos de perguntas e respostas, conforme o código abaixo:

Código:
```python 
import re
import json
import time
import google.generativeai as genai

from tqdm import tqdm

GOOGLE_API_KEY = "AIzaSyBsXzWgDSE-naipvx7I79AeAnsGQlHMO2w"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro-latest')

def extract_instruction_response_pairs(string: str):
  string = string.replace('\n', '').replace('\r', '')
  pattern = re.compile(r'\{.*?\}', re.DOTALL)
  json_strings = pattern.findall(string)
  return [json.loads(json_str) for json_str in json_strings]


def get_synthetics_instructions_and_responses(text, max_instructions=1):
  start = time.time()
  pred = ""
  synthetics = load_data(SYNTHETICS_INSTRUCTIONS_PATH)
  for idx in tqdm(range(max_instructions)):
    try:
      prompt = f"""### Baseado no texto abaixo, gerar 10 pares de respostas relevantes e detalhadas de instruções.
      Certifique-se de que a Instrução e Resposta esteja em um array no formato json:\n\n
      ### Exemplo: {{"Instrução": "a instrução", "Resposta": "a resposta"}}\n\n
      ### Texto: {text}\n\n
      ### Resposta:"""
      response = model.generate_content([prompt], stream=True)
      response.resolve()
      pred = response.text
      synthetics.extend(extract_instruction_response_pairs(pred))
    except Exception as e:
      print(f"ERROR: {e}. RESPONSE: {pred}")
  save_data(synthetics, SYNTHETICS_INSTRUCTIONS_PATH)
  print("\n\nTime: {} seconds".format(time.time()-start))
  return synthetics

SYNTHETICS_INSTRUCTIONS_PATH = "./sample_data/instructions_and_responses.json"
instructions_and_responses = [get_synthetics_instructions_and_responses(text=text, max_instructions=10) for text in CLEANED_PDFS_TEXTS]

```

## 3. Modelo Treinado

3.1. Código fonte utilizado para o treinamento do modelo de linguagem com LoRA/QLoRA.

- Todo o código fonte, incluindo o modelo utilizando LoRa, está localizado no arquivo “handler.ipynb”, conforme a figura abaixo.

Código:
```python 
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pierreguillou/gpt2-small-portuguese"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
max_seq_length = 768
trainingArgs = TrainingArguments(
    output_dir='output',
    num_train_epochs=50,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    learning_rate=2e-4
)
peft_config = LoraConfig(
  lora_alpha=32,
  lora_dropout=0.1,
  r=8,
  task_type="CAUSAL_LM",
)
model.train()
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=prompt_instruction_format_to_train,
    max_seq_length=max_seq_length,
    args=trainingArgs
)
PATH_TO_MODEL_CHECKPOINT = "sample_data/checkpoint"
history = trainer.train()
model.save_pretrained(PATH_TO_MODEL_CHECKPOINT)
tokenizer.save_pretrained(PATH_TO_MODEL_CHECKPOINT)
history
### -> Criando um arquivo zip com todos os arquivos necessarios para carregar o modelo
# import shutil
# shutil.make_archive('tpfinal-model_checkpoint', 'zip', PATH_TO_MODEL_CHECKPOINT)

```

3.2. Relatório de desempenho do modelo, incluindo métricas de avaliação e análise de resultados.

- - A participação de treino contém 900 (90%) de perguntas e respostas, enquanto a de teste ficou com 100 (10%) da base.


- O modelo foi treinado com 50 épocas, e suas métricas de avaliação na partição de traino foram as seguintes:

```python 
{'train_runtime': 1007.3974,
 'train_samples_per_second': 5.757,
 'train_steps_per_second': 1.439,
 'total_flos': 2281122614476800.0,
 'train_loss': 1.3643781148976293,
 'epoch': 50.0}
```

- As métricas de avaliação na partição de teste foram as seguintes:

```python 
{'eval_loss': 13.955838203430176,
 'eval_perplexity': 1150650.75,
 'eval_runtime': 10.5195,
 'eval_samples_per_second': 1.141,
 'eval_steps_per_second': 0.19,
 'epoch': 50.0}
```

- Em relação à análise de resultados, a figura abaixo mostra exemplos de entradas e predições realizadas pelo modelo após o treinamento.

Código:
```python 
def get_prediction(model, prompt):
  model.eval()
  inputs = tokenizer(prompt, return_tensors="pt").to(device)
  outputs = model.generate(
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      max_length=max_seq_length,
      return_dict_in_generate=True
  )
  pred = tokenizer.decode(outputs.sequences[0])
  return pred.split("<|endoftext|>")[0]


for i in range(10):
  sample = get_sample()
  print(f"PERGUNTA: {sample['Instrução']}")
  print(f"RESPOSTA CORRETA: {sample['Resposta']}")
  print(f"RESPOSTA DO MODELO: {get_prediction(model, sample['Instrução'])}\n\n")

```

Saida:
```python 
  Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
PERGUNTA: Qual é a data de vigência da Resolução?
RESPOSTA CORRETA: 27 de abril de 2007.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
RESPOSTA DO MODELO: Qual é a data de vigência da Resolução?

RESPOSTA: O decreto-lei nº 5.618/1969, de 15 de dezembro de 1969, estabelece a data de vigência da Resolução.



PERGUNTA: Como solicitar o aproveitamento das atividades?
RESPOSTA CORRETA: O aproveitamento das atividades deve ser solicitado exclusivamente por meio do portal eCampus, onde o discente deverá anexar a comprovação da conclusão do programa.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
RESPOSTA DO MODELO: Como solicitar o aproveitamento das atividades?

RESPOSTA: O aproveitamento das atividades é realizado por meio de documentos, visitas e visitas, que devem ser obrigatoriamente registrados no cadastro de alunos da instituição.



PERGUNTA: Identifique o responsável por assinar a portaria.
RESPOSTA CORRETA: Luiz Simo Botelho Neve, Pró-Reitor em Exercício.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
RESPOSTA DO MODELO: Identifique o responsável por assinar a portaria.



PERGUNTA: Qual é o órgão responsável pela coordenação do programa Residência Pedagógica?
RESPOSTA CORRETA: Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (Capes).
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
RESPOSTA DO MODELO: Qual é o órgão responsável pela coordenação do programa Residência Pedagógica?

RESPOSTA: O PROGUNG é responsável pela coordenação do programa Residência Pedagógica.



PERGUNTA: Qual é o objetivo do Decreto nº 8.537/15?
RESPOSTA CORRETA: O Decreto nº 8.537/15 regulamenta a Lei nº 12.852/13 e a Lei nº 12.933/13, que tratam do benefício de meia-entrada em acesso a eventos artístico-culturais e esportivos, além de estabelecer procedimentos e critérios para reserva de vagas para jovens de baixa renda em veículos do sistema de transporte coletivo interestadual.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
RESPOSTA DO MODELO: Qual é o objetivo do Decreto nº 8.537/15?

RESPOSTA: O objetivo do decreto é o aproveitamento das atividades de ensino e pesquisa para a formação de professores e alunos de graduação e pós-graduação, bem como para a formação de professores e alunos de pós-graduação.



PERGUNTA: Quando a resolução entra em vigor?
RESPOSTA CORRETA: A resolução entra em vigor na data de sua aprovação pela Câmara de Ensino de Graduação.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
RESPOSTA DO MODELO: Quando a resolução entra em vigor?

RESPOSTA: O decreto-lei nº 5.636, de 30 de dezembro de 1969, que regulamenta a obrigatoriedade da matrícula de jovens de 15 a 17 anos no ensino médio, é revogado.



PERGUNTA: Quais são os requisitos para o professor ministrar disciplinas semipresenciais?
RESPOSTA CORRETA: Possuir capacitação específica em docência a distância (EAD) em ambiente virtual de aprendizagem (AVA), obtida em curso reconhecido e credenciado pelo Ministério da Educação.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
RESPOSTA DO MODELO: Quais são os requisitos para o professor ministrar disciplinas semipresenciais?

RESPOSTA: O professor deve apresentar a documentação necessária para o processo de apresentação do curso.



PERGUNTA: Como são disciplinados os casos omissos na resolução?
RESPOSTA CORRETA: Os casos omissos disciplinados nesta resolução deverão ser decididos pela Câmara de Ensino de Graduação, observada a legislação pertinente.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
RESPOSTA DO MODELO: Como são disciplinados os casos omissos na resolução?

RESPOSTA: Os casos omissos na resolução são considerados omissos na resolução.



PERGUNTA: Qual é o critério para homologação da autodeclaração do discente como indígena?
RESPOSTA CORRETA: Registro administrativo de nascimento indígena (RANI) oficialmente emitido pela Fundação Nacional do Índio (FUNAI).
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
RESPOSTA DO MODELO: Qual é o critério para homologação da autodeclaração do discente como indígena?

RESPOSTA: O discente deve ser reconhecido como indígena, sem a condição de ser impedido de matrícula.



PERGUNTA: Quais são os documentos necessários para solicitar a Carteira de Identificação Estudantil (CIE)?
RESPOSTA CORRETA: Documento de identificação com foto, comprovante de matrícula e, no caso de estudantes de baixa renda, comprovação dos requisitos estabelecidos pela legislação.
RESPOSTA DO MODELO: Quais são os documentos necessários para solicitar a Carteira de Identificação Estudantil (CIE)?

RESPOSTA: O documento deve ser assinado pelo coordenador do curso, que deve ser aprovado pelo colegiado.

```

## 4. Sistema de RAG Implementado

4.1. Código fonte do sistema de RAG.

- Todo o código fonte, incluindo o sistema RAG implementado, está localizado no arquivo “handler.ipynb”, conforme a figura abaixo. Para fazer o gerenciador de índices, utilizei o modelo ‘'paraphrase-MiniLM-L6-v2'’. Código abaixo mostra como funciona o processo de indexação utilizado.

Código:
```python 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.storage import InMemoryStore

documents = []
for pdf_path in ALLOWED_PDFS:
  documents.extend(PyPDFLoader(pdf_path).load())

documents_texts = [i.page_content for i in documents]
documents_texts[0] # print demonstrativo do primeiro documento

def create_embeddings(data):
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return embedder.encode(data, convert_to_tensor=False)

# nessa função, passamos nossa pergunta na variavel "query" e conseguimos os textos relacionados a pergunta
def get_rag_data(index, query, k=1):
    query_embedding = create_embeddings([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [documents_texts[i] for i in indices[0]][0]


rag_embeddings = create_embeddings(documents_texts)
index = faiss.IndexFlatL2(rag_embeddings.shape[1])
index.add(np.array(rag_embeddings))

```

4.2. Demonstração de funcionamento do sistema com exemplos de perguntas e respostas.

- As figuras abaixo mostram alguns exemplos de perguntas e respostas geradas pelo RAG e modelo treinado.

Código:
```python 
def create_embeddings(data):
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return embedder.encode(data, convert_to_tensor=False)

# nessa função, passamos nossa pergunta na variavel "query" e conseguimos os textos relacionados a pergunta
def get_rag_data(index, query, k=1):
    query_embedding = create_embeddings([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [documents_texts[i] for i in indices[0]][0]


rag_embeddings = create_embeddings(documents_texts)
index = faiss.IndexFlatL2(rag_embeddings.shape[1])
index.add(np.array(rag_embeddings))

```

Saida:
```python 
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
RESPOSTA DO MODELO USANDO [RAG]: 
Você é um assistente para tarefas de resposta a perguntas.
Use o contexto fornecido apenas para responder à seguinte pergunta:

CONTEXTO:  7 § 1o – O Conselho Universitário funcionará em primeira convocação, com a presença 
da maioria dos seus membros e suas decisões, ressalvados os  casos expressos neste 
Estatuto, serão tomadas pela maioria dos votos dos presentes.  
 
§ 2o – Perderá o mandato o conselheiro que, sem causa justificada, faltar a 03 (três)  
reuniões consecutivas ou a 05 (cinco)  alternadas.  
 
§ 3o – A convocação do Conselho  Universitário far -se-á  por aviso pessoal, com a 
antecedência mínima de 02 (dois) dias úteis, mencionando -se a pauta e sinopse dos 
assuntos a serem tratados.  
 
§ 4o – Observado o disposto neste artigo, o Regimento Interno do Conselho 
Universitário disporá sobre as sessões plenárias e sobre a constituição, competência e 
funcionamento de comissões, quando for o caso, bem como acerca da organização da 
secretaria dos órgãos de deliberação superior.  
 
Art. 14 - O Conselho  de Administração será constituído pelos  seguintes membros:  
a) o Reitor, como Presidente;  
b) os Pró -Reitores de Administração, de Planejamento e de Assuntos da 
Comunidade Universitária;  
c) os Diretores de unidades acadêmicas;  
d) 03 (três) representantes dos servidores técnico -administrativos;  
e) 02 (dois)  rep resentantes discentes;  
f) 01 (um) representante da comunidade local ou regional.  
 
§ 1o – Os membros do Conselho de Administração a que se referem as alíneas d, e e f  
terão mandato de 1 (um) ano e serão escolhidos na forma do Regimento Geral;  
 
§  2º  - O Co nselho de Administração delibera em plenário ou através das seguintes 
câmaras:  
a) Câmara de  Administração e Finanças;  
b) Câmara de Recursos Humanos;  
c) Câmara de Assuntos da Comunidade Universitária.  
 
Art. 15  - Compete ao Conselho de Administração:  
I. conhecer de recursos interpostos de atos dos diretores das unidades 
acadêmicas e dos conselhos departamentais, assim como dos pró -reitores e 
dos dirigentes de órgãos suplementares, em matéria administrativa;  
II. homologar tabelas de valores a serem  cob rados pela Universidade;  
III. deliberar  sobre atos do Reitor praticados ad referendum   do Conselho;  
IV. deliberar sobre criação, modificação e extinção de órgãos administrativos;  
V. aprovar normas sobre admissão, lotação, remoção e aperfeiç oamento de 
pessoal técnico -administrativo;  

PERGUNTA: Quais são as vagas oferecidas no PSE?

RESPOSTA: As vagas oferecidas no PSE são destinadas a alunos com deficiência física, mental ou psicológica, que tenham condições de locomoção, locomoção e exploração, que tenham condições de saúde, que tenham condições de conforto e que tenham condições de segurança.

```
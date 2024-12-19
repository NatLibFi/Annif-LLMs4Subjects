import json
import glob
import sys
import time
from json.decoder import JSONDecodeError
from itertools import islice
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel

LLM_SYSTEM_PROMPT = "You are a professional translator specialized in translating bibliographic metadata."
LLM_PROMPT = """
Your task is to translate the given title and description into English and German.
When the text is already in the correct language, do not change or summarize it, keep it all as it is.
Always include the full description in both languages from BEGIN to END even if it includes other information such as the table of contents.
Respond with JSON having this structure:

{"title_de": "<title in German>",
 "title_en": "<title in English>",
 "desc_de": "<description in German>",
 "desc_en": "<description in English>"}

Translate this title and description:
""".strip()

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 2048
TEMPERATURE = 0.0
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.95
MAX_BATCHED_TOKENS = 16384

# Read input from file and write output to files alongside the original
source_file = sys.argv[1]
en_filename = source_file.replace('/all', '/en')
de_filename = source_file.replace('/all', '/de')

# Initialize vLLM engine and tokenizer
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_model_len=MAX_MODEL_LEN,
    enable_chunked_prefill=True,
    max_num_batched_tokens=MAX_BATCHED_TOKENS,
    enable_prefix_caching=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def parse_record(record):
    text, uris = record
    fields = text.split(' ¤ ')
    title = fields[0]
    desc = " ".join(fields[1:])
    return (title, desc, uris)


def generate_messages(title, desc):
    prompt = LLM_PROMPT + "\n\n" + \
	f"Title: {title}\n" + f"Description BEGIN:\n{desc}\nEND"

    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    return messages


def messages_to_token_ids(messages):
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


class TranslatedOutput(BaseModel):
    title_de: str
    title_en: str
    desc_de: str
    desc_en: str


# Function to process a batch of records
def process_batch(batch):
    prompt_token_ids = []
    record_uris = []

    for record in batch:
        text, desc, uris = parse_record(record)
        messages = generate_messages(text, desc)
        prompt_token_ids.append(messages_to_token_ids(messages))
        record_uris.append(uris)

    en_records = []
    de_records = []

    json_schema = TranslatedOutput.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params=SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, guided_decoding=guided_decoding_params)
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

    generated_text = [output.outputs[0].text for output in outputs]
    for text in generated_text:
        try:
            data = json.loads(text)
            en_records.append({'title': data['title_en'], 'desc': data['desc_en']})
            de_records.append({'title': data['title_de'], 'desc': data['desc_de']})
        except JSONDecodeError:
            print(f"Cannot parse {data}, skipping record")
            en_records.append(None)
            de_records.append(None)

    return en_records, de_records, record_uris

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

def read_file(filename):
    with open(filename) as cf:
        for idx, line in enumerate(cf):
            text, uris = line.strip().split('\t', 1)
            yield (text, uris)

def format_record(title, desc, uris):
    desc = " ".join(desc.strip().split())  # normalize whitespace, just to be sure
    return f"{title} ¤ {desc}\t{uris}"

# Process input lines in batches
batch_size = 256

starttime = time.time()
ndocs = 0

with open(en_filename, 'w') as en_file, open(de_filename, 'w') as de_file:
    for batch in batched(read_file(source_file), batch_size):
        ndocs += len(batch)
        en_records, de_records, record_uris = process_batch(batch)

        for en_rec, de_rec, uris in zip(en_records, de_records, record_uris):
            if not en_rec:
                continue  # skipping invalid record
            print(format_record(en_rec['title'], en_rec['desc'], uris), file=en_file)
            print(format_record(de_rec['title'], de_rec['desc'], uris), file=de_file)

elapsed = time.time() - starttime
print(f"Time taken: {elapsed} seconds ({elapsed/ndocs} seconds per document), batch size {batch_size}")

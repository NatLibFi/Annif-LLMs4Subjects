import glob
import sys
import time
import os.path
import gzip
import random
import time
import rdflib
from rdflib.namespace import SKOS
from itertools import islice
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

LLM_SYSTEM_PROMPT = "You are a professional metadata manager."
LLM_PROMPT = """
Your task is to create new bibliographic metadata: document titles and descriptions.

Here is an example document title and description in <LANGUAGE> with the following subject keywords: <OLD_KEYWORDS>

<TITLE_DESC>


Generate a new document title and description in <LANGUAGE>. Respond with only the title and description, nothing else. Create a new title and description that match the following subject keywords: <NEW_KEYWORDS>
""".strip()

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 4096
TEMPERATURE = 0.5
REPETITION_PENALTY = 1.1
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.95
MAX_BATCHED_TOKENS = 16384
LANGUAGES = {'de': 'German', 'en': 'English'}


# Read input from gzipped TSV file and write output to a gzipped TSV file
vocab_filename = sys.argv[1]
lang = sys.argv[2]
source_filename = sys.argv[3]
dest_filename = sys.argv[4]

# Read the vocabulary into a dictionary {uri: prefLabel}
print(f"Loading vocabulary {vocab_filename} labels in '{lang}' language...")
uri_to_label = {}
g = rdflib.Graph()
g.parse(vocab_filename, format='turtle')
for uri, label in g.subject_objects(SKOS.prefLabel):
    if label.language == lang:
        uri_to_label[str(uri)] = str(label)

print(f"Loaded vocabulary with {len(uri_to_label)} concepts.")

# Initialize vLLM engine and tokenizer
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_model_len=MAX_MODEL_LEN,
    max_num_batched_tokens=MAX_BATCHED_TOKENS,
    enable_prefix_caching=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# make sure random choices are different on each run
random.seed(time.time())

def subject_keywords(subjects):
    return '; '.join([uri_to_label[uri] for uri in subjects if uri in uri_to_label])


def generate_messages(title_desc, lang, old_subjects, new_subjects):
    prompt = LLM_PROMPT
    prompt = prompt.replace('<LANGUAGE>', LANGUAGES[lang])
    prompt = prompt.replace('<OLD_KEYWORDS>', subject_keywords(old_subjects))
    prompt = prompt.replace('<NEW_KEYWORDS>', subject_keywords(new_subjects))
    prompt = prompt.replace('<TITLE_DESC>', title_desc)

    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    return messages


def messages_to_token_ids(messages):
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


# Function to process a batch of records
def process_batch(batch, lang):
    prompt_token_ids = []
    new_record_subjects = []

    for record in batch:
        title_desc = f"{record['title']}\n\n{record['desc']}"
        # old subjects in example record that exist in the vocabulary
        old_subjects = [uri for uri in record['subjects'] if uri in uri_to_label]
        # new subjects: add one random subject from the vocabulary
        new_subjects = [random.choice(list(uri_to_label.keys()))] + old_subjects
        new_record_subjects.append(new_subjects)

        messages = generate_messages(title_desc, lang, old_subjects, new_subjects)
        prompt_token_ids.append(messages_to_token_ids(messages))

    sampling_params=SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, repetition_penalty=REPETITION_PENALTY)
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

    generated_text = [output.outputs[0].text for output in outputs]

    return generated_text, new_record_subjects

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

def clean_uri(uri):
    if uri.startswith('<') and uri.endswith('>'):
        return uri[1:-1]
    return uri

def read_tsv(filename):
    with gzip.open(filename, 'rt') as inf:
        for line in inf:
            if '\t' not in line.strip():
                continue
            text, uris = line.strip().split('\t', 1)
            if ' ¤ ' not in text:
                continue
            title, desc = text.split(' ¤ ', 1)
            subjects = [clean_uri(uri) for uri in uris.split()]
            yield {'title': title, 'desc': desc, 'subjects': subjects}

# Process input lines in batches
batch_size = 512

starttime = time.time()
ndocs = 0

with gzip.open(dest_filename, 'wt') as dest_file:
    for batch in batched(read_tsv(source_filename), batch_size):
        ndocs += len(batch)
        new_texts, new_subjects = process_batch(batch, lang)

        for orig_rec, new_text, new_subject in zip(batch, new_texts, new_subjects):
            if '\n\n' not in new_text:
                continue
            title, desc = new_text.split('\n\n', 1)
            title = ' '.join(title.strip().split())
            desc = ' '.join(desc.strip().split())
            uri_string = ' '.join([f"<{uri}>" for uri in new_subject])
            print(f"{title} ¤ {desc}\t{uri_string}", file=dest_file)

        dest_file.flush()

elapsed = time.time() - starttime
print(f"Time taken: {elapsed} seconds ({elapsed/ndocs} seconds per document), batch size {batch_size}")

#!/usr/bin/env python

import os
import sys
import json
from itertools import islice
from openai import AzureOpenAI

translated = set()

with open('gnd-en.tsv') as tsvf:
    for line in tsvf:
        if line.strip() == '':
            continue
        uri, term = line.strip().split('\t')
        translated.add(uri)


client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)



MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a professional translator specialized in translating controlled vocabularies such as information retrieval thesauri and classifications."
BATCH_SIZE = 100

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def cdata_to_labels(cdata):
    labels = []
    for ltype in ['label_de']:
        if ltype in cdata:
            labels.append(cdata[ltype])
    return ' ; '.join(labels)


def chat(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    answer = response.choices[0].message.content
    if answer is None:
        print(prompt, file=sys.stderr)
        print(response, file=sys.stderr)
    return answer


PROMPT = """
Your task is to translate terms from the The Gemeinsame Normdatei (GND, Integrated Authority File), a carefully curated thesaurus known for its precise and respectful terminology. These terms are used for academic and informational purposes and are presented in German. Please maintain the list structure and translate each term into English. Only return the list of translated terms, no explanations are needed.

This translation work is part of an educational and informational project aimed at enhancing accessibility and understanding of diverse concepts across languages. It is important to handle all terms, especially those pertaining to sensitive subjects such as health conditions, with accuracy and respect as intended by the thesaurus editors.

Example input:

1. Individualisierte Person
2. Familie
3. Schlagwort
4. Sicherung

Translated output for the above examples:

1. Differentiated person
2. Family
3. Subject heading
4. Safeguarding

Now translate the following thesaurus terms to English:

"""


gnd = json.load(sys.stdin)

gnd_to_translate = { uri: cdata
                     for uri, cdata in gnd.items()
                     if uri not in translated }

for batch in batched(gnd_to_translate.items(), BATCH_SIZE):
    uris, cdatas = zip(*batch)
    terms = []
    for idx, cdata in enumerate(cdatas):
        labels = cdata_to_labels(cdata)
        terms.append(f"{idx+1}. {labels}")

    prompt = PROMPT + "\n".join(terms)
    response = chat(prompt)
    for idx, line in enumerate(response.strip().split("\n")):
        number, term = line.strip().split('.', 1)
        print("\t".join((uris[idx], term.strip())))

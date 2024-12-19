#!/usr/bin/env python

import rdflib
from rdflib.namespace import RDF, RDFS, SKOS, OWL, DCTERMS
import sys
import json
import collections

g = rdflib.Graph()
g.parse(sys.stdin, format='turtle')

data = {}

gnd_cs = rdflib.URIRef('https://d-nb.info/gnd/')

# regular concepts
for concept in g.subjects(RDF.type, SKOS.Concept):
    cdata = collections.defaultdict(list)

    # prefLabel
    for label in g.objects(concept, SKOS.prefLabel):
        prop = f"label_{label.language}"
        cdata[prop] = str(label)
    # altLabel
    for altlabel in g.objects(concept, SKOS.altLabel):
        prop = f"altlabel_{altlabel.language}"
        cdata[prop].append(str(altlabel))
    # type
    for type in g.objects(concept, RDF.type):
        if type != SKOS.Concept:
            cdata['type'] = str(type)
    # collections
    for coll in g.subjects(SKOS.member, concept):
        cdata['collection'].append(str(coll))
    # semantic relations
    for prop in 'broader', 'narrower', 'related':
        for val in g.objects(concept, SKOS[prop]):
            cdata[prop].append(str(val))
    # documentation properties
    for prop in 'note', 'scopeNote', 'definition':
        for val in g.objects(concept, SKOS[prop]):
            langprop = f"{prop}_{val.language}"
            cdata[langprop].append(str(val))
    # timestamps
    for prop in 'created', 'modified':
        for val in g.objects(concept, DCTERMS[prop]):
            cdata[prop] = str(val)


    data[str(concept)] = cdata

# collections
for coll in g.subjects(RDF.type, SKOS.Collection):
    if (coll, OWL.deprecated, rdflib.Literal(True)) in g:
        continue

    cdata = collections.defaultdict(list)

    # prefLabel
    for label in g.objects(coll, SKOS.prefLabel):
        prop = f"label_{label.language}"
        cdata[prop] = str(label)
    # altLabel
    for altlabel in g.objects(coll, SKOS.altLabel):
        prop = f"altlabel_{label.language}"
        cdata[prop].append(str(label))
    # type
    for type in g.objects(coll, RDF.type):
        if type != SKOS.Concept:
            cdata['type'] = str(type)
    data[str(coll)] = cdata


json.dump(data, sys.stdout)

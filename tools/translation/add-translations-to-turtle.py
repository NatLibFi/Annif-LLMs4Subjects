import csv
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import SKOS

# gpath = "../../shared-task-datasets/GND/GND-Subjects-all_dnb-skos.ttl"
gpath = "../../shared-task-datasets/GND/GND-Subjects-tib-core_dnb-skos.ttl"

# Load the existing RDF vocabulary
g = Graph()
g.parse(gpath, format="ttl"
)


# Read the TSV file and add labels to the RDF vocabulary
# with open("gnd-all-en.tsv", "r", encoding="utf-8") as tsvfile:
with open("gnd-core-en.tsv", "r", encoding="utf-8") as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        if len(row) == 2:
            resource_uri = URIRef(row[0])
            label = row[1]
            g.add((resource_uri, SKOS.prefLabel, Literal(label, lang="en")))

# Save the updated RDF vocabulary
dest_gpath = gpath.replace(".ttl", "-with-en-labels.ttl")
g.serialize(destination=dest_gpath, format="ttl")

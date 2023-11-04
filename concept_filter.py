import json
import data_utils
import conceptset_utils

"""
CLASS_SIM_CUTOFF: Concenpts with cos similarity higher than this to any class will be removed
OTHER_SIM_CUTOFF: Concenpts with cos similarity higher than this to another concept will be removed
MAX_LEN: max number of characters in a concept

PRINT_PROB: what percentage of filtered concepts will be printed
"""

CLASS_SIM_CUTOFF = 0.85
OTHER_SIM_CUTOFF = 0.9
MAX_LEN = 30
PRINT_PROB = 1

dataset = "negative"
device = "cuda"

save_name = "./concept/{}_filtered_new.txt".format(dataset)

#EDIT these to use the initial concept sets you want

with open("data/concept_sets/gpt3_init/gpt3_{}_important.json".format(dataset), "r") as f:
    important_dict = json.load(f)
with open("data/concept_sets/gpt3_init/gpt3_{}_superclass.json".format(dataset), "r") as f:
    superclass_dict = json.load(f)
with open("data/concept_sets/gpt3_init/gpt3_{}_around.json".format(dataset), "r") as f:
    around_dict = json.load(f)
    
with open(data_utils.LABEL_FILES[dataset], "r") as f:
    classes = f.read().split("\n")
    
concepts = set()

for values in important_dict.values():
    concepts.update(set(values))

for values in superclass_dict.values():
    concepts.update(set(values))
    
for values in around_dict.values():
    concepts.update(set(values))

print(len(concepts))


concepts = conceptset_utils.remove_too_long(concepts, MAX_LEN, PRINT_PROB)

concepts = conceptset_utils.filter_too_similar_to_cls(concepts, classes, CLASS_SIM_CUTOFF, device, PRINT_PROB)

concepts = conceptset_utils.filter_too_similar(concepts, OTHER_SIM_CUTOFF, device, PRINT_PROB)


with open(save_name, "w") as f:
    f.write(concepts[0])
    for concept in concepts[1:]:
        f.write("\n" + concept)




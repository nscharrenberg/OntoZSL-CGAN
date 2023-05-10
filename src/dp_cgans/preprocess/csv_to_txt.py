import os

import pandas as pd

from dp_cgans.embeddings import log
from dp_cgans.preprocess import HPOHeaders, Relation
from dp_cgans.preprocess.utils import normalize_string, get_association_subclass, get_association_name, \
    get_frequency_association_codes, get_frequency_association_classes, get_frequency_association_entities, \
    get_diagnostic_criteria_entities, get_diagnostic_criteria_association_classes


def csv_to_txt(file_path: str, destination_path: str, verbose: bool = True):
    log(text=f'🔄️  About to start Preprocessing', verbose=verbose)

    if not os.path.exists(file_path):
        log(text=f'❌️The file {file_path} does not exists.', verbose=True)
        return

    if not os.path.exists(destination_path):
        log(text=f'🗂️  Destination path does not exist, creating this path...', verbose=verbose)
        os.makedirs(destination_path)

    log(text=f'🔄️  Reading CSV from {file_path}', verbose=verbose)
    df = pd.read_csv(file_path, dtype="object")
    dataset = read_dataset(df)
    df = dataset['df']

    association_entitites = dataset['association_entities']
    diagnostic_criteria_entities = get_diagnostic_criteria_entities()
    frequency_association_entities = get_frequency_association_entities()

    hpo_entities = dataset['hpo_entities']
    orpha_entities = dataset['orpha_entities']

    has_object_triples = dataset['has_object_triples']
    has_subject_triples = dataset['has_subject_triples']
    has_frequency_triples = dataset['has_frequency_triples']
    has_diagnostic_criteria_triples = dataset['has_diagnostic_criteria_triples']

    log(text=f'🔄️  Instantiating Triples...', verbose=verbose)
    triples = []
    triples_names = []
    entities = []
    entities_names = []
    relations = []

    sub_class_of_parents_dict = get_sub_class_of_parents(triples, triples_names, association_entitites,
                                                         diagnostic_criteria_entities, frequency_association_entities,
                                                         hpo_entities, orpha_entities, verbose)
    triples = sub_class_of_parents_dict['triples']
    triples_names = sub_class_of_parents_dict['triples_names']
    entities_and_parent_class = sub_class_of_parents_dict['entities_and_parent_class']

    other_properties_dict = get_other_properties(triples, triples_names, has_object_triples, has_subject_triples,
                                                 has_frequency_triples, has_diagnostic_criteria_triples, hpo_entities,
                                                 orpha_entities, frequency_association_entities,
                                                 diagnostic_criteria_entities, association_entitites, verbose)
    triples = other_properties_dict['triples']
    triples_names = other_properties_dict['triples_names']

    parents_dict = get_parents(entities, entities_names, entities_and_parent_class, verbose)
    entities = parents_dict['entities']
    entities_names = parents_dict['entities_names']

    entities_dict = get_entities(entities, entities_names, association_entitites, diagnostic_criteria_entities,
                                 frequency_association_entities, hpo_entities, orpha_entities, verbose)
    entities = entities_dict['entities']
    entities_names = entities_dict['entities_names']

    relations_dict = get_relations(relations, verbose)
    relations = relations_dict['relations']

    write_to_file(destination_path, triples, triples_names, entities, entities_names, relations, verbose)


def read_dataset(df: pd.DataFrame, verbose: bool = True) -> dict:
    hpo_entities = {}
    orpha_entities = {}

    association_entitites = {}

    has_object_triples = []
    has_subject_triples = []
    has_frequency_triples = []
    has_diagnostic_criteria_triples = []

    df[HPOHeaders.ORPHA_CODE.value] = df[HPOHeaders.ORPHA_CODE.value].map(lambda x: f"ORPHA:{x}")

    log(text=f'🔄️  Reading Dataset...', verbose=verbose)
    for orpha, orpha_name, frequency, hp, hpo_name, dc in zip(
            df[HPOHeaders.ORPHA_CODE.value],
            df[HPOHeaders.NAME.value],
            df[HPOHeaders.HPO_FREQUENCY_NAME.value],
            df[HPOHeaders.HPO_ID.value],
            df[HPOHeaders.HPO_TERM.value],
            df[HPOHeaders.DIAGNOSTIC_CRITERIA_NAME.value]
    ):
        if hp not in hpo_entities:
            hpo_entities[hp] = normalize_string(hpo_name)

        if orpha not in orpha_entities:
            orpha_entities[orpha] = normalize_string(orpha_name)

        association_class = get_association_subclass(orpha, frequency, hp)
        association_class_name = get_association_name(orpha, frequency, hp, orpha_entities, hpo_entities)
        association_entitites[association_class] = association_class_name

        has_object_triples.append((association_class, Relation.ASSOCIATION_HAS_OBJECT.value, hp))
        has_subject_triples.append((association_class, Relation.ASSOCIATION_HAS_SUBJECT.value, orpha))
        has_frequency_triples.append((association_class, Relation.HAS_FREQUENCY.value,
                                      get_frequency_association_codes().get(
                                          get_frequency_association_classes().get(frequency))))
        has_diagnostic_criteria_triples.append(
            (association_class, Relation.HAS_DC_ATTRIBUTE,
             get_diagnostic_criteria_association_classes().get(dc, 'exclusion')))

    return {
        "df": df,
        "hpo_entities": hpo_entities,
        "orpha_entities": orpha_entities,
        "association_entities": association_entitites,
        "has_object_triples": has_object_triples,
        "has_subject_triples": has_subject_triples,
        "has_frequency_triples": has_frequency_triples,
        "has_diagnostic_criteria_triples": has_diagnostic_criteria_triples
    }


def get_sub_class_of_parents(triples, triples_names, association_entitites, diagnostic_criteria_entities,
                             frequency_association_entities, hpo_entities, orpha_entities, verbose: bool = True):
    log(text=f'🔄️  Adding Sub Classes Of Parents...', verbose=verbose)
    entities_and_parent_class = [
        (association_entitites, "association"),
        (diagnostic_criteria_entities, "diagnostic_criteria"),
        (frequency_association_entities, "frequency_association"),
        (hpo_entities, HPOHeaders.HPO_ID.value),
        (orpha_entities, HPOHeaders.ORPHA_CODE.value)
    ]

    for (entities_temp, parent_temp) in entities_and_parent_class:
        for key, value in entities_temp.items():
            triples.append((key, Relation.SUB_CLASS_OF.value, normalize_string(parent_temp)))
            triples_names.append((value, Relation.SUB_CLASS_OF.value, normalize_string(parent_temp)))

    return {
        "triples": triples,
        "triples_names": triples_names,
        "entities_and_parent_class": entities_and_parent_class
    }


def get_other_properties(triples, triples_names, has_object_triples, has_subject_triples, has_frequency_triples,
                         has_diagnostic_criteria_triples, hpo_entities, orpha_entities, frequency_association_entities,
                         diagnostic_criteria_entities, association_entitites, verbose: bool = True):
    log(text=f'🔄️  Adding Other Properties...', verbose=verbose)
    triples_and_entities = [
        (has_object_triples, hpo_entities),
        (has_subject_triples, orpha_entities),
        (has_frequency_triples, frequency_association_entities),
        (has_diagnostic_criteria_triples, diagnostic_criteria_entities)
    ]

    for (properties_temp, entities_temp) in triples_and_entities:
        for (s, r, o) in properties_temp:
            triples.append((s, r, o))
            triples_names.append((association_entitites.get(s), r, entities_temp.get(o)))

    return {
        "triples": triples,
        "triples_names": triples_names
    }


def get_parents(entities, entities_names, entities_and_parent_class, verbose: bool = True):
    log(text=f'🔄️  Adding Parents...', verbose=verbose)
    for i, (key, value) in enumerate(entities_and_parent_class):
        parent = (i, normalize_string(value))
        entities.append(parent)
        entities_names.append(parent)

    return {
        "entities": entities,
        "entities_names": entities_names,
    }


def get_entities(entities, entities_names, association_entitites, diagnostic_criteria_entities,
                 frequency_association_entities, hpo_entities, orpha_entities, verbose: bool = True):
    log(text=f'🔄️  Adding Entities...', verbose=verbose)
    parents_count = len(entities)

    for i, (key, value) in enumerate(
            {**association_entitites, **diagnostic_criteria_entities, **frequency_association_entities, **hpo_entities,
             **orpha_entities}.items()):
        entities.append((i + parents_count, normalize_string(key)))
        entities_names.append((i + parents_count, normalize_string(value)))

    return {
        "entities": entities,
        "entities_names": entities_names
    }


def get_relations(relations, verbose: bool = True):
    log(text=f'🔄️  Adding Relations...', verbose=verbose)
    for i, r in enumerate(
            [Relation.SUB_CLASS_OF.value, Relation.ASSOCIATION_HAS_OBJECT.value, Relation.ASSOCIATION_HAS_SUBJECT.value,
             Relation.HAS_FREQUENCY.value, Relation.HAS_DC_ATTRIBUTE.value]):
        relations.append((i, r))

    return {
        "relations": relations
    }


def write_to_file(destination_path, triples, triples_names, entities, entities_names, relations, verbose: bool = True):
    log(text=f'🔄️  Writing files to "{destination_path}"...', verbose=verbose)
    lists_and_files = [
        (triples, "triples.txt"),
        (triples_names, "triples_names.txt"),
        (entities, "entities.dict"),
        (entities_names, "entities_names.dict"),
        (relations, "relations.dict")
    ]

    for (l, n) in lists_and_files:
        current_file_to_write = os.path.join(destination_path, n)
        log(text=f'🔄️ About to write "{current_file_to_write}"...', verbose=True)
        with open(current_file_to_write, "w") as f:
            for c in l:
                f.write('\t'.join(str(e) for e in c) + '\n')

            log(text=f'🔄️ File has been written.', verbose=True)

    log(text=f'✅️Success! Files have been saved to {destination_path}.', verbose=True)

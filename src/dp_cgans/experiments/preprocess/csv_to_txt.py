import pandas as pd

from dp_cgans.experiments.preprocess import HPOHeaders, Relation
from dp_cgans.experiments.preprocess.utils import get_association_subclass, get_association_name, normalize_string, \
    get_frequency_association_codes, get_frequency_association_classes, get_diagnostic_criteria_association_classes
from dp_cgans.utils.logging import log


def read_dataset(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Read and process a DataFrame to extract the necessary phenotype data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        verbose (bool): If True, log progress messages. Default is True.

    Returns:
        dict: A dictionary containing the processed data, including the original DataFrame and extracted entities and triples.
    """
    hpo_entities = {}
    orpha_entities = {}

    association_entities = {}

    has_object_triples = []
    has_subject_triples = []
    has_frequency_triples = []
    has_diagnostic_criteria_triples = []

    df[HPOHeaders.ORPHA_CODE.value] = df[HPOHeaders.ORPHA_CODE.value].map(lambda x: f"ORPHA:{x}")

    log(text=f"Reading Dataset...", verbose=verbose)

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
        association_entities[association_class] = association_class_name

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
        "association_entities": association_entities,
        "has_object_triples": has_object_triples,
        "has_subject_triples": has_subject_triples,
        "has_frequency_triples": has_frequency_triples,
        "has_diagnostic_criteria_triples": has_diagnostic_criteria_triples
    }


def get_sub_class_of_parents(triples: list, triples_names: list, association_entities: dict, diagnostic_criteria_entities: dict,
                             frequency_association_entities: dict, hpo_entities: dict, orpha_entities: dict, verbose: bool = True):
    """
    Add subclass relationships to the given triples and triples_names lists.

    Args:
        triples (list): The list of triples.
        triples_names (list): The list of triple names.
        association_entities (dict): The association entities dictionary.
        diagnostic_criteria_entities (dict): The diagnostic criteria entities dictionary.
        frequency_association_entities (dict): The frequency association entities dictionary.
        hpo_entities (dict): The HPO entities dictionary.
        orpha_entities (dict): The ORPHA entities dictionary.
        verbose (bool): If True, log progress messages. Default is True.

    Returns:
        dict: A dictionary containing the updated triples, triples_names, and entities_and_parent_class.
    """
    log(text=f"Adding sub classes of parents...", verbose=verbose)

    entities_and_parent_class = [
        (association_entities, "association"),
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


def get_other_properties(triples: list, triples_names: list, has_object_triples: list, has_subject_triples: list, has_frequency_triples: list,
                         has_diagnostic_criteria_triples: list, hpo_entities: dict, orpha_entities: dict, frequency_association_entities: dict,
                         diagnostic_criteria_entities: dict, association_entities: dict, verbose: bool = True):
    """
    Add other properties to the given triples and triples_names lists.

    Args:
        triples (list): The list of triples.
        triples_names (list): The list of triple names.
        has_object_triples (list): The list of triples representing the "has_object" relationship.
        has_subject_triples (list): The list of triples representing the "has_subject" relationship.
        has_frequency_triples (list): The list of triples representing the "has_frequency" relationship.
        has_diagnostic_criteria_triples (list): The list of triples representing the "has_dc_attribute" relationship.
        hpo_entities (dict): The HPO entities dictionary.
        orpha_entities (dict): The ORPHA entities dictionary.
        frequency_association_entities (dict): The frequency association entities dictionary.
        diagnostic_criteria_entities (dict): The diagnostic criteria entities dictionary.
        association_entities (dict): The association entities dictionary.
        verbose (bool): If True, log progress messages. Default is True.

    Returns:
        dict: A dictionary containing the updated triples and triples_names.
    """
    log(text=f"Adding other properties...", verbose=verbose)

    triples_and_entities = [
        (has_object_triples, hpo_entities),
        (has_subject_triples, orpha_entities),
        (has_frequency_triples, frequency_association_entities),
        (has_diagnostic_criteria_triples, diagnostic_criteria_entities)
    ]

    for (properties_temp, entities_temp) in triples_and_entities:
        for (s, r, o) in properties_temp:
            triples.append((s, r, o))
            triples_names.append((association_entities.get(s), r, entities_temp.get(o)))

    return {
        "triples": triples,
        "triples_names": triples_names
    }


def get_parents(entities: list, entities_names: list, entities_and_parent_class: list, verbose: bool = True):
    """
    Add parent entities to the given entities and entities_names lists.

    Args:
        entities (list): The list of entities.
        entities_names (list): The list of entity names.
        entities_and_parent_class (list): The list of entities and parent classes.
        verbose (bool): If True, log progress messages. Default is True.

    Returns:
        dict: A dictionary containing the updated entities and entities_names.
    """
    log(text=f"Adding parents...", verbose=verbose)

    for i, (key, value) in enumerate(entities_and_parent_class):
        parent = (i, normalize_string(value))
        entities.append(parent)
        entities_names.append(parent)

    return {
        "entities": entities,
        "entities_names": entities_names,
    }


def get_entities(entities: list, entities_names: list, association_entities: dict, diagnostic_criteria_entities: dict,
                 frequency_association_entities: dict, hpo_entities: dict, orpha_entities: dict, verbose: bool = True):
    """
    Add entities to the given entities and entities_names lists.

    Args:
        entities (list): The list of entities.
        entities_names (list): The list of entity names.
        association_entities (dict): The association entities dictionary.
        diagnostic_criteria_entities (dict): The diagnostic criteria entities dictionary.
        frequency_association_entities (dict): The frequency association entities dictionary.
        hpo_entities (dict): The HPO entities dictionary.
        orpha_entities (dict): The ORPHA entities dictionary.
        verbose (bool): If True, log progress messages. Default is True.

    Returns:
        dict: A dictionary containing the updated entities and entities_names.
    """
    log(text=f"Adding entities...", verbose=verbose)

    parents_count = len(entities)

    for i, (key, value) in enumerate(
            {**association_entities, **diagnostic_criteria_entities, **frequency_association_entities, **hpo_entities,
             **orpha_entities}.items()):
        entities.append((i + parents_count, normalize_string(key)))
        entities_names.append((i + parents_count, normalize_string(value)))

    return {
        "entities": entities,
        "entities_names": entities_names
    }


def get_relations(relations: list, verbose: bool = True):
    """
    Add relations to the given relations list.

    Args:
        relations (list): The list of relations.
        verbose (bool): If True, log progress messages. Default is True.

    Returns:
        dict: A dictionary containing the updated relations.
    """
    log(text=f"Adding relations...", verbose=verbose)

    for i, r in enumerate(
            [Relation.SUB_CLASS_OF.value, Relation.ASSOCIATION_HAS_OBJECT.value, Relation.ASSOCIATION_HAS_SUBJECT.value,
             Relation.HAS_FREQUENCY.value, Relation.HAS_DC_ATTRIBUTE.value]):
        relations.append((i, r))

    return {
        "relations": relations
    }


def write_to_file(destination_path: str, lists_and_files: list, verbose: bool = True):
    """
    Write lists of phenotype data to files in the specified destination path.

    Args:
        destination_path (str): The path where the files will be saved.
        lists_and_files (list): A list of tuples containing the data lists and their corresponding file names.
        verbose (bool): If True, log progress messages. Default is True.
    """

    log(text=f"Writing files to \"{destination_path}\"...", verbose=verbose)

    for (l, n) in lists_and_files:
        current_file_to_write = f"{destination_path}/{n}"
        log(text=f"About to write \"{current_file_to_write}\"...", verbose=verbose)

        with open(current_file_to_write, "w") as f:
            for c in l:
                f.write('\t'.join(str(e) for e in c) + '\n')

            log(text=f"File has been written...", verbose=verbose)

    log(text=f"Files have been saved to \"{destination_path}\".", verbose=verbose)

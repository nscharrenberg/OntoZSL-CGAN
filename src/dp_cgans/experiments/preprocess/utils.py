from dp_cgans.experiments.preprocess import HPOHeaders


def get_association_name(code: str, frequency: str, id: str, orphanet_entities: dict, hpo_entities: dict) -> str:
    """
    Retrieves association name based on code, frequency, and ID, using orphanet_entities and hpo_entities dictionaries.

    Args:
        code: The code used to retrieve the orphanet_entity from orphanet_entities dictionary.
        frequency: The frequency used to retrieve the frequency_code from get_frequency_association_codes() dictionary.
        id: The ID used to retrieve the hpo_entity from hpo_entities dictionary.
        orphanet_entities: A dictionary mapping codes to orphanet entities.
        hpo_entities: A dictionary mapping IDs to hpo entities.

    Returns:
        The association name in a normalized string format, combining orphanet_entity, hpo_entity, and frequency_code.
    """
    orphanet_entity = orphanet_entities.get(code)
    hpo_entity = hpo_entities.get(id)
    frequency_code = get_frequency_association_codes().get(get_frequency_association_classes().get(frequency))

    return normalize_string(f"{orphanet_entity} and {hpo_entity} {frequency_code} association")


def get_association_subclass(code: str, frequency: str, id: str) -> str:
    """
    Retrieves association subclass based on code, frequency, and ID.

    Args:
        code: The code used to compose the association subclass.
        frequency: The frequency used to retrieve the frequency_code from get_frequency_association_classes() dictionary.
        id: The ID used to compose the association subclass.

    Returns:
        The association subclass as a formatted string, combining code, ID, and frequency_code.
    """
    return f"{code}_{id}_FREQ:{get_frequency_association_classes().get(frequency)}"


def normalize_string(value: str) -> str:
    """
    Normalizes the given string by converting it to lowercase and replacing spaces with underscores.

    Args:
        value: The string to be normalized.

    Returns:
        The normalized string.
    """
    return value.lower().replace(" ", "_")


def get_frequency_association_classes() -> dict:
    """
    Returns a dictionary mapping frequency association classes to their corresponding codes.

    Returns:
        A dictionary mapping frequency association classes to codes.
    """
    return {
        "Obligate (100%)": "OB",
        "Very frequent (99-80%)": "VF",
        "Frequent (79-30%)": "FR",
        "Occasional (29-5%)": "OC",
        "Very rare (<4-1%)": "VR",
        "Excluded (0%)": "EX"
    }


def get_frequency_association_codes() -> dict:
    """
    Returns a dictionary mapping frequency association codes to their corresponding entities.

    Returns:
        A dictionary mapping frequency association codes to entities.
    """
    return {
        "OB": "obligate",
        "VF": "very_frequent",
        "FR": "frequent",
        "OC": "occasional",
        "VR": "very_rare",
        "EX": "excluded"
    }


def get_diagnostic_criteria_association_classes() -> dict:
    """
    Returns a dictionary mapping diagnostic criteria association classes to their corresponding codes.

    Returns:
        A dictionary mapping diagnostic criteria association classes to codes.
    """
    return {
        "Diagnostic criterion": "diagnostic_criterion",
        "Pathognomonic sign": "pathognomonic_sign"
    }


def get_diagnostic_criteria_entities() -> dict:
    """
    Returns a dictionary mapping diagnostic criteria entities to their corresponding entities.

    Returns:
        A dictionary mapping diagnostic criteria entities to entities.
    """
    return {
        "diagnostic_criterion": "diagnostic_criterion",
        "pathognomonic_sign": "pathognomonic_sign",
        "exlusion": "exclusion"
    }


def get_frequency_association_entities() -> dict:
    """
    Returns a dictionary mapping frequency association entities to their corresponding entities.

    Returns:
        A dictionary mapping frequency association entities to entities.
    """
    return {
        "obligate": "obligate",
        "very_frequent": "very_frequent",
        "frequent": "frequent",
        "occasional": "occasional",
        "very_rare": "very_rare",
        "excluded": "excluded"
    }


def get_orpha_iri(orphacode):
    """
    Returns the URL for the given orphanet code.

    Args:
        orphacode: The orphanet code.

    Returns:
        The URL representing the orphanet code.
    """
    return f"http://www.orpha.net/ORDO/Orphanet_{orphacode}"


def get_frequency_distribution():
    """
    Returns a dictionary representing a frequency distribution.

    Returns:
        A dictionary representing a frequency distribution.
    """
    return {  # frequency ids + associated probability
        28405: 1,  # Obligate (100%)
        28412: 0.895,  # Very frequent (99-80%)
        28419: 0.545,  # Frequent (79-30%)
        28426: 0.17,  # Occasional (29-5%)
        28433: 0.025,  # Very rare (<4-1%)
        28440: 0  # Excluded (0%)
    }


def find_xml(row, source_tag, target_tag, field, text=True, attrib="id"):
    """
    Finds and retrieves the value of a specific XML tag within a source tag.

    Args:
        row: The dictionary representing a row of data.
        source_tag: The source tag containing the target tag.
        target_tag: The tag to find and retrieve the value from.
        field: The key to assign the retrieved value in the row dictionary.
        text: A boolean indicating whether to retrieve the text value (default: True).
        attrib: The attribute to retrieve if text=False (default: "id").

    Returns:
        The found target tag or None if not found.
    """
    tag = source_tag.find(target_tag)
    tag_v = None

    if tag is not None:
        if text:
            tag_v = tag.text
        elif len(tag.attrib) > 0:
            tag_v = tag.attrib[attrib]

    row[field] = tag_v if tag_v is not None else ""

    return tag


def get_headers():
    """
    Returns the headers for the HPO data.

    Returns:
        The headers as a list of strings.
    """
    return HPOHeaders.values()

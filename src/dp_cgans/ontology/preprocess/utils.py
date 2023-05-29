import matplotlib.pyplot as plt
import numpy as np

from dp_cgans.ontology.preprocess.preprocessing_enums import HPOHeaders


def get_association_name(code: str, frequency: str, id: str, orphanet_entities: dict, hpo_entities: dict) -> str:
    orphanet_entity = orphanet_entities.get(code)
    hpo_entity = hpo_entities.get(id)
    frequency_code = get_frequency_association_codes().get(get_frequency_association_classes().get(frequency))

    return normalize_string(f"{orphanet_entity} and {hpo_entity} {frequency_code} association")


def get_association_subclass(code: str, frequency: str, id: str) -> str:
    return f"{code}_{id}_FREQ:{get_frequency_association_classes().get(frequency)}"


def normalize_string(value: str) -> str:
    return value.lower().replace(" ", "_")


def get_frequency_association_classes() -> dict:
    return {
        "Obligate (100%)": "OB",
        "Very frequent (99-80%)": "VF",
        "Frequent (79-30%)": "FR",
        "Occasional (29-5%)": "OC",
        "Very rare (<4-1%)": "VR",
        "Excluded (0%)": "EX"
    }


def get_frequency_association_codes() -> dict:
    return {
        "OB": "obligate",
        "VF": "very_frequent",
        "FR": "frequent",
        "OC": "occasional",
        "VR": "very_rare",
        "EX": "excluded"
    }


def get_diagnostic_criteria_association_classes() -> dict:
    return {
        "Diagnostic criterion": "diagnostic_criterion",
        "Pathognomonic sign": "pathognomonic_sign"
    }


def get_diagnostic_criteria_entities() -> dict:
    return {
        "diagnostic_criterion": "diagnostic_criterion",
        "pathognomonic_sign": "pathognomonic_sign",
        "exlusion": "exclusion"
    }


def get_frequency_association_entities() -> dict:
    return {
        "obligate": "obligate",
        "very_frequent": "very_frequent",
        "frequent": "frequent",
        "occasional": "occasional",
        "very_rare": "very_rare",
        "excluded": "excluded"
    }


def get_orpha_iri(orphacode):
    return f"http://www.orpha.net/ORDO/Orphanet_{orphacode}"


def get_frequency_distribution():
    return {  # frequency ids + associated probability
        28405: 1,  # Obligate (100%)
        28412: 0.895,  # Very frequent (99-80%)
        28419: 0.545,  # Frequent (79-30%)
        28426: 0.17,  # Occasional (29-5%)
        28433: 0.025,  # Very rare (<4-1%)
        28440: 0  # Excluded (0%)
    }


def show_distribution(distributions):
    frequency_dict = get_frequency_distribution()
    sorted_items = sorted(distributions.items(), key=lambda x: x[0])
    frequency_obs = [t[1][0] / t[1][1] if t[1][0] != 0 else t[1][0] for t in sorted_items]
    sorted_freq = sorted(frequency_dict.items(), key=lambda x: x[0])
    frequency_th = [x[1] for x in sorted_freq]

    index = np.arange(len(frequency_obs))
    bar_width = 0.35

    fix, ax = plt.subplots()
    ax.bar(index, frequency_obs, bar_width, label='Observed')
    ax.bar(index + bar_width, frequency_th, bar_width, label='Theoretical')

    ax.set_xlabel('Frequency category')
    ax.set_ylabel('Frequency')
    ax.set_title('Observed frequency VS theoretical frequency')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['100%', '99-80%', '79-30%', '29-5%', '<4-1%', '0%'])
    ax.legend()

    plt.show()


def find_xml(row, source_tag, target_tag, field, text=True, attrib="id"):
    """
    Find sub-tag of a source-tag and place its value into the dictionary of the current rows data.
    Args:
        row: Data of the current row
        source_tag: Parent Tag
        target_tag: The Sub-tag to find
        field: The field to place it as in the CSV file
        text: If true, then the value of the tag is its inner text, otherwise its attribute (attrib) is taken.
        attrib: If text is False, then the value of the attrib will be used to take the attribute.

    Returns: the tag that was found

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
    Get a list of all the header names that should be in the xml file.
    Returns: the array of header names

    """
    return HPOHeaders.values()
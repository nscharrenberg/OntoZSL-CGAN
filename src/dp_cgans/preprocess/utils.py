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
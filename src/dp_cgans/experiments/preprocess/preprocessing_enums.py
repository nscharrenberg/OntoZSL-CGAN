from enum import Enum


class HPOHeaders(Enum):
    HPO_DISORDER_SET_STATUS_TAG_ID = "hpo_disorder_set_status_tag_id"
    DISORDER_TAG_ID = "disorder_tag_id"
    ORPHA_CODE = "orpha_code"
    EXPERT_LINK = "expert_link"
    NAME = "name"
    DISORDER_TYPE_TAG_ID = "disorder_type_tag_id"
    DISORDER_TYPE_NAME = "disorder_type_name"
    DISORDER_GROUP_TAG_ID = "disorder_group_tag_id"
    DISORDER_GROUP_NAME = "disorder_group_name"
    HPO_DISORDER_ASSOCIATION_TAG_ID = "hpo_disorder_association_tag_id"
    HPO_TAG_ID = "hpo_tag_id"
    HPO_ID = "hpo_id"
    HPO_TERM = "hpo_term"
    HPO_FREQUENCY_TAG_ID = "hpo_frequency_tag_id"
    HPO_FREQUENCY_NAME = "hpo_frequency_name"
    DIAGNOSTIC_CRITERIA_TAG_ID = "diagnostic_criteria_tag_id"
    DIAGNOSTIC_CRITERIA_NAME = "diagnostic_criteria_name"
    SOURCE = "source"
    VALIDATION_STATUS = "validation_status"
    ONLINE = "online"
    VALIDATION_DATE = "validation_date"

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


class HPOTags(Enum):
    HPO_DISORDER_SET_STATUS_LIST = "HPODisorderSetStatusList"
    HPO_DISORDER_SET_STATUS = "HPODisorderSetStatus"
    DISORDER = "Disorder"
    ORPHA_CODE = "OrphaCode"
    EXPERT_LINK = "ExpertLink"
    NAME = "Name"
    DISORDER_TYPE = "DisorderType"
    DISORDER_TYPE_NAME = "Name"
    DISORDER_GROUP = "DisorderGroup"
    DISORDER_GROUP_NAME = "Name"
    SOURCE = "Source"
    ONLINE = "Online"
    VALIDATION_STATUS = "ValidationStatus"
    VALIDATION_DATE = "ValidationDate"
    HPO_DISORDER_ASSOCIATION_LIST = "HPODisorderAssociationList"
    HPO_DISORDER_ASSOCIATION = "HPODisorderAssociation"
    HPO = "HPO"
    HPO_ID = "HPOId"
    HPO_TERM = "HPOTerm"
    HPO_FREQUENCY = "HPOFrequency"
    HPO_FREQUENCY_NAME = "Name"
    DIAGNOSTIC_CRITERIA = "DiagnosticCriteria"
    DIAGNOSTIC_CRITERIA_NAME = "Name"


class Relation(Enum):
    ASSOCIATION_HAS_OBJECT = "association_has_object"
    ASSOCIATION_HAS_SUBJECT = "association_has_subject"
    HAS_FREQUENCY = "has_frequency"
    HAS_DC_ATTRIBUTE = "has_DC_attribute"
    SUB_CLASS_OF = "subClassOf"

import csv
import os
from xml.etree import ElementTree

from dp_cgans.embeddings import log
from dp_cgans.preprocess import HPOHeaders, HPOTags


def xml_to_csv(file_path: str, destination_path: str, verbose=True):
    log(text=f'🔄️  About to Convert HPO XML to CSV.', verbose=verbose)

    if not os.path.exists(file_path):
        log(text=f'❌️The file {file_path} does not exists, and can therefore not be converted', verbose=True)
        return

    log(text=f'🔄️  Reading XML from "{file_path}"...', verbose=verbose)
    tree = ElementTree.parse(file_path)
    root = tree.getroot()

    log(text=f'🔄️  Writing CSV to "{destination_path}"...', verbose=verbose)
    with open(destination_path, "w", encoding='utf-8') as f:
        writer = csv.DictWriter(f, delimiter=",", fieldnames=get_headers())
        writer.writeheader()

        for status in root.find(HPOTags.HPO_DISORDER_SET_STATUS_LIST.value).findall(
                HPOTags.HPO_DISORDER_SET_STATUS.value):
            row = {
                HPOHeaders.HPO_DISORDER_SET_STATUS_TAG_ID.value: status.attrib["id"]
            }

            disorder_tag_id = find(row, status, HPOTags.DISORDER.value, HPOHeaders.HPO_TAG_ID.value, text=False)
            orpha_code = find(row, disorder_tag_id, HPOTags.ORPHA_CODE.value, HPOHeaders.ORPHA_CODE.value, text=True)
            expert_link = find(row, disorder_tag_id, HPOTags.EXPERT_LINK.value, HPOHeaders.EXPERT_LINK.value, text=True)
            name = find(row, disorder_tag_id, HPOTags.NAME.value, HPOHeaders.NAME.value, text=True)

            disorder_type_tag_id = find(row, disorder_tag_id, HPOTags.DISORDER_TYPE.value,
                                        HPOHeaders.DISORDER_TYPE_TAG_ID.value, text=False)
            disorder_type_name = find(row, disorder_type_tag_id, HPOTags.DISORDER_TYPE_NAME.value,
                                      HPOHeaders.DISORDER_TYPE_NAME.value, text=True)

            disorder_group_tag_id = find(row, disorder_tag_id, HPOTags.DISORDER_GROUP.value,
                                         HPOHeaders.DISORDER_GROUP_TAG_ID.value, text=False)
            disorder_group_name = find(row, disorder_group_tag_id, HPOTags.DISORDER_GROUP.value,
                                       HPOHeaders.DISORDER_GROUP_NAME.value, text=True)

            source = find(row, status, HPOTags.SOURCE.value, HPOHeaders.SOURCE.value, text=True)
            online = find(row, status, HPOTags.ONLINE.value, HPOHeaders.ONLINE.value, text=True)
            validation_status = find(row, status, HPOTags.VALIDATION_STATUS.value, HPOHeaders.VALIDATION_STATUS.value,
                                     text=True)
            validation_date = find(row, status, HPOTags.VALIDATION_DATE.value, HPOHeaders.VALIDATION_DATE.value,
                                   text=True)

            for association in disorder_tag_id.find(HPOTags.HPO_DISORDER_ASSOCIATION_LIST.value).findall(
                    HPOTags.HPO_DISORDER_ASSOCIATION.value):
                row[HPOHeaders.HPO_DISORDER_ASSOCIATION_TAG_ID.value] = association.attrib["id"]

                hpo_tag_id = find(row, association, HPOTags.HPO.value, HPOHeaders.HPO_TAG_ID.value, text=False)
                hpo_id = find(row, hpo_tag_id, HPOTags.HPO_ID.value, HPOHeaders.HPO_ID.value, text=True)
                hpo_term = find(row, hpo_tag_id, HPOTags.HPO_TERM.value, HPOHeaders.HPO_TERM.value, text=True)

                hpo_frequency_tag_id = find(row, association, HPOTags.HPO_FREQUENCY.value,
                                            HPOHeaders.HPO_FREQUENCY_TAG_ID.value, text=False)
                hpo_frequency_name = find(row, hpo_frequency_tag_id, HPOTags.HPO_FREQUENCY_NAME.value,
                                          HPOHeaders.HPO_FREQUENCY_NAME.value, text=True)

                diagnostic_criteria_tag_id = find(row, association, HPOTags.DIAGNOSTIC_CRITERIA.value,
                                                  HPOHeaders.DIAGNOSTIC_CRITERIA_TAG_ID.value, text=False)
                diagnostic_criteria_name = find(row, diagnostic_criteria_tag_id, HPOTags.DIAGNOSTIC_CRITERIA_NAME.value,
                                                HPOHeaders.DIAGNOSTIC_CRITERIA_NAME.value, text=True)

                writer.writerow(row)

        log(text=f'✅️Success! File has been saved to "{destination_path}".', verbose=True)


def find(row, source_tag, target_tag, field, text=True, attrib="id"):
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

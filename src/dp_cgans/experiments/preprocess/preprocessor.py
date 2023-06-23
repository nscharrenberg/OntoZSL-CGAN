import csv
import os.path
import pickle
import urllib.request
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import rdflib
import requests
from rdflib import Namespace

from dp_cgans.experiments.preprocess import HPOTags, HPOHeaders
from dp_cgans.experiments.preprocess.csv_to_txt import read_dataset, get_sub_class_of_parents, get_other_properties, \
    get_parents, get_entities, write_to_file, get_relations
from dp_cgans.experiments.preprocess.sparql import build_dictionary, populate_rd_sets
from dp_cgans.experiments.preprocess.utils import get_headers, find_xml, get_diagnostic_criteria_entities, \
    get_frequency_association_entities, get_frequency_distribution, get_orpha_iri
from dp_cgans.utils.config import Config
from dp_cgans.utils.logging import log, LogLevel
from dp_cgans.utils.files import download, get_or_create_directory

hpo_file_url = "https://www.orphadata.com/data/xml/en_product4.xml"
hp_file_url = "https://data.bioontology.org/ontologies/HP/submissions/600/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"
ordo_file_url = "https://data.bioontology.org/ontologies/ORDO/submissions/26/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"
hoom_file_url = "https://data.bioontology.org/ontologies/HOOM/submissions/7/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"


hpo_file_name = "hpo.xml"
hp_file_name = "hp.obo"
ordo_file_name = "ordo.owl"
hoom_file_name = "hoom.owl"

default_directory_path = "data"
download_directory_path = "downloads"


class Preprocessor:
    """
    A class for preprocessing data to generate datasets for training and testing.

    The Preprocessor class performs several steps including downloading necessary files,
    converting XML to CSV format, converting CSV to TXT format, awaiting a SPARQL URL,
    creating training and test datasets, generating data, and generating dictionaries.

    Args:
        config (str or Config): A path to a configuration file or a Config object
            containing the configuration settings.

    Attributes:
        config (Config): The configuration object containing the settings for the preprocessing process.
        verbose (bool): A flag indicating whether to enable verbose logging.
        directories (dict): A dictionary containing the directory paths for downloads, conversion,
            training and test datasets, generator, and dictionaries.
        downloads (dict): A dictionary containing the download file names and URLs for the required datasets.
        sparql (str): The SPARQL URL obtained during the preprocessing process.
        dataset (dict): The created training and test datasets.
        directory (str): The base directory for the preprocessing process.
        hpo_file_url (str): The URL for downloading the "Phenotypes Associated with Rare Disorders" dataset.
        hp_file_url (str): The URL for downloading the "Human Phenotype Ontologies" dataset.
        ordo_file_url (str): The URL for downloading the "Orphanet Rare Disease Ontologies" dataset.
        hoom_file_url (str): The URL for downloading the "HPO - ORDO Ontological Modules" dataset.
        hpo_file_name (str): The name of the downloaded "Phenotypes Associated with Rare Disorders" XML file.
        hp_file_name (str): The name of the downloaded "Human Phenotype Ontologies" file.
        ordo_file_name (str): The name of the downloaded "Orphanet Rare Disease Ontologies" file.
        hoom_file_name (str): The name of the downloaded "HPO - ORDO Ontological Modules" file.
        default_directory_path (str): The default directory path used if no directory is specified in the configuration.
        download_directory_path (str): The directory path for storing downloaded files used if no download directory
            is specified in the configuration.
    """
    def __init__(self, config: str or Config):
        """
        Initialize the Preprocessor object.

        Args:
            config (str or Config): A path to a configuration file or a Config object
                containing the configuration settings.
        """
        if isinstance(config, str):
            self.config = Config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise Exception("Configuration could not be read.")

        self._init_verbose()
        self._init_directory()
        self._define_download_paths()

    def start(self):
        """
        Start the preprocessing process by executing all the necessary steps in the correct order.
        """
        log(f"Starting to Preprocess...", verbose=self.verbose)

        self._download()
        self._xml_to_csv()
        self._csv_to_txt()
        self._await_sparql_url()
        self._create_training_and_test_dataset()
        self._generate()
        self._generate_dictionaries()

    def _download(self):
        """
        Download the required datasets from specified URLs and save them to the specified download directory.
        """
        log(f"Preparing to download datasets...", verbose=self.verbose)

        directory = self.directories["downloads"]
        hpo = self.downloads["hpo"]
        hp = self.downloads["hp"]
        ordo = self.downloads["ordo"]
        hoom = self.downloads["hoom"]

        log(f"Downloading \"Phenotypes Associated with Rare Disorders\"...", level=LogLevel.INFO)
        download(url=hpo_file_url, location_path=directory, file_name=hpo, verbose=self.verbose)

        log(f"Downloading \"Orphanet Rare Disease Ontologies\"...", level=LogLevel.INFO)
        download(url=ordo_file_url, location_path=directory, file_name=ordo, verbose=self.verbose)

        log(f"Downloading \"Human Phenotype Ontologies\"...", level=LogLevel.INFO)
        download(url=hp_file_url, location_path=directory, file_name=hp, verbose=self.verbose)

        log(f"Downloading \"HPO - ORDO Ontological Modules\"...", level=LogLevel.INFO)
        download(url=hoom_file_url, location_path=directory, file_name=hoom, verbose=self.verbose)

        log(f"Datasets have been successfully downloaded to \"{directory}\"!", level=LogLevel.OK, verbose=self.verbose)

    def _xml_to_csv(self):
        """
        Convert the downloaded XML file to CSV format and save it to the download directory.
        """
        download_directory = self.directories["downloads"]
        download_hpo = self.downloads["hpo"]
        xml_file = f"{download_directory}/{download_hpo}"
        csv_hpo = download_hpo.replace('.xml', '.csv')
        new_file = f"{download_directory}/{csv_hpo}"
        self.downloads["hpo_csv"] = csv_hpo

        if not os.path.isfile(xml_file):
            log(text=f"The XML file located at \"{xml_file}\" does not exist, and can therefore not be converted.", level=LogLevel.ERROR)

        if os.path.isfile(new_file):
            log(text=f"The CSV file located at \"{new_file}\" already exist, we'll therefore not create a new one.")
            return

        tree = ElementTree.parse(xml_file)
        root = tree.getroot()

        with open(new_file, "w", encoding='utf-8') as f:
            writer = csv.DictWriter(f, delimiter=",", fieldnames=get_headers())
            writer.writeheader()

            for status in root.find(HPOTags.HPO_DISORDER_SET_STATUS_LIST.value).findall(
                    HPOTags.HPO_DISORDER_SET_STATUS.value):
                row = {
                    HPOHeaders.HPO_DISORDER_SET_STATUS_TAG_ID.value: status.attrib["id"]
                }

                disorder_tag_id = find_xml(row, status, HPOTags.DISORDER.value, HPOHeaders.HPO_TAG_ID.value, text=False)
                orpha_code = find_xml(row, disorder_tag_id, HPOTags.ORPHA_CODE.value, HPOHeaders.ORPHA_CODE.value,
                                      text=True)
                expert_link = find_xml(row, disorder_tag_id, HPOTags.EXPERT_LINK.value, HPOHeaders.EXPERT_LINK.value,
                                       text=True)
                name = find_xml(row, disorder_tag_id, HPOTags.NAME.value, HPOHeaders.NAME.value, text=True)

                disorder_type_tag_id = find_xml(row, disorder_tag_id, HPOTags.DISORDER_TYPE.value,
                                                HPOHeaders.DISORDER_TYPE_TAG_ID.value, text=False)
                disorder_type_name = find_xml(row, disorder_type_tag_id, HPOTags.DISORDER_TYPE_NAME.value,
                                              HPOHeaders.DISORDER_TYPE_NAME.value, text=True)

                disorder_group_tag_id = find_xml(row, disorder_tag_id, HPOTags.DISORDER_GROUP.value,
                                                 HPOHeaders.DISORDER_GROUP_TAG_ID.value, text=False)
                disorder_group_name = find_xml(row, disorder_group_tag_id, HPOTags.DISORDER_GROUP.value,
                                               HPOHeaders.DISORDER_GROUP_NAME.value, text=True)

                source = find_xml(row, status, HPOTags.SOURCE.value, HPOHeaders.SOURCE.value, text=True)
                online = find_xml(row, status, HPOTags.ONLINE.value, HPOHeaders.ONLINE.value, text=True)
                validation_status = find_xml(row, status, HPOTags.VALIDATION_STATUS.value,
                                             HPOHeaders.VALIDATION_STATUS.value,
                                             text=True)
                validation_date = find_xml(row, status, HPOTags.VALIDATION_DATE.value, HPOHeaders.VALIDATION_DATE.value,
                                           text=True)

                for association in disorder_tag_id.find(HPOTags.HPO_DISORDER_ASSOCIATION_LIST.value).findall(
                        HPOTags.HPO_DISORDER_ASSOCIATION.value):
                    row[HPOHeaders.HPO_DISORDER_ASSOCIATION_TAG_ID.value] = association.attrib["id"]

                    hpo_tag_id = find_xml(row, association, HPOTags.HPO.value, HPOHeaders.HPO_TAG_ID.value, text=False)
                    hpo_id = find_xml(row, hpo_tag_id, HPOTags.HPO_ID.value, HPOHeaders.HPO_ID.value, text=True)
                    hpo_term = find_xml(row, hpo_tag_id, HPOTags.HPO_TERM.value, HPOHeaders.HPO_TERM.value, text=True)

                    hpo_frequency_tag_id = find_xml(row, association, HPOTags.HPO_FREQUENCY.value,
                                                    HPOHeaders.HPO_FREQUENCY_TAG_ID.value, text=False)
                    hpo_frequency_name = find_xml(row, hpo_frequency_tag_id, HPOTags.HPO_FREQUENCY_NAME.value,
                                                  HPOHeaders.HPO_FREQUENCY_NAME.value, text=True)

                    diagnostic_criteria_tag_id = find_xml(row, association, HPOTags.DIAGNOSTIC_CRITERIA.value,
                                                          HPOHeaders.DIAGNOSTIC_CRITERIA_TAG_ID.value, text=False)
                    diagnostic_criteria_name = find_xml(row, diagnostic_criteria_tag_id,
                                                        HPOTags.DIAGNOSTIC_CRITERIA_NAME.value,
                                                        HPOHeaders.DIAGNOSTIC_CRITERIA_NAME.value, text=True)

                    writer.writerow(row)

        log(text=f"Success! XML has been converted to CSV and saved to \"{new_file}\".", level=LogLevel.OK, verbose=True)

    def _csv_to_txt(self):
        """
        Convert the downloaded CSV file to TXT format and save it to the download directory.
        """
        download_directory = self.directories["downloads"]
        hpo_csv = self.downloads["hpo_csv"]
        csv_file = f"{download_directory}/{hpo_csv}"
        hpo_txt = hpo_csv.replace('.csv', '.txt')
        new_file = f"{download_directory}/{hpo_txt}"
        self.downloads["hpo_txt"] = hpo_txt

        if not os.path.isfile(csv_file):
            log(text=f"The CSV file located at \"{csv_file}\" does not exist, and can therefore not be converted.", level=LogLevel.ERROR)

        if os.path.isfile(new_file):
            log(text=f"The TXT file located at \"{new_file}\" already exists, and can therefore not be saved.", level=LogLevel.ERROR)

        df = pd.read_csv(csv_file, dtype="object")
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

        triples = []
        triples_names = []
        entities = []
        entities_names = []
        relations = []

        sub_class_of_parents_dict = get_sub_class_of_parents(triples, triples_names, association_entitites,
                                                             diagnostic_criteria_entities,
                                                             frequency_association_entities,
                                                             hpo_entities, orpha_entities, self.verbose)
        triples = sub_class_of_parents_dict['triples']
        triples_names = sub_class_of_parents_dict['triples_names']
        entities_and_parent_class = sub_class_of_parents_dict['entities_and_parent_class']

        other_properties_dict = get_other_properties(triples, triples_names, has_object_triples, has_subject_triples,
                                                     has_frequency_triples, has_diagnostic_criteria_triples,
                                                     hpo_entities,
                                                     orpha_entities, frequency_association_entities,
                                                     diagnostic_criteria_entities, association_entitites, self.verbose)

        triples = other_properties_dict['triples']
        triples_names = other_properties_dict['triples_names']

        parents_dict = get_parents(entities, entities_names, entities_and_parent_class, self.verbose)
        entities = parents_dict['entities']
        entities_names = parents_dict['entities_names']

        entities_dict = get_entities(entities, entities_names, association_entitites, diagnostic_criteria_entities,
                                     frequency_association_entities, hpo_entities, orpha_entities, self.verbose)

        entities = entities_dict['entities']
        entities_names = entities_dict['entities_names']

        relations_dict = get_relations(relations, self.verbose)
        relations = relations_dict['relations']

        lists_and_files = [
            (triples, self.config.get_nested('convert', 'files', 'triples')),
            (triples_names, self.config.get_nested('convert', 'files', 'triples_names')),
            (entities, self.config.get_nested('convert', 'files', 'entities')),
            (entities_names, self.config.get_nested('convert', 'files', 'entities_names')),
            (relations, self.config.get_nested('convert', 'files', 'relations'))
        ]

        write_to_file(self.directories["convert"], lists_and_files, self.verbose)

        log(text=f"Success! CSV has been converted to TXT and saved to \"{new_file}\".", level=LogLevel.OK, verbose=True)

    def _await_sparql_url(self):
        """
        Await the SPARQL URL that will be used for further processing.
        """
        config_url = self.config.get_nested('sparql', 'url')

        log(f"We are at a step, where some manual actions have to be performed. \n 1. Merge the \"hp.obo\" into \"hoom.owl\". \n 2. Merge \"ordo.owl\" into the merged dataset from step 1. \n 3. Save the merged dataset to \"owl/xml\" and \"ttl\". \n 4. Setup a SparQL server (e.g. Apache Jena Fuseki) \n 5. Upload the merged dataset to the SparQL server. \n 6. Copy the query URL into the terminal (e.g. \"http://localhost:3030/hoom/query\").",
            level=LogLevel.INFO)
        is_valid = False
        sparql_url = None

        while not is_valid:
            if config_url is not None:
                sparql_url = self.config.get_nested('sparql', 'url')
                log(f"You already declared the url in your config file: \"{config_url}\", we will check this first.", level=LogLevel.INFO)
            else:
                sparql_url = input("SPARQL url: ")

            f = requests.get(sparql_url)

            if "Service Description: /merged/query" in f.text:
                is_valid = True
            elif config_url is not None:
                config_url = None

        log(f"The SPARQL url: \"{config_url}\" is correct and will be used.", level=LogLevel.OK)

        self.sparql = sparql_url

    def _create_training_and_test_dataset(self):
        """
        Create the training and test datasets based on the SPARQL data obtained from the previous steps.
        """
        directory = self.directories["train_test"]
        seen_name = self.config.get_nested('train_test', 'files', 'seen')
        unseen_name = self.config.get_nested('train_test', 'files', 'unseen')
        seen_file = f"{directory}/{seen_name}"
        unseen_file = f"{directory}/{unseen_name}"

        if os.path.isfile(seen_file):
            log(text=f"The seen file located at \"{seen_file}\" already exists, and can therefore not be created.", level=LogLevel.ERROR)

        if os.path.isfile(unseen_file):
            log(text=f"The unseen file located at \"{unseen_file}\" already exists, and can therefore not be created.", level=LogLevel.ERROR)

        graph = rdflib.Graph(store="SPARQLStore")
        graph.open(self.sparql)
        graph.bind("HOOM", Namespace("http://www.semanticweb.org/ontology/HOOM"))
        graph.bind("rdf", Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"))
        graph.bind("owl", Namespace("ttp://www.w3.org/2002/07/owl#"))

        rd_group_dictionary, rd_set, rd_list = build_dictionary(graph)

        seen = self.config.get_nested('train_test', 'populate', 'seen')
        unseen = self.config.get_nested('train_test', 'populate', 'unseen')
        method = self.config.get_nested('train_test', 'populate', 'method')

        seen_rds, unseen_rds = populate_rd_sets(rd_set=rd_set, rd_list=rd_list, rd_groups_dict=rd_group_dictionary,
                                                method=method, selected_seen_rds=seen,
                                                selected_unseen_rds=unseen)

        self.dataset = {
            "seen": seen_rds,
            "unseen": unseen_rds
        }

        with open(seen_file, 'wb') as f:
            pickle.dump(seen_rds, f)

        with open(unseen_file, 'wb') as f:
            pickle.dump(unseen_rds, f)

        log(text=f"Success! The Seen and Unseen Datasets have been created.", level=LogLevel.OK, verbose=True)

    def _generate(self):
        """
        Generate data for training and testing using the created datasets.
        """
        directory = self.directories["train_test"]
        seen_name = self.config.get_nested('train_test', 'files', 'seen')
        unseen_name = self.config.get_nested('train_test', 'files', 'unseen')
        seen_file = f"{directory}/{seen_name}"
        unseen_file = f"{directory}/{unseen_name}"

        if not os.path.isfile(seen_file):
            log(text=f"The seen file located at \"{seen_file}\" does not exist, and can therefore not be processed.", level=LogLevel.ERROR)

        if not os.path.isfile(unseen_file):
            log(text=f"The unseen file located at \"{unseen_file}\" does not exist, and can therefore not be processed.", level=LogLevel.ERROR)

        save_directory = self.directories["generator"]
        save_seen_csv = self.config.get_nested('generator','files', 'seen')
        save_unseen_csv = self.config.get_nested('generator', 'files', 'unseen')
        save_seen_path = f"{save_directory}/{save_seen_csv}"
        save_unseen_path = f"{save_directory}/{save_unseen_csv}"

        if os.path.isfile(save_seen_path):
            log(text=f"The seen patient dataset located at \"{save_seen_path}\" already exists, and can therefore not be saved.", level=LogLevel.ERROR)

        if os.path.isfile(save_unseen_path):
            log(text=f"The unseen patient dataset located at \"{save_unseen_path}\" already exists, and can therefore not be saved.", level=LogLevel.ERROR)

        download_directory = self.directories["downloads"]
        hpo_csv = self.downloads["hpo_csv"]
        csv_file = f"{download_directory}/{hpo_csv}"

        if not os.path.isfile(csv_file):
            log(text=f"The hpo file located at \"{csv_file}\" does not exist, and can therefore not be processed.", level=LogLevel.ERROR)

        df = pd.read_csv(csv_file)
        df['Orpha_URI'] = "http://www.orpha.net/ORDO/Orphanet_" + df[HPOHeaders.ORPHA_CODE.value].astype(str)

        with open(seen_file, 'rb') as f:
            seen_rds_set = pickle.load(f)

        with open(unseen_file, 'rb') as f:
            unseen_rds_set = pickle.load(f)

        df = df.loc[df['Orpha_URI'].isin(seen_rds_set.union(unseen_rds_set))]

        gen_small_file = self.config.get_nested('generator', 'small_file')

        if gen_small_file:
            log(f"Minify Data to ensure smaller file sizes...", verbose=self.verbose)
            grouped = df.groupby(HPOHeaders.NAME.value, sort=False)
            df = pd.concat([group for name, group in grouped][:10])

        grouped = df.groupby(HPOHeaders.NAME.value, sort=False)

        rd_count = grouped.ngroups
        unique_hps = df[HPOHeaders.HPO_TERM.value].unique()
        total_hp_count = len(unique_hps)

        log(f"There are {rd_count} unique Rare Diseases.", level=LogLevel.INFO, verbose=self.verbose)
        log(f"There are {total_hp_count} unique Phenotypes.", level=LogLevel.INFO, verbose=self.verbose)

        phenotypes = unique_hps.tolist()
        phenotypes_dict = {
            hp: i for i, hp in enumerate(phenotypes)
        }

        header = [['rare_disease'] + phenotypes]
        seen_patients_data = []
        unseen_patients_data = []

        frequency_dict = get_frequency_distribution()

        distribution_check = {  # value: count of patients with hp + maximum count of patients that could've had the hp
            28405: [0, 0],  # Obligate (100%)
            28412: [0, 0],  # Very frequent (99-80%)
            28419: [0, 0],  # Frequent (79-30%)
            28426: [0, 0],  # Occasional (29-5%)
            28433: [0, 0],  # Very rare (<4-1%)
            28440: [0, 0]  # Excluded (0%)
        }

        patients_count = 0

        use_ontology_rds = self.config.get_nested('generator', 'use_ontology')
        patients_per_rd = self.config.get_nested('generator', 'patients_per_rd')

        if not use_ontology_rds:
            vectorize = np.vectorize(get_orpha_iri)
            seen_rds = vectorize(df[HPOHeaders.ORPHA_CODE.value].unique())
            unseen_rds = []

        for group_nb, (name, group) in enumerate(grouped):
            hp_count = len(group)

            for patient in range(patients_count, patients_count + patients_per_rd):
                temp_hp = []

                proba_results = np.random.rand(hp_count)
                rd_n = ""
                rd_iri = ""

                for i, (orpha_code, rd_name, hp_name, frequency_id) in enumerate(zip(
                        group[HPOHeaders.ORPHA_CODE.value],
                        group[HPOHeaders.NAME.value],
                        group[HPOHeaders.HPO_TERM.value],
                        group[HPOHeaders.HPO_FREQUENCY_TAG_ID.value]
                )):
                    distribution_check.get(frequency_id)[1] += 1

                    if rd_n == "":
                        rd_n = rd_name
                        rd_iri = get_orpha_iri(orpha_code)

                    if proba_results[i] >= 1 - frequency_dict[frequency_id]:
                        temp_hp.append(hp_name)
                        distribution_check.get(frequency_id)[0] += 1

                if len(temp_hp) > 0:
                    row = np.zeros((total_hp_count,), dtype=int)

                    for hp in temp_hp:
                        row[phenotypes_dict.get(hp)] = 1

                    if use_ontology_rds:
                        if rd_iri in seen_rds_set:
                            seen_patients_data.append(np.concatenate([[rd_n], row]))
                        elif rd_iri in unseen_rds_set:
                            unseen_patients_data.append(np.concatenate([[rd_n], row]))
                        else:
                            log(f"The Rare Disease '{rd_n}' is unknown", verbose=self.verbose)
                            break
                    else:
                        if rd_iri in seen_rds:
                            seen_patients_data.append(np.concatenate([[rd_n], row]))
                        elif rd_iri in unseen_rds:
                            unseen_patients_data.append(np.concatenate([[rd_n], row]))
                        else:
                            log(f"The Rare Disease '{rd_n}' is unknown", verbose=self.verbose)
                            break

                    print_every = self.config.get_nested('generator', 'print_every')
                    if print_every > 0:
                        if patients_count % print_every == 0:
                            log(f"{patients_count} / {patients_per_rd * rd_count} patients generated", level=LogLevel.INFO)

                    patients_count += 1

        unseen_index = len(seen_patients_data) + 1
        full_data = np.array(header + seen_patients_data + unseen_patients_data)

        del_col_th = self.config.get_nested('generator', 'del_col_th')

        if del_col_th > 0:
            col_sum = np.sum(full_data[1:, 1:].astype(int), axis=0)
            mask = [True] + [s >= del_col_th for s in col_sum]
            full_data = np.transpose(np.transpose(full_data)[mask])

        seen_patients_data = full_data[:unseen_index]
        headers = seen_patients_data[0]

        seen_patients_data = pd.DataFrame(seen_patients_data[1:], columns=headers)
        seen_patients_data = seen_patients_data.loc[:, (seen_patients_data != "0").any(axis=0)]

        sort = self.config.get_nested('generator', 'sort')

        if sort:
            seen_patients_data = seen_patients_data.sort_values(by=['rare_disease'])

        unseen_patients_data = full_data[unseen_index:]
        unseen_unique = np.unique(unseen_patients_data[:, :1])
        unseen_patients_data = pd.DataFrame(unseen_patients_data, columns=headers)
        unseen_patients_data = unseen_patients_data[seen_patients_data.columns]

        if sort:
            unseen_patients_data = unseen_patients_data.sort_values(by=['rare_disease'])

        max_columns = self.config.get_nested('generator', 'max_columns')

        if max_columns is not None and max_columns > 0:
            max_columns = max_columns + 1
            total_data = pd.concat([seen_patients_data, unseen_patients_data])
            total_data = total_data.astype({col: 'int32' for col in total_data.columns if col != 'rare_disease'})
            total_data['rare_disease'] = total_data['rare_disease'].astype(str)
            columns_to_remove = len(total_data.columns) - max_columns
            if columns_to_remove > 0:
                total_data = Preprocessor.remove_n_lowest_from_df(total_data, columns_to_remove)

                seen_patients_data = seen_patients_data[total_data.columns]
                unseen_patients_data = unseen_patients_data[total_data.columns]

        log(f"{len(seen_patients_data)} seen patients generated ({len(seen_patients_data.columns)} columns).", level=LogLevel.INFO, verbose=self.verbose)
        log(f"{len(unseen_patients_data)} unseen patients generated ({len(unseen_patients_data.columns)} columns).", level=LogLevel.INFO,
            verbose=self.verbose)

        seen_patients_data.to_csv(
            save_seen_path,
            encoding="utf-8",
            index=False,
            header=True
        )

        if len(unseen_patients_data) > 0:
            unseen_patients_data.to_csv(
                save_unseen_path,
                encoding="utf-8",
                index=False,
                header=True
            )

            with open(os.path.join(directory, ("small_" if gen_small_file else "") + "unseen_rare_diseases.txt"),
                      "w") as unseen_file:
                for rd in unseen_unique:
                    unseen_file.write(f"{rd}\n")

        log(f"Total Rare Diseases: {rd_count}", level=LogLevel.INFO, verbose=self.verbose)
        log(f"{rd_count - len(unseen_unique)} seen Rare Diseases", level=LogLevel.INFO, verbose=self.verbose)
        log(f"{len(unseen_unique)} unseen Rare Diseases", level=LogLevel.INFO, verbose=self.verbose)

    def _generate_dictionaries(self):
        """
        Generate dictionaries for mapping between entities and their corresponding indices.
        """
        download_directory = self.directories["downloads"]
        hpo_csv = self.downloads["hpo_csv"]
        csv_file = f"{download_directory}/{hpo_csv}"

        if not os.path.isfile(csv_file):
            log(text=f"The CSV file located at \"{csv_file}\" does not exist, and can therefore not be converted.", level=LogLevel.ERROR)

        dict_directory = self.directories["dictionaries"]
        dict_hpo_name = self.config.get_nested('dictionaries', 'files', 'hpo')
        dict_ordo_name = self.config.get_nested('dictionaries', 'files', 'ordo')
        dict_hpo_file = f"{dict_directory}/{dict_hpo_name}"
        dict_ordo_file = f"{dict_directory}/{dict_ordo_name}"

        if os.path.isfile(dict_hpo_file):
            log(text=f"The HPO dictionary file located at \"{dict_hpo_file}\" already exists, and can therefore not be saved.", level=LogLevel.ERROR)

        if os.path.isfile(dict_ordo_file):
            log(text=f"The ORDO dictionary file located at \"{dict_ordo_file}\" already exists, and can therefore not be saved.", level=LogLevel.ERROR)

        df = pd.read_csv(csv_file)
        df_hp = df[[HPOHeaders.HPO_TERM.value, HPOHeaders.HPO_ID.value]].drop_duplicates()
        df_rd = df[[HPOHeaders.NAME.value, HPOHeaders.ORPHA_CODE.value]].drop_duplicates()
        df_hp[HPOHeaders.HPO_ID.value] = "http://purl.obolibrary.org/obo/" + df_hp[HPOHeaders.HPO_ID.value].str.replace(
            ":",
            "_")
        df_rd[HPOHeaders.ORPHA_CODE.value] = "http://www.orpha.net/ORDO/Orphanet_" + df_rd[
            HPOHeaders.ORPHA_CODE.value].astype(str)

        df_hp.to_csv(dict_hpo_file, sep=';', encoding='utf-8', index=False, header=False)
        df_rd.to_csv(dict_ordo_file, sep=';', encoding='utf-8', index=False, header=False)

    def _init_directory(self):
        """
        Initialize the specified directory for storing downloaded files and processed data.

        Args:
            directory (str): The directory path.
        """
        base = get_or_create_directory(self.config.get_nested('dp_cgans', 'files', 'directory'))
        self.directory = base

        if self.config.get_nested('downloads', 'use_root'):
            downloads = get_or_create_directory(f"{self.directory}/{self.config.get_nested('downloads', 'directory')}")
        else:
            downloads = get_or_create_directory(f"{self.config.get_nested('downloads', 'directory')}")

        convert = get_or_create_directory(f"{self.directory}/{self.config.get_nested('convert', 'files', 'directory')}")
        train_test = get_or_create_directory(f"{self.directory}/{self.config.get_nested('train_test', 'files', 'directory')}")
        generator = get_or_create_directory(f"{self.directory}/{self.config.get_nested('generator', 'files', 'directory')}")
        dictionaries = get_or_create_directory(f"{self.directory}/{self.config.get_nested('dictionaries', 'files', 'directory')}")

        self.directories = {
            "base": base,
            "downloads": downloads,
            "convert": convert,
            "train_test": train_test,
            "generator": generator,
            "dictionaries": dictionaries
        }

    def _init_verbose(self):
        """
        Initialize the verbosity level of the preprocessor.

        Args:
            verbose (bool): A flag indicating whether to enable verbose mode.
        """
        self.verbose = self.config.get('verbose')

        if self.verbose is None or not isinstance(self.verbose, bool):
            self.verbose = False

    @staticmethod
    def remove_n_lowest_from_df(df: pd.DataFrame, max_cols: int):
        """
        Remove the n lowest values from the specified column of the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to modify.
            column (str): The column name to remove values from.
            n (int): The number of lowest values to remove.

        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        if max_cols < 0:
            return df

        numeric_columns = df.select_dtypes(include=['number']).columns
        column_sums = df[numeric_columns].sum()
        lowest_sum_incides = column_sums.nsmallest(max_cols).index

        return df.drop(columns=lowest_sum_incides)

    def _define_download_paths(self):
        """
        Define the paths for downloading and saving the required files.

        Args:
            download_dir (str): The directory path for downloading files.
        """
        hpo = self.config.get_nested('downloads', 'hpo')
        if hpo is None:
            hpo = hpo_file_name

        hp = self.config.get_nested('downloads', 'hp')
        if hp is None:
            hp = hp_file_name

        ordo = self.config.get_nested('downloads', 'ordo')
        if ordo is None:
            ordo = ordo_file_name

        hoom = self.config.get_nested('downloads', 'hoom')
        if hoom is None:
            hoom = hoom_file_name

        self.downloads = {
            "hpo": hpo,
            "hp": hp,
            "ordo": ordo,
            "hoom": hoom
        }

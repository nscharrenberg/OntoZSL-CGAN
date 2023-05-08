import requests
import os

from dp_cgans.embeddings import log

hpo_xml = 'https://www.orphadata.com/data/xml/en_product4.xml'
ordo_owl = 'https://data.bioontology.org/ontologies/ORDO/submissions/26/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb'
hp_obo = 'https://data.bioontology.org/ontologies/HP/submissions/600/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb'
hoom_owl = 'https://data.bioontology.org/ontologies/HOOM/submissions/7/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb'


def download_datasets(location_path: str, verbose: bool = True):
    log(text=f'🔄️  About to download and save all default datasets.', verbose=verbose)

    log(text=f'🔄️  Trying to download "Phenotypes Associated with Rare Disorders"...', verbose=verbose)
    download(hpo_xml, location_path, file_name="hpo.xml")

    log(text=f'🔄️  Trying to download "Orphanet Rare Disease Ontology"...', verbose=verbose)
    download(ordo_owl, location_path, file_name="ordo.owl")

    log(text=f'🔄️  Trying to download "Human Phenotype Ontology"...', verbose=verbose)
    download(hp_obo, location_path, file_name="hp.obo")

    log(text=f'🔄️  Trying to download "HPO - ORDO Ontological Module"...', verbose=verbose)
    download(hoom_owl, location_path, file_name="hoom.owl")

    log(text=f'✅️Success! All default datasets have been downloaded and saved to "{location_path}".', verbose=verbose)


def download(url: str, location_path: str, file_name: str = None, verbose: bool = True):
    log(text=f'🔄️  Checking if destination path "{location_path}" exists...', verbose=verbose)
    if not os.path.exists(location_path):
        log(text=f'🗂️  Destination path does not exist, creating this path...', verbose=verbose)
        os.makedirs(location_path)

    if file_name is None:
        log(text=f'🗂️  No file name given, using the url filename instead!', verbose=verbose)
        file_name = url.split('/')[-1].replace(" ", "_")

    file_path = os.path.join(location_path, file_name)

    if os.path.exists(file_path):
        log(text=f'❌️The file {file_path} already exists. Make sure this does not exist!',
            verbose=True)
        return

    log(text=f'🔄️  Downloading file from {url}...', verbose=verbose)
    req = requests.get(url, stream=True)

    if req.ok:
        with open(file_path, 'wb') as f:
            log(text=f'💾️  Saving downloaded file to {file_path}...', verbose=verbose)
            for chunk in req.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
            log(text=f'✅️Success! File has been saved to {file_path}.', verbose=True)
    else:
        log(text=f'❌️Failed to download file from {url}. Status code: {req.status_code} with message: {req.text}',
            verbose=True)



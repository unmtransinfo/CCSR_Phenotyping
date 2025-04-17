import pandas as pd
import numpy as np
import argparse
import yaml
import pickle
import gzip

def generate_concept_code_id_dict(ifile=None):
    """
    read the data from the concept table and crate a dictionary with concept_code as key and concept_id as value
    """
    print("read concept table data and create dictionary")
    code_id_dict = {}
    df = pd.read_csv(ifile, sep="\t", compression='gzip', low_memory=False)
    concept_code = df['concept_code'].to_numpy()
    concept_name = df['concept_name'].to_numpy()
    concept_id = df['concept_id'].to_numpy()
    domain_id = df['domain_id'].to_numpy()
    vocabulary_id = df['vocabulary_id'].to_numpy()

    # create dictionary
    for j, c in enumerate(concept_code):
        if vocabulary_id[j] == 'ICD10CM':
            code_id_dict[c.replace('.', '')] = [concept_id[j], domain_id[j], vocabulary_id[j], concept_name[j]]

    return code_id_dict


def generate_ccsr_icd_mapping(concept_code_id_dict, ifile=None):
    """
    read xls file and select CCSR coded and ICD codes
    """
    print("read XLS file and create a dictionary")
    # read CCSR xls file
    df = pd.read_excel(ifile, sheet_name="DX_to_CCSR_Mapping")
    ccsr_codes, icd_codes = df['CCSR Category'].to_numpy(), df['ICD-10-CM Code'].to_numpy()

    # create dictionary with CCSR as keys and ICD as values
    not_found_count = 0
    ccsr_icd_mapping = {}
    icd_ccsr_mapping = {}
    for j, ccsr in enumerate(ccsr_codes):
        # CCSR -> ICD10 mapping and vice versa; exclude ICD10CM measurement and procedure
        if icd_codes[j] in concept_code_id_dict:
            if concept_code_id_dict[icd_codes[j]][1] not in ['Measurement', 'Procedure']:
                ccsr_icd_mapping.setdefault(ccsr, set()).add(concept_code_id_dict[icd_codes[j]][0])
                icd_ccsr_mapping.setdefault(concept_code_id_dict[icd_codes[j]][0], set()).add(ccsr)
            else:
                print(f"CCSR: {ccsr} is from domain: {concept_code_id_dict[icd_codes[j]][1]}")
        else:
            not_found_count+=1

    print(f"{not_found_count} ICD10CM codes from XLS file not present in Concept table")
    return ccsr_icd_mapping, icd_ccsr_mapping

def save_mapping_dict(ccsr_icd_mapping, icd_ccsr_mapping, ofile1=None, ofile2=None):
    """
    save ccsr->icd and icd->ccsr mapping dictionaries
    """
    print("save dictionary - CCSR and ICD codes")
    with open(ofile1, 'wb') as f:
        pickle.dump(ccsr_icd_mapping, f, protocol=5)

    with open(ofile2, 'wb') as f:
        pickle.dump(icd_ccsr_mapping, f, protocol=5)

def main():
    """
    the program creates a mapping between CCSR code and the corresponding ICD codes.
    """
    # load IO filenames from the YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # get data from the concept table
    concept_code_id_dict = generate_concept_code_id_dict(ifile=iodata['input_files']['concepts_file'])

    # get data from the input file
    ccsr_icd_mapping, icd_ccsr_mapping = generate_ccsr_icd_mapping(concept_code_id_dict, ifile=iodata['input_files']['ccsr_inpfile'])
    print(f"elements in ccsr_icd_mapping dictionary: {len(ccsr_icd_mapping)}, "
          f"elements in icd_ccsr_mapping dictionary: {len(icd_ccsr_mapping)}")

    # save dictionaries in pickle files
    save_mapping_dict(ccsr_icd_mapping, icd_ccsr_mapping, ofile1=iodata['ccsr_icd_mapping_file'], ofile2=iodata['icd_ccsr_mapping_file'])


if __name__ == "__main__":
    main()
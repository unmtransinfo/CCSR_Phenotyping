import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
import yaml
import argparse
import gzip
from scipy.sparse import csr_matrix, lil_matrix
from copy import deepcopy


def fetch_concept_id_name(ifile=None, afile=None):
    """
    Read concept file and return a dictionary with concept_id as key, and concept_name and vocabulary as value. Also, update the key in the ancestor dictionary
    """
    print("\ncreating dictionary using concept id and name")
    id_name_dict = {}

    # open ancestor dict file
    with open(afile, 'rb') as f:
        concept_ancestor_dict = pickle.load(f)


    # read the input file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 2500000 == 0:
                print("{0} concept records processed".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                cid_idx = pos_idx["concept_id"]
                cname_idx = pos_idx["concept_name"]
                vocab_idx = pos_idx["vocabulary_id"]
                ccode_idx = pos_idx["concept_code"]
            else:
                cid, cname, cvocab, ccode = int(vals[cid_idx]), vals[cname_idx], vals[vocab_idx], vals[ccode_idx]
                id_name_dict[cid] = [cvocab, cname]

                # replace old key with new key in ancestor dictionary
                if ccode in concept_ancestor_dict:
                    concept_ancestor_dict[cid] = concept_ancestor_dict.pop(ccode)

    print("{0} concept records processed".format(line_count))

    return id_name_dict, concept_ancestor_dict

def select_phenotype_icd10_codes(phenotype, iodata):
    """
    select list of ICD10 codes for the selected phenotype
    """
    all_phenotype_codes = []
    if phenotype == 'PTSD':
        # phenotype_df = pd.read_csv(iodata['input_files']['ptsd_codes_file'], sep="\t", header=0, compression='gzip')
        # all_phenotype_codes = set(list(phenotype_df['concept_id']))
        all_phenotype_codes = iodata['icd10_ptsd_codes']
    elif phenotype == 'MDD':
        all_phenotype_codes = iodata['icd10_mdd_codes']
    elif phenotype == 'BD':
        all_phenotype_codes = iodata['icd10_bd_codes']
    elif phenotype == 'Schizo':
        all_phenotype_codes = iodata['icd10_sch_codes']
    elif phenotype == 'SD':
        all_phenotype_codes = iodata['icd10_sd_codes']
    return all_phenotype_codes


def include_ancestors(covars, concept_ancestors, phenotypes):
    """
    include all ICD10 ancestors to the list of covariates. do not include the ancestors of phenotype codes
    """
    covars_set = set()
    print("number of condition codes before including ancestors: ", len(covars))
    for cov in covars:
        covars_set.add(cov)
        if cov not in phenotypes and cov in concept_ancestors:
            covars_set.update(concept_ancestors[cov])
    print("number of condition codes after including ancestors: ", len(covars_set))
    return covars_set


def include_ttl_ttf(covars, conditions):
    """
    include TTF and TTL in the list of covars
    """
    print("number of condition codes before including TTL and TTF: ", len(covars))
    covars_list = []
    for cov in covars:
        covars_list.append(cov)
        if cov in conditions:
            covars_list.append(str(cov)+"_TTF")
            covars_list.append(str(cov)+"_TTL")
    print("number of condition codes after including TTL and TTF: ", len(covars))
    return covars_list

def determine_label(covar, phenotype, phenotype_codes, iodata, condition_covars, time_window):
    """
    determine label for the given record using hierarchy
    if MDD is the phenotype, check if BD or Schizo is present in the selected time window. If yes, label will be 0
    if BD is the phenotype, check if Schizo is present in the selected time window. If yes, label will be 0
    """
    if phenotype == 'MDD':
        UPPER_CODE_FOUND = False
        codes_to_check = set(iodata['icd10_bd_codes'].extend(iodata['icd10_sch_codes']))
        for cc in codes_to_check:
            ttc = [d for d in condition_covars[cc] if d <= time_window]
            if len(ttc) > 0:    # BD or Schizo happened during the time window
                UPPER_CODE_FOUND = True
                break
        if UPPER_CODE_FOUND:
            return False
        else:
            return True
    elif phenotype == 'BD':
        UPPER_CODE_FOUND = False
        codes_to_check = iodata['icd10_sch_codes']
        for cc in codes_to_check:
            ttc = [d for d in condition_covars[cc] if d <= time_window]
            if len(ttc) > 0:    # Schizo happened during the time window
                UPPER_CODE_FOUND = True
                break
        if UPPER_CODE_FOUND:
            return False
        else:
            return True
    elif covar in phenotype_codes:
        return True
    else:
        return False


def generate_feature_and_label(person_conditions, person_drugs, phenotype_codes, iodata, phenotype, time_window=None, use_ancestor=None,
                               concept_ancestors=None, icd9_codes=None):
    """
    generate features in CSR format for ML
    """
    print("\ngenerate data for ML models")
    # unique persons
    all_persons = set(list(person_conditions.keys()))
    all_persons.update(list(person_drugs.keys()))
    all_persons = list(all_persons)

    # unique covariates
    all_covars = {v for inner_dict in person_conditions.values() for v in inner_dict}   # all conditions
    cond_covars = deepcopy(all_covars)  # keep a copy to add TTL and TTF
    if use_ancestor:
        all_covars = include_ancestors(all_covars, concept_ancestors, phenotype_codes)
    all_covars.update(v for inner_dict in person_drugs.values() for v in inner_dict)    # all drugs
    all_covars = list(all_covars)
    all_covars = include_ttl_ttf(all_covars, cond_covars)
    print("total unique persons and unique covars: ", len(all_persons), len(all_covars))
    all_covars_idx = dict(zip(all_covars, range(len(all_covars))))  # get index of each covariates in the list

    # generate CSR matrix and labels
    labels = []
    icd9_coded_persons = set()
    feature_matrix = lil_matrix((len(all_persons), len(all_covars)), dtype=np.uint16)
    for i, pid in enumerate(all_persons):
        LABELED_PID = False

        # condition covariates
        if pid in person_conditions:
            condition_covars = person_conditions[pid]
            for covar, vals in condition_covars.items():
                # if covariate is the selected phenotype, keep it 0 in the feature matrix. Else set the corresponding cell to 1
                if covar not in phenotype_codes:
                    j = all_covars_idx[covar]
                    feature_matrix[i, j] = 1
                    # set ancestors to 1
                    for anc_covar in concept_ancestors[covar]:
                        j = all_covars_idx[anc_covar]
                        feature_matrix[i, j] = 1

                    # find days for TTF and TTL
                    ttc = [d for d in vals if d<=time_window]
                    if len(ttc) == 0:
                        feature_matrix[i, j+1] = 1  # default TTF
                        feature_matrix[i, j+2] = 1  # default TTL
                    elif len(ttc) == 1:
                        feature_matrix[i, j+1] = 1  # default TTF
                        feature_matrix[i, j+2] = ttc[0]
                    else:
                        feature_matrix[i, j+1] = min(ttc)
                        feature_matrix[i, j+2] = max(ttc)

                # check for ICD9 phenotype codes
                if covar in icd9_codes: 
                    icd9_coded_persons.add(pid)
                
                # determine label for the record
                LABELED_PID = determine_label(covar, phenotype, phenotype_codes, iodata, condition_covars, time_window)

        # drug covariates
        if pid in person_drugs:
            drug_covars = person_drugs[pid]
            for covar, vals in drug_covars.items():
                total_exposure_count = 0
                j = all_covars_idx[covar]
                for d_days, d_count in vals:
                    if d_days <= time_window:
                        total_exposure_count+=d_count
                # set the corresponding cell value
                feature_matrix[i, j] = 65535 if total_exposure_count > 65535 else total_exposure_count

        # set label
        if LABELED_PID:
            labels.append(1)
        else:
            labels.append(0)

        # keep track
        if (i + 1) % 100000 == 0:
            print("CSR matrix generated for {0} persons".format(i + 1))
    print("CSR matrix generated for {0} persons".format(i + 1))
    print("persons coded with ICD9 code: ", len(icd9_coded_persons))

    return csr_matrix(feature_matrix), np.asarray(labels), np.asarray(all_persons), np.asarray(all_covars)



def save_data_labels(X, y, persons, covariates, features_file=None, labels_file=None, persons_file=None, covars_file=None):
    """
    save data for future use as processing takes hours
    """
    # features
    with open(features_file, "wb") as fh:
        pickle.dump(X, fh, protocol=5)

    # labels
    with open(labels_file, "wb") as fh:
        pickle.dump(y, fh, protocol=5)

    # persons
    with open(persons_file, "wb") as fh:
        pickle.dump(persons, fh, protocol=5)

    # covariates
    with open(covars_file, "wb") as fh:
        pickle.dump(covariates, fh, protocol=5)


def str_to_bool(s):
    """
    convert command line True/False to boolean as argparse considers them string.
    """
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        print("invalid boolean value provided")


def main():
    """
    This program reads the SQL output files and generate feature matrix and labels for the ML models. The feature matrix and labels are saved as
    pickle files.
    """
    # load IO filenames from the YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
    parser.add_argument("-useAncestors", type=str_to_bool, choices=[True, False], default=True, help="use ancestor terms as covariates?")
    parser.add_argument("-time_window_length", type=int, default=365, help="choose time window length")
    parser.add_argument("-phenotype", default='PTSD', choices=['PTSD', 'MDD', 'BD', 'Schizo', 'SD'], help="select a phenotype")
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # STEP 1: read concept file and create a dictionary with concept_id as key and vocab_id and concept_name as values. Also, update key in ancestor file
    concept_id_name_dict, descendant_ancestor_dict = fetch_concept_id_name(ifile=iodata['input_files']['concepts_file'],
                                                                         afile=iodata['output_files']['ICD10_concept_ancestors_file'])
    print("number of concepts: ", len(concept_id_name_dict))

    # STEP 2: fetch person conditions
    if not os.path.exists(iodata['output_files']['person_cond_file']):
        print("run fetch_covariates.py to select condition data")
        exit(-1)
    else:
        print("load condition data from pickle file")
        with open(iodata['output_files']['person_cond_file'], 'rb') as f:
            person_conditions_dict = pickle.load(f)
    print("keys in person_conditions_dict: ", len(person_conditions_dict))

    # STEP 3: fetch person drugs
    if not os.path.exists(iodata['output_files']['person_drug_file']):
        print("run fetch_covariates.py to select drug data")
        exit(-1)
    else:
        print("load drugs data from pickle file")
        with open(iodata['output_files']['person_drug_file'], 'rb') as f:
            person_drugs_dict = pickle.load(f)
    print("keys in person_drugs_dict: ", len(person_drugs_dict))

    # STEP 4: select codes for the given phenotype
    phenotype_codes = select_phenotype_icd10_codes(p_args.phenotype, iodata)

    # STEP 5: generate ML data using conditions and drugs
    X_all, y_all, persons_all, covariates_all = generate_feature_and_label(person_conditions_dict, person_drugs_dict, phenotype_codes, iodata,
                                                                           p_args.phenotype, time_window=p_args.time_window_length,
                                                                           use_ancestor=p_args.useAncestors,
                                                                           concept_ancestors=descendant_ancestor_dict,
                                                                           icd9_codes=iodata['icd9_ptsd_codes'])
    print("len(persons_all), len(y_all), len(covariates_all), sum(y_all): ", len(persons_all), len(y_all), len(covariates_all), np.sum(y_all))

    # save data, labels, covariates and person_ids
    save_data_labels(X_all, y_all, persons_all, covariates_all, features_file=iodata['output_files']['features_file'],
                     labels_file=iodata['output_files']['labels_file'], persons_file=iodata['output_files']['persons_file'],
                     covars_file=iodata['output_files']['covars_file'])



if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
import yaml
import argparse
import gzip
from scipy.sparse import csr_matrix, hstack


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


def fetch_observation_dates(ifile=None, ofile=None):
    """
    for each patient, select observation start and end date
    """
    print("\nselect observation start and end dates for all patients")
    person_first_last_obs_dict = {}

    # read large file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 5000000 == 0:
                print("{0} records processed to determine observation dates".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                pid_idx = pos_idx["person_id"]
                s_obs_date_idx = pos_idx["observation_period_start_date"]
                e_obs_date_idx = pos_idx["observation_period_end_date"]
            else:
                # get start and end observation dates for patient
                pid, p_obs_start_date, p_obs_end_date = int(vals[pid_idx]), vals[s_obs_date_idx], vals[e_obs_date_idx]
                person_first_last_obs_dict[pid] = [p_obs_start_date, p_obs_end_date]

    print("{0} records processed to determine observation dates".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(person_first_last_obs_dict, f, protocol=5)

    return person_first_last_obs_dict


def get_person_details(p_obs_dates_dict, ifile=None, ofile=None):
    """
    fetch all selected persons' data (age, sex, state) from the input file
    """
    print("\nfetch person gender, age and location from input file")
    person_data_dict = {}
    loc_id_state = {'1': 'AK', '2': 'AL', '3': 'AR', '4': 'AZ', '5': 'CA', '6': 'CO', '7': 'CT', '8': 'DC', '9': 'DE', '10': 'FL', '11': 'GA',
                    '12': 'HI', '13': 'IA', '14': 'ID', '15': 'IL', '16': 'IN', '17': 'MI', '18': 'KS', '19': 'KY', '20': 'LA', '21': 'MA',
                    '22': 'MD', '23': 'ME', '24': 'MN', '25': 'MO', '26': 'MS', '27': 'MT', '28': 'NC', '29': 'ND', '30': 'NE', '31': 'NH',
                    '32': 'NJ', '33': 'NM', '34': 'NV', '35': 'NY', '36': 'OH', '37': 'OK', '38': 'OR', '39': 'PA', '40': 'PR', '41': 'RI',
                    '42': 'SC', '43': 'SD', '44': 'TN', '45': 'TX', '46': 'UN', '47': 'UN', '48': 'UN', '49': 'UN', '50': 'UN', '51': 'UN',
                    '52': 'UN', '53': 'UT', '54': 'VA', '55': 'VT', '56': 'WA', '57': 'WI', '58': 'WV', '59': 'WY'}

    # read the file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 5000000 == 0:
                print("{0} persons details were fetched".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                pid_idx = pos_idx["person_id"]
                gender_idx = pos_idx["gender_concept_id"]
                yob_idx = pos_idx["year_of_birth"]
                location_id = pos_idx["location_id"]
            else:
                pid = int(vals[pid_idx])
                if pid not in p_obs_dates_dict:
                    continue
                else:
                    obs_date = datetime.strptime(p_obs_dates_dict[pid][0], "%Y-%m-%d")
                    p_age = obs_date.year - int(vals[yob_idx])  # age
                    p_gender = 1 if vals[gender_idx] == '8507' else 0
                    p_state = loc_id_state[vals[location_id]]
                    # update dictionary
                    person_data_dict[pid] = [p_gender, p_age, p_state]

    print("{0} persons details were fetched".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(person_data_dict, f, protocol=5)

    return person_data_dict


def date_to_object(p_date_dict, n_elem=1):
    """
    convert date from YYYY-MM-DD format to datetime object
    """
    dateobj_dict = {}
    for kk, vv in p_date_dict.items():
        if n_elem == 1: # first diabetes date
            dateobj_dict[kk] = datetime.strptime(vv, "%Y-%m-%d")
        elif n_elem == 2: # observation dates
            dateobj_dict[kk] = [datetime.strptime(vv[0], "%Y-%m-%d"), datetime.strptime(vv[1], "%Y-%m-%d")]
    return dateobj_dict


def get_person_conditions(pp_obs_date_dict, concept_id_vocab_dict, ifile=None, ofile=None):
    """
    fetch person conditions from the input file.
    """
    print("\nselect condition covariates for persons")
    person_conditions_dict = {}

    print("covert dates to datetime object")
    # convert observation dates to object
    p_obs_date_dict = date_to_object(pp_obs_date_dict, n_elem=2)

    print("read the file and process records")
    # read the file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 25000000 == 0:
                print("fetched conditions from {0} records".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                pid_idx = pos_idx["person_id"]
                cond_idx = pos_idx["condition_concept_id"]
                cond_src_idx = pos_idx["condition_source_concept_id"]
                cond_date_idx = pos_idx["condition_start_date"]
            else:
                if vals[cond_idx] == '0':   # skip bad codes
                    continue

                # get data from the line
                pid, p_src_cond, p_cond_date = int(vals[pid_idx]), int(vals[cond_src_idx]), vals[cond_date_idx]

                # select only persons who were observed
                if pid in p_obs_date_dict:
                    obs_start_date = p_obs_date_dict[pid][0]
                    obs_end_date = p_obs_date_dict[pid][1]
                    cond_date = datetime.strptime(p_cond_date, "%Y-%m-%d")

                    # condition should occur between observation period and should be ICD10CM
                    if obs_start_date < cond_date < obs_end_date and concept_id_vocab_dict[p_src_cond][0] == 'ICD10CM':
                        # initialize the dictionary for the pid and condition code
                        if pid not in person_conditions_dict:
                            person_conditions_dict.setdefault(pid, {})
                        if p_src_cond not in person_conditions_dict[pid]:
                            person_conditions_dict[pid].setdefault(p_src_cond, set())

                        # time to condition
                        ttc =  (cond_date - obs_start_date).days
                        person_conditions_dict[pid][p_src_cond].add(ttc)

    print("fetched conditions from {0} records".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(person_conditions_dict, f, protocol=5)

    return person_conditions_dict



def get_person_drugs(pp_obs_date_dict, ifile=None, ofile=None):
    """
    fetch drugs for each person from the input file
    """
    print("\nselect drug covariates for persons")
    person_drugs_dict = {}

    print("covert dates to datetime object")
    # convert observation dates to object
    p_obs_date_dict = date_to_object(pp_obs_date_dict, n_elem=2)

    print("read the file and process records")
    # read the file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 5000000 == 0:
                print("fetched drugs from {0} records".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                pid_idx = pos_idx["person_id"]
                drug_idx = pos_idx["drug_concept_id"]
                drug_date_idx = pos_idx["drug_era_start_date"]
                drug_exposure_idx = pos_idx["drug_exposure_count"]
            else:
                if vals[drug_idx] == '0':  # bad code
                    continue
        
                pid, drug_concept_id = int(vals[pid_idx]), int(vals[drug_idx])
                drug_start_date, drug_exposure_count = vals[drug_date_idx], int(vals[drug_exposure_idx])

                # select only persons who were observed
                if pid in p_obs_date_dict:
                    obs_start_date = p_obs_date_dict[pid][0]
                    obs_end_date = p_obs_date_dict[pid][1]
                    d_date = datetime.strptime(drug_start_date, "%Y-%m-%d")

                    # drug date should be within observation period
                    if obs_start_date < d_date < obs_end_date:
                        # initialize the dictionary for the pid and drug code 
                        if pid not in person_drugs_dict:
                            person_drugs_dict.setdefault(pid, {})
                        if drug_concept_id not in person_drugs_dict[pid]:
                            person_drugs_dict[pid].setdefault(drug_concept_id, set())
                        
                        # time to drug exposure
                        ttc =  (d_date - obs_start_date).days
                        person_drugs_dict[pid][drug_concept_id].add((ttc, drug_exposure_count))

    print("fetched drugs from {0} records".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(person_drugs_dict, f, protocol=5)

    return person_drugs_dict


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
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # STEP 1: read concept file and create a dictionary with concept_id as key and vocab_id and concept_name as values. Also, update key in ancestor file
    concept_id_name_dict, descendant_ancestor_dict = fetch_concept_id_name(ifile=iodata['input_files']['concepts_file'],
                                                                         afile=iodata['output_files']['ICD10_concept_ancestors_file'])
    print("number of concepts: ", len(concept_id_name_dict))

    # STEP 2: create person and observation dates dictionary
    if not os.path.exists(iodata['output_files']['person_obs_date_file']):
        person_obs_dates_dict = fetch_observation_dates(ifile=iodata['input_files']['person_observation_file'],
                                                        ofile=iodata['output_files']['person_obs_date_file'])
    else:
        with open(iodata['output_files']['person_obs_date_file'], 'rb') as f:
            person_obs_dates_dict = pickle.load(f)
    print("unique patient observation count: ", len(person_obs_dates_dict))

    # STEP 3: fetch person details - age, sex, state
    if not os.path.exists(iodata['output_files']['person_data_file']):
        person_data_dict = get_person_details(person_obs_dates_dict, ifile=iodata['input_files']['person_data_file'],
                                              ofile=iodata['output_files']['person_data_file'])
    else:
        with open(iodata['output_files']['person_data_file'], 'rb') as f:
            person_data_dict = pickle.load(f)
    print("unique person count: ", len(person_data_dict.keys()))
    '''
    # STEP 4: fetch person conditions
    if not os.path.exists(iodata['output_files']['person_cond_file']):
        person_conditions_dict = get_person_conditions(person_obs_dates_dict, concept_id_name_dict, 
                                                       ifile=iodata['input_files']['person_condition_file'],
                                                       ofile=iodata['output_files']['person_cond_file'])
    else:
        with open(iodata['output_files']['person_cond_file'], 'rb') as f:
            person_conditions_dict = pickle.load(f)
    print("keys in person_conditions_dict: ", len(person_conditions_dict))
    '''
    # STEP 5: fetch person drugs
    if not os.path.exists(iodata['output_files']['person_drug_file']):
        person_drugs_dict = get_person_drugs(person_obs_dates_dict, ifile=iodata['input_files']['person_drug_file'], 
                                             ofile=iodata['output_files']['person_drug_file'])
    else:
        with open(iodata['output_files']['person_drug_file'], 'rb') as f:
            person_drugs_dict = pickle.load(f)
    print("keys in person_drugs_dict: ", len(person_drugs_dict))


if __name__ == "__main__":
    main()

import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUM_THREADS'] = '8'

import pandas as pd
import numpy as np
import argparse
import yaml
import pickle
import gzip
from datetime import datetime, timedelta
from scipy.sparse import lil_matrix
from PULSNAR import PULSNAR
from sklearn.utils import shuffle
from collections import Counter
from copy import deepcopy


def fetch_concept_id_name(ifile=None):
    """
    Read concept file and return a dictionary with concept_id as key and concept_name + vocab_id as value
    """
    print("\ncreating dictionary using concept id, vocab_id, and name")
    id_name_dict = {}

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
            else:
                cid, cname, cvocab = int(vals[cid_idx]), vals[cname_idx], vals[vocab_idx]
                id_name_dict[cid] = [cvocab, cname]

    print("{0} concept records processed".format(line_count))
    return id_name_dict


def get_ccsr_icd_mapping(ifile1=None, ifile2=None):
    """
    fetch CCSR ICD10 mapping dictionaries
    """
    print("save dictionary - CCSR and ICD codes")
    with open(ifile1, 'rb') as f:
        ccsr_icd_mapping= pickle.load(f)

    with open(ifile2, 'rb') as f:
        icd_ccsr_mapping = pickle.load(f)

    return ccsr_icd_mapping, icd_ccsr_mapping


def date_to_object(p_date_dict, n_elem=1):
    """
    convert date from YYYY-MM-DD format to datetime object
    """
    dateobj_dict = {}
    for kk, vv in p_date_dict.items():
        if n_elem == 1: # first condition date
            dateobj_dict[kk] = datetime.strptime(vv, "%Y-%m-%d")
        elif n_elem == 2: # observation dates
            dateobj_dict[kk] = [datetime.strptime(vv[0], "%Y-%m-%d"), datetime.strptime(vv[1], "%Y-%m-%d")]
    return dateobj_dict


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
    loc_id_state = {'1': 'AK', '2': 'AL', '3': 'AR', '4': 'AZ', '5': 'CA', '6': 'CO', '7': 'CT', '8': 'DC',
                    '9': 'DE', '10': 'FL', '11': 'GA', '12': 'HI', '13': 'IA', '14': 'ID', '15': 'IL',
                    '16': 'IN', '17': 'MI', '18': 'KS', '19': 'KY', '20': 'LA', '21': 'MA', '22': 'MD',
                    '23': 'ME', '24': 'MN', '25': 'MO', '26': 'MS', '27': 'MT', '28': 'NC', '29': 'ND',
                    '30': 'NE', '31': 'NH', '32': 'NJ', '33': 'NM', '34': 'NV', '35': 'NY', '36': 'OH',
                    '37': 'OK', '38': 'OR', '39': 'PA', '40': 'PR', '41': 'RI', '42': 'SC', '43': 'SD',
                    '44': 'TN', '45': 'TX', '46': 'UN', '47': 'UN', '48': 'UN', '49': 'UN', '50': 'UN',
                    '51': 'UN', '52': 'UN', '53': 'UT', '54': 'VA', '55': 'VT', '56': 'WA', '57': 'WI',
                    '58': 'WV', '59': 'WY'}

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
                        # time to condition
                        ttc =  (cond_date - obs_start_date).days
                        person_conditions_dict.setdefault(pid, set()).add((p_src_cond, ttc))

    print("fetched conditions from {0} records".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(person_conditions_dict, f, protocol=5)

    return person_conditions_dict

def select_time_window_data(ccsr_code, ccsr_covariates, person_ids, person_conds_dict, icd_ccsr_mapping, num_days):
    """
    this function selects all conditions during the given time window
    """
    features, labels = lil_matrix((len(person_ids), len(ccsr_covariates))), []
    # update lil_matrix
    for r, pid in enumerate(person_ids):
        pid_covars = []
        conds = person_conds_dict[pid]
        for cond in conds:
            if cond[1] <= num_days:
                pid_covars.extend(v for v in icd_ccsr_mapping.get(cond[0], []))
        c = np.isin(ccsr_covariates, pid_covars).nonzero()[0]
        labels.append(1 if ccsr_code in pid_covars else 0)
        features[r, c] = 1
    return features.tocsr(), np.asarray(labels)

def run_pulsnar_on_selected_data(X, Y, rec_ids, user_param_file, rseed=0):
    """
    run PULSNAR algorithm on the selected data
    """
    orig_person_ids = deepcopy(rec_ids) # keep a copy for sorting predictions
    X, Y, rec_ids = shuffle(X, Y, rec_ids, random_state=rseed)
    Y_true = Y
    count_01 = Counter(Y)
    print(f"class 0: {count_01[0]}, class 1:{count_01[1]}, class ratio (0:1): {count_01[0]/count_01[1]}")

    # instantiate PULSNARClassifier
    pls = PULSNAR.PULSNARClassifier(scar=True, csrdata=True, classifier='xgboost',
                                    bin_method='rice', bw_method='hist', lowerbw=0.05, upperbw=0.5, optim='local',
                                    calibration=True, calibration_data='U', calibration_method='sigmoid',
                                    calibration_n_bins=100, smooth_isotonic=False,
                                    classification_metrics=False,
                                    n_iterations=1, kfold=5, kflips=1,
                                    pulsnar_params_file=user_param_file)

    # get results
    res = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)
    print("Estimated alpha: {0}".format(res['estimated_alpha']))

    # get prediction for unlabeled from the file
    u_df = pd.read_csv(res['prediction_file'], sep="\t", header=0)

    # sort the prediction according to person_ids
    u_pid_preds = dict(zip(u_df['rec_id'], u_df['calibrated_prob']))
    # print("run_pulsnar_on_selected_data: ", orig_person_ids)
    ccsr_pulsnar_preds = [u_pid_preds[v] if v in u_pid_preds else 1 for v in orig_person_ids]
    return ccsr_pulsnar_preds

def main():
    """
    This code runs PULSNAR for 500+ CCSR codes as outcomes
    """
    # load IO filenames from the YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
    parser.add_argument("-window_size", default=7, type=int, help="provide window size (number of days)")
    parser.add_argument("-paramfiles", default="ml_params.yaml", help="provide ML parameter file")
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # STEP 1: read concept file and create a dictionary with concept_id as key and vocab_id + concept_name as values
    concept_id_name_dict = fetch_concept_id_name(ifile=iodata['input_files']['concepts_file'])
    print("number of concepts: ", len(concept_id_name_dict))

    # load mapping
    ccsr_icd_mapping, icd_ccsr_mapping = get_ccsr_icd_mapping(ifile1=iodata['ccsr_icd_mapping_file'],
                                                              ifile2=iodata['icd_ccsr_mapping_file'])
    print(f"number of elements in ccsr_icd_mapping: {len(ccsr_icd_mapping)}")
    print(f"number of elements in icd_ccsr_mapping: {len(icd_ccsr_mapping)}")

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

    # STEP 4: fetch person conditions
    if not os.path.exists(iodata['output_files']['person_cond_file']):
        person_conds_dict = get_person_conditions(person_obs_dates_dict, concept_id_name_dict,
                                                  ifile=iodata['input_files']['person_condition_file'],
                                                  ofile=iodata['output_files']['person_cond_file'])
    else:
        with open(iodata['output_files']['person_cond_file'], 'rb') as f:
            person_conds_dict = pickle.load(f)
        print("unique person condition count: ", len(person_conds_dict))

    # generate ML data for each window and run PULSNAR
    ccsr_covars = np.asarray(list(ccsr_icd_mapping.keys())) # use CCSR codes as covariates
    person_ids = np.asarray(list(person_conds_dict.keys())) # select all persons

    # keep the sequence of persons unchanged
    orig_pids = deepcopy(person_ids)

    # dataframe to store predictions
    pulsnar_ccsr_predictions = pd.DataFrame(person_ids, columns=['person_id'])
    for j, ccsr in enumerate(ccsr_covars):
        for num_days in range(365, 730 + p_args.window_size, p_args.window_size):
            print(f"processing data for window size: {num_days} and CCSR: {ccsr}")
            person_ids = deepcopy(orig_pids)    # to ensure pid sequence remains same
            data, labels = select_time_window_data(ccsr, ccsr_covars, person_ids, person_conds_dict, icd_ccsr_mapping, num_days)

            # drop outcome column from feature matrix
            # print(f"data shape: {data.shape}")
            cols = np.ones(data.shape[1], dtype=bool)
            cols[j] = False
            data = data[:, cols]
            print(f"size of the ML data for PULSNAR: {data.shape}, {len(labels)}")

            # run PULSNAR on the data for the selected
            print(f"running PULSNAR on the selected data of size: {data.shape}")
            preds_ccsr = run_pulsnar_on_selected_data(data, labels, person_ids, p_args.paramfiles, rseed=1001)

            # save prediction in the dataframe
            col_name = "window_" + str(num_days)
            pulsnar_ccsr_predictions[col_name] = preds_ccsr
            # print(pulsnar_ccsr_predictions)

            if num_days >= 400:
                ccsr_fileName = iodata['output_files']['pulsnar_predictions_file'] + ccsr + ".tsv"
                pulsnar_ccsr_predictions.to_csv(ccsr_fileName, sep="\t", index=False)
                exit()

if __name__ == "__main__":
    main()
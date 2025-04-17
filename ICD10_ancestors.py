import pickle
import yaml
import argparse
import os
import gzip


def get_concept_code_and_id(ifile=None):
    """
    create a dictionary with concept code as key and concept id as value (only for ICD10 conditions)
    """
    print("\ncreate dictionary with concept code and concept id")
    concept_code_id = {}

    # read the input file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            line_count += 1
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                cid_idx = pos_idx["concept_id"]
                code_idx = pos_idx["concept_code"]
                vocab_idx = pos_idx["vocabulary_id"]
            else:
                cid, ccode, cvocab = int(vals[cid_idx]), vals[code_idx], vals[vocab_idx]
                if 'ICD10' in cvocab and cvocab != 'ICD10PCS':
                    concept_code_id.setdefault(ccode, set()).add(cid)

            # track progress
            if line_count % 2500000 == 0:
                print("{0} concept records processed".format(line_count))
    print("{0} concept records processed".format(line_count))

    return concept_code_id


def find_ancestors(concept_data, icd10_chapter, ifile=None, ofile=None):
    """
    find all ancestors of ICD10 condition codes
    """
    print("\ncreate dictionary with concept code as key and ancestors as values")
    concept_ancestors_dict = {}

    # read the input file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            line_count += 1
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                cid_idx = pos_idx["concept_id"]
                code_idx = pos_idx["concept_code"]
                vocab_idx = pos_idx["vocabulary_id"]
            else:
                cid, ccode, cvocab = int(vals[cid_idx]), vals[code_idx], vals[vocab_idx]
                if 'ICD10' in cvocab and cvocab != 'ICD10PCS':
                    concept_code = ccode
                    if '.' not in ccode:
                        if ccode in concept_data:
                            concept_ancestors_dict.setdefault(concept_code, set()).update(concept_data[ccode])
                        kk = find_icd10_chapter_key(ccode, icd10_chapter)
                        if kk is not None:
                            concept_ancestors_dict.setdefault(concept_code, set()).add(icd10_chapter[kk])
                    else:
                        DOT_REACHED = False
                        while not DOT_REACHED:
                            if ccode in concept_data:
                                concept_ancestors_dict.setdefault(concept_code, set()).update(concept_data[ccode])
                            ccode = ccode[:-1]
                            if ccode[-1] == '.':
                                ccode = ccode[:-1]
                                if ccode in concept_data:
                                    concept_ancestors_dict.setdefault(concept_code, set()).update(concept_data[ccode])
                                kk = find_icd10_chapter_key(ccode, icd10_chapter)
                                if kk is not None:
                                    concept_ancestors_dict.setdefault(concept_code, set()).add(icd10_chapter[kk])
                                DOT_REACHED = True
                            else:
                                if ccode in concept_data:
                                    concept_ancestors_dict.setdefault(concept_code, set()).update(concept_data[ccode])

            # track progress
            if line_count % 2500000 == 0:
                print("{0} concept records processed for ancestors".format(line_count))
    print("{0} concept records processed for ancestors".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(concept_ancestors_dict, f, protocol=5)

    return concept_ancestors_dict


def find_icd10_chapter_key(concept_code, icd10_chapter):
    """
    determine key in the icd10_chapter
    """
    kkey = None
    for kk in icd10_chapter:
        st, en = kk.split('-')
        if st <= concept_code <= en:
            kkey = kk
    return kkey


def main():
    """
    identify ancestors for all ICD10 conditions
    """
    # load IO filenames from the YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # STEP 1: create a dictionary with concept code as key and concept id as value
    concept_code_id_data = get_concept_code_and_id(ifile=iodata['input_files']['concepts_file'])
    print("number of records in concept_code_id dictionary: ", len(concept_code_id_data))

    # STEP 2: determine ancestors of all ICD10 condition codes
    if not os.path.exists(iodata['output_files']['ICD10_concept_ancestors_file']):
        concept_ancestors_dict = find_ancestors(concept_code_id_data, iodata['icd10_webpage_database_mapping'],
                                                ifile=iodata['input_files']['concepts_file'],
                                                ofile=iodata['output_files']['ICD10_concept_ancestors_file'])
    else:
        with open(iodata['output_files']['ICD10_concept_ancestors_file'], 'rb') as f:
            concept_ancestors_dict = pickle.load(f)

    # print(concept_ancestors_dict)

if __name__ == "__main__":
    main()

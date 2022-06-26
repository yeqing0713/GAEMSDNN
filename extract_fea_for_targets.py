
import rdkit
from rdkit import Chem
import csv
from Bio import SeqIO
import pandas as pd
from propy import PyPro
import numpy as np
save_target_feas_path = r'E:\DTI\drug_PaDel\tf.csv'
seq_path = r"E:\DTI\drug_PaDel\target.csv"
def extract_feas_by_fasta():
    f = open(save_target_feas_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    fa = r"E:\DTI\drug_bank_all\fasta\protein.fasta"
    targets = pd.read_csv(seq_path, header=None).values
    for ii in range(targets.shape[0]):
        if ii == 0:
            continue
        print(ii)
        pseq = targets[ii][2]
        print(pseq)
        DesObject = PyPro.GetProDes(pseq)
        csv_writer.writerow(DesObject.GetALL().values())


extract_feas_by_fasta()
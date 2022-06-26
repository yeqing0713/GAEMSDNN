import DTI_all_randoms_MS

Enzyme = [r'data2\Enzyme_df.csv', r'data2\Enzyme_tf.csv', r'data2\Enzyme_Y.csv']
GPCR = [r'data2\GPCR_df.csv', r'data2\GPCR_tf.csv', r'data2\GPCR_Y.csv']
IC = [r'data2\IC_df.csv', r'data2\IC_tf.csv', r'data2\IC_Y.csv']
NR = [r'data2\NR_df.csv', r'data2\NR_tf.csv', r'data2\NR_Y.csv']
#Enzyme = [r'dataR8S8\Enzyme_df.csv', r'dataR8S8\Enzyme_tf.csv', r'dataR8S8\Enzyme_Y.csv']
#GPCR = [r'dataR8S8\GPCR_df.csv', r'dataR8S8\GPCR_tf.csv', r'dataR8S8\GPCR_Y.csv']
#IC = [r'dataR8S8\IC_df.csv', r'dataR8S8\IC_tf.csv', r'dataR8S8\IC_Y.csv']
#NR = [r'dataR8S8\NR_df.csv', r'dataR8S8\NR_tf.csv', r'dataR8S8\NR_Y.csv']
drug_bank = [r'data\drug_bank_df.csv', r'data\drug_bank_tf.csv', r'data\drug_bank_Y.csv']
tabei = [r'data\tabei_df.csv', r'data\tabei_tf.csv', r'data\tabei_Y.csv']
drug_bank_approv = [r'data\drug_bank_approv_df.csv', r'data\drug_bank_approve_tf.csv', r'data\drug_bank_approve_Y.csv']

datasets = []
datasets.append(Enzyme)
datasets.append(GPCR)
datasets.append(IC)
datasets.append(NR)


#datasets.append(tabei)
#datasets.append(drug_bank)
#datasets.append(drug_bank_approv)
import warnings
warnings.filterwarnings('ignore')

DR_types = ['GCNAutoEncode', 'VGAEAutoEncode', 'SRC', 'DNNAutoEncode', ]
#exp_types = ['CVP', 'CVD', 'CVT']
exp_types = ['CVP']
model_types = ['RBM_n_MS']
graph_types = ['knn_graph', 'gdc_knn_graph']

#hider_dims = [300, 500]
#drop_ratios = [0.5, 0.7]
#layer_UUs = [1, 3, 5, 7, 9, 11]
hider_dims = [300]
drop_ratios = [0.5]
layer_UUs = [2]
neg_ratios = [5]
sub_ratios = [0.7]
is_SMOTEs = [False]
neg_ratio = neg_ratios[0]
ks = [4]
if neg_ratio == 1:
    is_SMOTEs = [False]
print('test use all')
for exp_type in exp_types:
    for DR_type in DR_types:
        for dataset in datasets:
            print(dataset)
            for model_type in model_types:
                for layer_UU in layer_UUs:
                    for hider_dim in hider_dims:
                        for drop_ratio in drop_ratios:
                            for is_SMOTE in is_SMOTEs:
                                for k in ks:
                                    for sub_ratio in sub_ratios:
                                        print('负面样本比例:', neg_ratio, end='')
                                        print(',  实验类型:', exp_type, end='')
                                        print(',  降维类型:', DR_type, end='')
                                        print(',  深度学习类型:', model_type, end='')
                                        print(',  层数:', layer_UU, end='')
                                        print(',  隐藏层维度:', hider_dim, end='')
                                        print(',  drop比例:', drop_ratio, end='')
                                        print(',  子空间个数:', k, end='')
                                        print(',  子空间维度比例:', sub_ratio, end='')
                                        print(',  使用SMOTE:', is_SMOTE)
                                        DTI_all_randoms_MS.DTI_all_randoms(dataset[0], dataset[1], dataset[2], exp_type, layer_UU, hider_dim, drop_ratio, DR_type, model_type, 'knn_graph', neg_ratio, k, sub_ratio, is_SMOTE, False)





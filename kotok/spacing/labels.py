# All valid labels
#  O: Other/Nothing
#  N: Normal
#  SM: Space Missing
#  SE: Space Extra
all_labels = ['O', 'N', 'SM', 'SE']

label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for i, label in enumerate(all_labels)}
num_labels = len(all_labels)

# All valid POS tags
pos_tags = [
    'NNG', 'NNP', 'NNB',
    'NR', 'NP',
    'VV', 'VA', 'VX', 'VCP', 'VCN',
    'MM', 'MAG', 'MAJ', 'IC',
    'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
    'EP', 'EF', 'EC', 'ETN', 'ETM',
    'XPN', 'XSN', 'XSV', 'XSA', 'XR',
    'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SH', 'SL', 'SN',
]

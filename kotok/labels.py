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

# All valid labels
#  O: Other/Nothing
#  B-POS: Begin POS
#  I-POS: Inside POS (continuation of B-POS tag)
all_labels = ['O'] + [f'B-{tag}' for tag in pos_tags] + [f'I-{tag}' for tag in pos_tags]

label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for i, label in enumerate(all_labels)}
num_labels = len(all_labels)

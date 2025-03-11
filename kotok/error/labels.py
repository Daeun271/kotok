# All valid labels
#  O: Other/Nothing
#  B-: Begin Morpheme
#  I-: Inside Morpheme
#  -M: Morpheme
#  -ME: Morpheme with error
all_labels = ['O', 'B-M', 'I-M', 'B-ME', 'I-ME']

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

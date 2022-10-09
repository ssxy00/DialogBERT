# DialogBERT-GLCM
Re-implement DialogBERT to compare with GLCM

All experiments should be run on group

## Experiment on DailyDialog
fix lr, tune lr on [5e-5, 8e-5, 1e-4, 3e-4, 5e-4]

choose lr=3e-4, step2792, ckpt8

test result:

avg_len = 15.36171310629515
bleu = 0.061740663318785916
meteor = 0.08774378189106594
nist = 0.24757966547917468
perplexity = 39.911922454833984
rouge-L = 0.11350940054182108
valid_loss = 3.6866750781750164


## Experiment on MultiWOZ
fix lr, tune lr on [1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 3e-4]
choose lr=3e-5, step37390, ckpt41

test result:

avg_len = 17.04050053821313
bleu = 0.0826381049789987
meteor = 0.1327813784639214
nist = 0.4203184224172626
perplexity = 5.963738918304443
rouge-L = 0.1450864178597785
valid_loss = 1.7856975986980166

## Experiment on PersonaChat
因为没想好 persona 的部分要怎么处理，所以现在的实验中直接把 persona 扔掉了

fix lr, tune lr on [5e-5, 8e-5, 1e-4]
choose lr=8e-5, step15070, ckpt7

test results:

avg_len = 12.786873840445269
bleu = 0.06556580196044658
meteor = 0.07826479254076116
nist = 0.32445281356699346
perplexity = 46.892879486083984
rouge-L = 0.1082527998176345
valid_loss = 3.847865749972981
# DialogBERT-GLCM
Re-implement DialogBERT to compare with GLCM

All experiments should be run on group

## Experiment on DailyDialog
fix lr, tune lr on [5e-5, 8e-5, 1e-4, 3e-4, 5e-4]

choose lr=3e-4, step2792, ckpt8

test result:

avg_len = 15.223426212590299
bleu = 0.06265570520804029
meteor = 0.08802391577631702
nist = 0.25103816171942167
perplexity = 38.02140426635742
rouge-L = 0.11353710231240573
valid_loss = 3.6381492361933825


## Experiment on MultiWOZ
fix lr, tune lr on [1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 3e-4]
choose lr=3e-5, step37390, ckpt41

test result:

avg_len = 16.90191065662002
bleu = 0.08416763725538154
meteor = 0.13466733392824404
nist = 0.4286227693253055
perplexity = 5.599846839904785
rouge-L = 0.14710063210354277
valid_loss = 1.7227392059072162

# TRAINING

Il codice `trainBasicUNet.py` viene eseguito tramite bash. 

    python trainBasicUNet.py --train.csv --valid.csv --dict_save_model --gpu --loss --reduction --epochs --train_batch_size --valid_batch_size

Dove i parametri rappresentano:
  1. Path `train.csv`
  2. Path `valid.csv`
  3. Path directory dove verranno salvati i pesi del modello
  4. La gpu su cui verrà eseguito il codice
  5. Il tipo di loss che verrà utilizzata
  6. Il tipo di reduzione adotattao
  7. Numero di epoche
  8. Valore batch del train
  9. Valore batch del validation

**Solo i primi tre argomenti sono obbligatori**.

## Formato CSV

| ImmA | ImmB  | GT                     | 
| ----------- | --- | ------------------------------- |
| path_immA | path_immB | path_GT| 

[Esempio CSV](https://github.com/Raciti/Brain-Change-Estimation/blob/main/Data/train.csv) 

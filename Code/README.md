# TRAINING

Il codice `trainBasicUNet.py` viene eseguito tramite bash. 

    python trainBasicUNet.py train.csv valid.csv dict_save_model epochs train_batch_size valid_batch_size

Dove i parametri rappresentano:
  1. Path `train.csv`
  2. Path `valid.csv`
  3. Path directory dove verranno salvati i pesi del modello
  4. Numero di epoche
  5. Valore batch del train
  6. Valore batch del validation

**Solo i primi tre argomenti sono obbligatori**.

## Formato CSV

| ImmA | ImmB  | GT                     | 
| ----------- | --- | ------------------------------- |
| path_immA | path_immB | path_GT| 

[Esempio CSV](https://github.com/Raciti/Brain-Change-Estimation/blob/main/Data/train.csv) 

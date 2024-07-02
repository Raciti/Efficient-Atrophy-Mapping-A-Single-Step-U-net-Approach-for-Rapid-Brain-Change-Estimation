# TRAINING

The `trainBasicUNet.py` code is executed via bash. 

    python trainBasicUNet.py --train.csv --valid.csv --dict_save_model --gpu --loss --reduction --exp --load_model --load_optim --scheduler --epochs --train_batch_size --valid_batch_size --start_epoch

Where the parameters represent:
  1. Path `train.csv`
  2. Path `valid.csv`
  3. Path directory where model weights will be saved
  4. The gpu on which the code will be executed
  5. The type of loss that will be used
  6. The type of reduction adotattao
  7. The value of the exponent in the use of the MSE
  8. The path of model.pth
  9. The path of optim.pth
  10. Enabling the Scheduler
  11. Number of epochs
  12. Batch value of the train
  13. Batch value of the validation
  14. For the final grhap


**Only the first three topics are mandatory**.

The code `results.py` is executed via bash.

    python results.py --calculate 

Where
 1. Calculate represents a bollenano value to regenerate dictionaries, if placed in False it will generate only the plots

## Formato CSV

| ImmA | ImmB  | GT                     | 
| ----------- | --- | ------------------------------- |
| path_immA | path_immB | path_GT| 

[CSV Example](https://github.com/Raciti/Brain-Change-Estimation/blob/main/Data/train.csv) 

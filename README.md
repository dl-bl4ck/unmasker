# UnMasker

Given a masked face image our task is to create an approximate description of the facial features (or an unmasked face) using generative deep learning models.

#TODO ADD SAMPLE IMAGE FLOWCHART

# Try out the code

To test a running model of the program [https://unmasker.herokuapp.com/](https://unmasker.herokuapp.com/)

# Run the code

Clone the repository and cd into it

```
git clone https://github.com/dl-bl4ck/unmasker.git
cd unmasker
```

## Installing Dependencies

Make sure you have a Python3+ version. Run the following command - 

```bash
pip install -r requirements.txt
```

## Training the Model

### Dataset

Download the dataset from [here](https://drive.google.com/file/d/1h6Mg9RsjUIJyerOZlH4yYRQCIQ5GP0dZ/view?usp=sharing) and unzip the downloaded dataset. This will create a ```./data/``` in the root folder of the repository.  

### Training
```bash
python3 train.py [--data_dir (str)] [--wandb (str)] [--max_train (int)] [--max_test (int)] [--init_model_cn (str)] [--steps_1 (int)] [--snaperiod_1 (int)] [--num_test_completions (int)] [--alpha (float)] [--batch_size (int)] [--learning_rate (float)]
```
Options : 
```
--data_dir              Path of directory of data/ folder

--wandb                 Name of the [wandb](https://wandb.ai/) project.

--max_train             Maximum number of training images to load to use for testing. 

--max_test              Maximum number of testing images to load use for evaluation. 

--init_model_cn         Path of the model with which you want to initialise the model before training. 

--steps_1               No of epochs you want to train your generative model. 

--snaperiod_1           Number of batches after you want to print the loss for one epoch.

--num_test_completions  Number of test images to use for evaluation while training

--alpha                 Regularisation Constant

--batch_size            Batch size to be used while training

--learning_rate         Learning Rate to be used while training
```

## Inference

Download the trained model from (here)[]. 
To finally test your trained model run the following command

```
python3 predict.py [--model (str)] [--input_img (str)] [--output_img (int)] 
```
Options : 
```
--model                 Path of trained model

--input_img             Path of input masked image

--output_img            Path of output un-masked image

```

## License 

Copyright (c) 2021  Paras Mehan

For license information, see [LICENSE](LICENSE) or http://mit-license.org


- - -

Done by [Pragyan Mehrotra](https://github.com/pragyanmehrotra), [Vrinda Narayan](https://github.com/vrindaaa) and [Paras Mehan](https://github.com/parasmehan123)



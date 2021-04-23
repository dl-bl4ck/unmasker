# Problem Description

Given a masked low resolution (64x64) masked face image our task is to create an approximate description of the facial features (or an unmasked face)
without a mask using generative deep learning models.

#TODO ADD SAMPLE IMAGE FLOWCHART

# Try out the code

To test a running model of the program [https://unmasker.herokuapp.com/](https://unmasker.herokuapp.com/)

# Run the code

Clone the repository and cd into it

```
git clone https://github.com/dl-bl4ck/unmasker.git
cd unmasker
```

## Install dependencies

`pip -r install requirements.txt`


## Training the model

To train the model efficiently you can edit the configurations for the model according to your systems requirements and your needs using the following arguments
```python
python3 train.py --data-dir=<path-to-dataset> --use_one_dir=<int> --wandb=<int> --max_train=<int> --max_test=<int> --result_dir=<path-to-directory> ....... #TODO ADD MODE TAGS
```
`--data-dir`: To locate the dataset for the model </br>
`--max_train`: Maximum training data to use </br>
`--max_test`: Maximum Testing data to use </br>
`--result_dir`: Path to a result directory to store the output of the model in </br>
`--wandb`: 1 to create and log results on wandb and 0 to not use wandb (weights and biases) </br>
#TODO ADD MORE TAGS

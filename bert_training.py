import sys, os, getopt
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import torch
from copy import deepcopy
from pprint import pprint
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

# From my helper function file.
from bert_helper import prepare_input, get_predictions, train_model, adapt_model

## Initialize constants.
MODEL = ''
MODEL_DIR = "C:\\Users\\hecto\\Documents\\MIT\\bert-priming\\models\\"
TRAIN_DIR = ''
ADAPT_DIR = ''
TEST_DIR = ''
OUTPUT_DIR = ''
SCRAMBLE_TRAIN = False
SCRAMBLE_ADAPT = False
RANDOM_SEED = 69 # Nice.

## Other hyperparameters.
#  These were recommended by the authors of the original BERT paper,
#  So these will be hardcoded for the moment.
MAX_LEN = 64
BATCH_SIZE = 20
EPOCHS = 4

def get_args(argv):
    ## Tell python that I want to deal with the constants at the top of the file.
    global MODEL, TRAIN_DIR, ADAPT_DIR, TEST_DIR, OUTPUT_DIR
    global SCRAMBLE_TRAIN, SCRAMBLE_ADAPT, RANDOM_SEED
    
    ## Get command line arguments and set constants for the rest of the file.
    try:
        opts, args = getopt.getopt(argv, "", ["model=","train=", "adapt=", "test=",
                                              "out=", "scramble_train=", "scramble_adapt=", "seed="])
    except getopt.GetoptError:
        print('''test.py --model <model name or directory>
                         --train <path/to/training_set>
                         --test  <path/to/test_set>
                         --out <path/to/save/model>
                         --scramble <True or False>
            ''')

    ## Get all the arguments used to call the script.
    for opt, arg in opts:
        if opt == '--model':
            MODEL = arg
        elif opt == "--train":
            TRAIN_DIR = arg
            if os.path.exists(arg):
                TRAIN_DIR = arg
            else:
                TRAIN_DIR = False
        elif opt == "--adapt":
            ADAPT_DIR = arg
            if os.path.exists(arg):
                ADAPT_DIR = arg
            else:
                ADAPT_DIR = False
        elif opt == "--test":
            TEST_DIR  = arg
            if os.path.exists(arg):
                TEST_DIR = arg
            else:
                TEST_DIR = False
        elif opt == "--out":
            OUTPUT_DIR = arg
        elif opt == "--scramble_train":
            SCRAMBLE_TRAIN = arg.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']
        elif opt == "--scramble_adapt":
            SCRAMBLE_ADAPT = arg.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']
        elif opt == "--seed":
            try:
                RANDOM_SEED = int(arg)
            except:
                print("Invalid random seed given.\nUsing default random seed:", RANDOM_SEED)
    
    ## Sanity checks.
    if not TEST_DIR:
        print(TEST_DIR)
        print('Test set not provided for the model.\nTerminating program execution...')
        exit()


def log_results(model_results, output_dir=None):
    
    ## Store the model's performance in a .csv file.
    if os.path.exists(OUTPUT_DIR):
        try:
            old_results = pd.read_csv(OUTPUT_DIR, index_col=0)
            new_cols = model_results.columns.difference(old_results.columns)
            
            new_results = old_results.join(model_results[new_cols])
            new_results.to_csv(OUTPUT_DIR)
            
            print("Saved new results to: {}".format(OUTPUT_DIR))
            print(model_results[new_cols])
        except:
            print('An unknown error occurred: results could not be saved.  \nNow outputing results to the console...\n')
            print(model_results)
            print("Terminating program execution...")
            exit()
    else:
            print('The specified output file was not found.  Creating a new output file...')
            model_results.to_csv(OUTPUT_DIR)

            
def training(model, tokenizer, device, training_set, random_seed=RANDOM_SEED):
    
    
    ## Split the dataframes into sentences and labels.
    train_sentences = training_set.sentence.values
    train_labels = training_set.label.values
    
    ## Training.
    # Prepare the input for training.
    input_ids, attention_masks = prepare_input(train_sentences, tokenizer, max_len=MAX_LEN)

    # Train BERT.
    model = train_model(model, device, input_ids, attention_masks, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS,
                       random_seed=random_seed, random_state=random_seed)
    
    return model

def adaptation(model, tokenizer, device, adaptation_set, test_set, random_seed=RANDOM_SEED):
    
    # Create test sentence and label lists (adaptation sentences will be created ad-hoc).
    test_sentences = test_set.sentence.values
    test_labels = test_set.label.values
    
    # Get a list of the syntactic phenomena.
    phenomena = adaptation_set.phenomenon.unique().astype(str)

    # Put it in snake_case.
    phenomena = np.char.lower(phenomena)
    phenomena = np.char.replace(phenomena, ' ', '_')

    # Show the results.
    print('Syntactic phenomena to be examined:')
    pprint(phenomena)
    
    # For logging.
    model_results = test_set.copy()
    pred_col = 'model_predictions_seed{}'.format(random_seed)
    conf_col = 'model_confidence_seed{}'.format(random_seed)

    for phenomenon in phenomena:
        
        # Get the corresponding sentences for priming.
        adapt_group = adaptation_set.loc[adaptation_set['phenomenon'] == phenomenon.upper().replace('_', ' ')]
        adapt_sentences = adapt_group.sentence.values
        adapt_labels = adapt_group.label.values
        
        # Clone the base model for adaptation.
        adapted_model = deepcopy(model)
        
        # Prepare the input for training.
        input_ids, attention_masks = prepare_input(adapt_sentences, tokenizer, max_len=MAX_LEN)
        
         
        adapted_model = adapt_model(adapted_model, device, input_ids, attention_masks, adapt_labels,
                                    epochs=EPOCHS, random_seed=random_seed, random_state=random_seed)
        
        # Testing.
        adapt_results = test(adapted_model, tokenizer, device, test_set)
        
        ## Logging.
        # Store into labeled column in the dataframe.
        col_name = '_'.join(['bert', 'adapted', 'on', phenomenon])
        model_results[col_name + '_predictions_seed{}'.format(random_seed)] = adapt_results[pred_col].copy()
        model_results[col_name + '_confidence_seed{}'.format(random_seed)] = adapt_results[conf_col].copy()

    return model_results


def test(model, tokenizer, device, test_set):
    global MAX_LEN, BATCH_SIZE
    
    # Split the dataframes into sentences and labels.
    test_sentences = test_set.sentence.values
    test_labels = test_set.label.values
        
    # Prepare the test sentences for predictions.
    input_ids, attention_masks = prepare_input(test_sentences, tokenizer, max_len=MAX_LEN)

    # Get predictions from BERT.
    predictions, confidence = get_predictions(model, device, input_ids, attention_masks, test_labels,
                                                batch_size=BATCH_SIZE)
    
    # Logging: Store into labeled column in the dataframe.
    model_results = test_set.copy()
    pred_col = 'model_predictions_seed{}'.format(RANDOM_SEED)
    conf_col = 'model_confidence_seed{}'.format(RANDOM_SEED)
    model_results[pred_col] = predictions
    model_results[conf_col] = confidence
    
    return model_results

def get_model_path():
        ## Tell python that I want to deal with the constants at the top of the file.
        global MODEL, TRAIN_DIR, MODEL_DIR, SCRAMBLE_TRAIN, RANDOM_SEED

        # Get the name of the training dataset.
        if 'cola' in TRAIN_DIR:
            dataset = 'cola'
            if 'repeated' in TRAIN_DIR:
                if 'ungrammatical' in TRAIN_DIR:
                    dataset = 'repeated_ungramm_cola'
                else:
                    dataset = 'repeated_gramm_cola'
        elif 'pepsi' in TRAIN_DIR:
            dataset = 'pepsi'
        else:
            print('No valid dataset provided. \nTerminating program execution.')
            print(TRAIN_DIR)
            exit()

        # To scramble or not to scramble.
        if SCRAMBLE_TRAIN:
            data_type = 'scrambled'
        else:
            data_type = 'control'

        # Pretrained vs. untrained BERT.
        if MODEL == 'None':
            model_name = 'bert-untrained'
        else:
            model_name = 'bert-pretrained'
        
        model_name = '-'.join([model_name, dataset, data_type, 'seed{}'.format(RANDOM_SEED)])
        model_path = os.path.join(MODEL_DIR, model_name)

        return model_path

def load_model():
    ## Tell python that I want to deal with the constants at the top of the file.
    global MODEL, TRAIN_DIR


    if TRAIN_DIR:
        # Get the name of the training dataset.
        model_path = get_model_path()

        if os.path.exists(model_path):
            print('Using model saved at:', model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(model_path)
            print('Model loaded succesfully!  Skipping training...')
            TRAIN_DIR = False
            return model, tokenizer
    try:
        model = BertForSequenceClassification.from_pretrained(MODEL)
        tokenizer = BertTokenizer.from_pretrained(MODEL)
    except:
        print('No model provided!  Loading basic BERT model without pretraining...')
        model = BertForSequenceClassification(BertConfig())
        print('Using tokenizer for bert-base-uncased...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return model, tokenizer
    

def main(argv):
    
    # Get command line arguments.
    get_args(argv)

    ##  Hardware setup.
    # TensorFlow GPU setup.
    device_name = tf.test.gpu_device_name()
    print(device_name)
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        raise SystemError('GPU device not found')

    # Torch GPU setup.
    if torch.cuda.is_available():  # Tell PyTorch to use the GPU. 
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
    ## Load the model.
    model, tokenizer = load_model()

    # Send the model to my nice GPU.
    model.to(device)
    
    # Load the test set.
    # NOTE: It is mandatory to provide a test set.
    test_set = pd.read_csv(TEST_DIR, index_col=0)
    test_set = test_set.sample(frac=1, random_state=RANDOM_SEED) # Preserve indices for pandas join() operation below.
    
    print('Number of test sentences: {:,}\n'.format(test_set.shape[0]))
    print(test_set.sample(10))
    
    
    ## Training.
    if TRAIN_DIR:
        
        # Load the data & shuffle the rows.
        training_set = pd.read_csv(TRAIN_DIR, index_col=0)
        training_set = training_set.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        # Scramble the data if required. #eliminate punctuation and only use lowercase
        if SCRAMBLE_TRAIN: training_set.sentence = training_set.sentence.apply(
            lambda x: ' '.join(np.random.RandomState(seed=RANDOM_SEED).permutation(
                re.findall(r"[\w]+|[^\s\w]", x))))
        
    
        print('Number of training sentences: {:,}\n'.format(training_set.shape[0]))
        print(training_set.sample(10))
        
        # Train the model.
        print('Now training the model...')
        model = training(model, tokenizer, device, training_set, random_seed=RANDOM_SEED)
        print('Training completed successfully!')

        
        # If a model was trained, it was not previously saved in the .../bert-priming/models/ directory.
        # Below I will save it in order to repeat the training process on future experiments.
        model_out_dir = get_model_path()
        
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_out_dir)
            tokenizer.save_pretrained(model_out_dir)
        else:
            print('Directory', model_out_dir, '\n already exists')
            print('but training executed regardless!  Please verify that this is expected behavior.')
        
        
    ## Adaptation/Priming.
    # NOTE: Program execution will terminate after adaptation test results.
    if ADAPT_DIR:
        
        # Load the data & shuffle the rows.
        adaptation_set = pd.read_csv(ADAPT_DIR, index_col=0)
        adaptation_set.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        # Scramble the data if required.
        if SCRAMBLE_ADAPT: adaptation_set.sentence = adaptation_set.sentence.apply(
            lambda x: ' '.join(np.random.RandomState(seed=RANDOM_SEED).permutation(
                re.findall(r"[\w]+|[^\s\w]", x))))
    
        print('Number of adaptation sentences: {:,}\n'.format(adaptation_set.shape[0]))
        print(adaptation_set.sample(10))
        
        # Perform adaptation routine on the model.
        print('Now priming the model...')
        model_results = adaptation(model, tokenizer, device, adaptation_set, test_set, 
                                   random_seed=RANDOM_SEED)
        
        # Logging.
        print('Adaptation completed successfully!')
        log_results(model_results, output_dir=OUTPUT_DIR)
        exit()
    
    
    ## Testing.
    model_results = test(model, tokenizer, device, test_set)
    
    # Logging.
    print(model_results.head(20))
    log_results(model_results, OUTPUT_DIR)

            
if __name__=="__main__":
    main(sys.argv[1:])

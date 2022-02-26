import datetime
import numpy as np
import plotly.express as px
import random
import time
import torch
from keras.preprocessing.sequence import pad_sequences # Set the maximum sequence length.
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup


def prepare_input(sentences, tokenizer, max_len=64):
    '''
    Tokenizes and applies attention masks to the given sentences using the tokenizer.
    Returns a list of the padded input tokens as well as the attention masks.
    '''
    MAX_LEN = max_len
    
    input_ids = []
    attention_masks = []

    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent, # Sentence to encode.
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            )
    
        input_ids.append(encoded_sent)

    # Pad our input tokens.
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
                            dtype="long", truncating="post", padding="post")
    # Create attention masks.

    # Create a mask of 1s for each token followed by 0s for padding.
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
        
    return input_ids, attention_masks

def get_predictions(model, device, input_ids, attention_masks, labels, batch_size=32):
    '''
    Computes trained BERT's predictions on the given dataset.  The data must already be tokenized
    and have the attention masks applied to it before handing off to this function.
    Returns an array of prediction labels and their corresponding confidence scores in another array.
    The confidence scores are obtained by applying a softmax to the output logits of the model.
    '''
    
    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)


    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler,
                                       batch_size=batch_size)

    # Prediction on test set
    print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
    # Put model in evaluation mode
    model.eval()
    # Tracking variables 
    predictions , confidence = np.array([]), np.array([])
    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids.to(torch.int64), token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and confidence.
        predictions = np.append(predictions, np.argmax(logits, axis=1))
        confidence = np.append(confidence, softmax(logits, axis=1).max(axis=1))

    print('DONE.')
    
    return predictions, confidence

def train_model(model, device, input_ids, attention_masks, labels, batch_size=32, epochs=4, 
                plot_training_loss=False, random_seed=42, random_state=2018):
    ## Set the seed value all over the place to make this reproducible.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    ## Training (90%) and Validation (10%) Split.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        input_ids, labels, random_state=random_state, test_size=0.1)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks, labels, random_state=random_state, test_size=0.1)


    ## Converting to PyTorch Data Types.
    # Convert all inputs and labels into torch tensors (required by BERT).
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Create an iterator for the dataset from torch DataLoader class for memory-efficient training.

    # Create the DataLoader for the training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for the validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler,
                                       batch_size=batch_size)

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    ## Hyperparameters and Optimizer.
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                       lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                       eps = 1e-8 # args.adam_epsilon - default is 1e-8.
                       )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                 num_warmup_steps = 0, # Default value in run_glue.py
                                                 num_training_steps = total_steps)

    ## Training loop.

    # Helper function to calculate accuracy of predictions vs. labels.
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    # Helper function for formatting elapsed times.
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    '''
    Training code based on 'run_glue.py' script from the following url:

    https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    '''

    # Store the average loss after each epoch for plotting.
    loss_values = []
    for epoch_i in range(0, epochs):
        '''
        ===================================
        +           Training              +
        ===================================
        '''

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...') 

        # Measure how long the training epochs take.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode.
        model.train() # NOTE: this puts the model into training mode.  It does NOT perform training.

        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print(' Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, 
                                                                      len(train_dataloader), elapsed)) 

            # Unpack this training batch from our dataloader and copy it to the GPU. 
            b_input_ids = batch[0].to(device).to(torch.int64)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear any previously calculated gradients before the next backward pass.
            model.zero_grad() # PyTorch does not do it automatically.

            # Perform a forward pass.
            outputs = model(b_input_ids,          # This will return the loss because the labels
                            token_type_ids=None,  # were given as a parameter.
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            loss = outputs[0] 

            # Accumulate the training loss over all batches to average.
            total_loss += loss.item()

            # Perform backward pass to calculate the gradients.
            loss.backward()

            # Clip norm of gradients to 1.0 to avoid the exploding gradients problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a gradient step.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Record the loss for plotting.
        loss_values.append(avg_train_loss)

        print("")
        print(" Average training loss: {0:.2f}".format(avg_train_loss))
        print(" Training epoch took: {:}".format(format_time(time.time() - t0)))

        '''
        ===================================
        +           Validation            +
        ===================================
        '''
        # Measure performance on the validation set.
        print("")
        print("Running Validation...") 
        t0 = time.time() 

        # Put the model in evaluation mode--the dropout layers behave differently here.
        model.eval()

        # Tracking variables.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0 

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU.
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from the dataloader.
            b_input_ids, b_input_mask, b_labels = batch

            # Tell the model not to compute gradients.
            with torch.no_grad():
                outputs = model(b_input_ids.to(torch.int64),         # This will return the logits instead of loss
                                token_type_ids=None, # because the labels weren't provided.
                                attention_mask=b_input_mask)

            # Get the logits output by the model.
            # NOTE: Logits = output values before applying activation function (i.e. softmax)
            logits = outputs[0]
            # Move logits and labels to my CPU, which is also nice.
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate teh accuracy for this batch of the test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the toal accuracy.
            eval_accuracy += tmp_eval_accuracy 
            # Track the number of batches.
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print(" Validation took: {:}".format(format_time(time.time() - t0)))
        print("")

    print("Training complete!")

    if plot_training_loss:
        ## Plot the training loss over all batches.
        f = pd.DataFrame(loss_values)
        f.columns = ['loss']
        fig = px.line(f, x=f.index, y=f.loss)
        fig.update_layout(title='Training loss of the Model',
                           xaxis_title='Epoch',
                           yaxis_title='Loss')
        fig.show()
    
    return model

def adapt_model(model, device, input_ids, attention_masks, labels, epochs=4, batch_size=20,
                plot_training_loss=False, random_seed=42, random_state=2018):
    '''Same as train_model(), but because the training set is expected to be very small (on the order of 20 sentences...), there will be no split between the training and validation set and the batch size is set to 1.'''
    
    ## Set the seed value all over the place to make this reproducible.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    ## Rename variables for compatibility.
    train_inputs, train_labels = input_ids, labels
    # Do the same for the masks.
    train_masks, _ = attention_masks, labels

    ## Converting to PyTorch Data Types.
    # Convert all inputs and labels into torch tensors (required by BERT).
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)

    train_masks = torch.tensor(train_masks)

    # Create an iterator for the dataset from torch DataLoader class for memory-efficient training.

    # Create the DataLoader for the training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Get all of the model's parameters as a list of tuples.
    ''' Note: Too much output.  Will only print during training.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    '''
    ## Hyperparameters and Optimizer.
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                       lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                       eps = 1e-8 # args.adam_epsilon - default is 1e-8.
                       )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                 num_warmup_steps = 0, # Default value in run_glue.py
                                                 num_training_steps = total_steps)

    ## Training loop.

    # Helper function to calculate accuracy of predictions vs. labels.
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    # Helper function for formatting elapsed times.
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    '''
    Training code based on 'run_glue.py' script from the following url:

    https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    '''

    # Store the average loss after each epoch for plotting.
    loss_values = []
    for epoch_i in range(0, epochs):
        '''
        ===================================
        +           Adaptation              +
        ===================================
        '''

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...') 

        # Measure how long the training epochs take.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode.
        model.train() # NOTE: this puts the model into training mode.  It does NOT perform training.

        for step, batch in enumerate(train_dataloader):
                
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step + 1, 
                                                                  len(train_dataloader), elapsed)) 

            # Unpack this training batch from our dataloader and copy it to the GPU. 
            b_input_ids = batch[0].to(device).to(torch.int64)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear any previously calculated gradients before the next backward pass.
            model.zero_grad() # PyTorch does not do it automatically.

            # Perform a forward pass.
            outputs = model(b_input_ids,          # This will return the loss because the labels
                            token_type_ids=None,  # were given as a parameter.
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            loss = outputs[0] 

            # Accumulate the training loss over all batches to average.
            total_loss += loss.item()

            # Perform backward pass to calculate the gradients.
            loss.backward()

            # Clip norm of gradients to 1.0 to avoid the exploding gradients problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a gradient step.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Record the loss for plotting.
        loss_values.append(avg_train_loss)

        print("")
        print(" Average adaptation loss: {0:.2f}".format(avg_train_loss))
        print(" Adaptation epoch took: {:}".format(format_time(time.time() - t0)))

       

    print("Adaptation complete!")

    if plot_training_loss:
        ## Plot the training loss over all batches.
        f = pd.DataFrame(loss_values)
        f.columns = ['loss']
        fig = px.line(f, x=f.index, y=f.loss)
        fig.update_layout(title='Adaptation loss of the Model',
                           xaxis_title='Epoch',
                           yaxis_title='Loss')
        fig.show()
    
    return model
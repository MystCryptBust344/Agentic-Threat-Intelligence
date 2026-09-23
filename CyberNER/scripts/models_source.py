import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import BertModel, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils import data
from sklearn.model_selection import train_test_split
import time
import argparse
import json
from tqdm import tqdm

# Configuration class to hold all parameters
class Config:
    def __init__(self, args):
        # Model parameters
        self.dataset_source = args.dataset_source
        self.model_type = args.model_type
        self.max_seq_length = args.max_seq_length
        self.batch_size = args.batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.total_train_epochs = args.epochs
        self.output_dir = args.output_dir
        self.learning_rate = args.learning_rate
        self.lr_crf_fc = args.lr_crf_fc
        self.weight_decay_crf_fc = args.weight_decay_crf_fc
        self.weight_decay_finetune = args.weight_decay_finetune
        self.warmup_proportion = args.warmup_proportion
        self.early_stopping_patience = args.early_stopping_patience
        self.checkpoint_freq = args.checkpoint_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = args.seed
        self.max_grad_norm = args.max_grad_norm
        self.test_size = args.test_size
        self.val_size = args.val_size
        self.num_workers = args.num_workers
        self.dataset_path = args.dataset_path
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

# Data processing classes
class InputExample:
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids

def prepare_dataset(df, source):
    # Filter by source
    filtered_df = df[df['Source'] == source].copy()
    
    # Group by sentence ID to reconstruct sentences
    grouped = filtered_df.groupby('Sentence_ID')
    examples = []
    
    for idx, (sentence_id, group) in enumerate(grouped):
        words = group['Word'].tolist()
        labels = group['STIX_Tag'].tolist()
        examples.append(InputExample(guid=idx, words=words, labels=labels))
    
    return examples

def example2feature(example, tokenizer, label_map, max_seq_length):
    add_label = 'X'
    tokens = ['[CLS]']
    predict_mask = [0]
    label_ids = [label_map['[CLS]']]
    
    for i, w in enumerate(example.words):
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        
        tokens.extend(sub_words)
        for j in range(len(sub_words)):
            if j == 0:
                predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
            else:
                predict_mask.append(0)
                label_ids.append(label_map[add_label])

    # Truncate if too long
    if len(tokens) > max_seq_length - 1:
        print(f'Example No.{example.guid} is too long, length is {len(tokens)}, truncated to {max_seq_length}!')
        tokens = tokens[0:(max_seq_length - 1)]
        predict_mask = predict_mask[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]
    
    tokens.append('[SEP]')
    predict_mask.append(0)
    label_ids.append(label_map['[SEP]'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        predict_mask=predict_mask,
        label_ids=label_ids
    )

class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(self.examples[idx], self.tokenizer, self.label_map, self.max_seq_length)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids

    @classmethod
    def pad(cls, batch):
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
        input_ids_list = torch.LongTensor(f(0, maxlen))
        input_mask_list = torch.LongTensor(f(1, maxlen))
        segment_ids_list = torch.LongTensor(f(2, maxlen))
        predict_mask_list = torch.BoolTensor(f(3, maxlen))
        label_ids_list = torch.LongTensor(f(4, maxlen))

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list

# Model utility functions
def log_sum_exp_batch(log_tensor, axis=-1):
    return torch.max(log_tensor, axis)[0] + torch.log(
        torch.exp(log_tensor - torch.max(log_tensor, axis)[0].view(log_tensor.shape[0], -1, 1)).sum(axis)
    )

def calculate_metrics(y_true, y_pred, idx2label=None):
    """
    Calculate precision, recall, and F1 score. Ignore O (class id 3)
    Also calculate class-wise metrics and return them as a dictionary
    """
    ignore_id = 3  # O tag

    num_proposed = len(y_pred[y_pred > ignore_id])
    num_correct = (np.logical_and(y_true == y_pred, y_true > ignore_id)).sum()
    num_gold = len(y_true[y_true > ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    # Class-wise metrics
    class_metrics = {}
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    for cls in unique_classes:
        if cls <= ignore_id:  # Skip [CLS], [SEP], X, O
            continue
            
        cls_y_true = (y_true == cls)
        cls_y_pred = (y_pred == cls)
        
        tp = np.logical_and(cls_y_true, cls_y_pred).sum()
        fp = np.logical_and(~cls_y_true, cls_y_pred).sum()
        fn = np.logical_and(cls_y_true, ~cls_y_pred).sum()
        
        try:
            cls_precision = tp / (tp + fp)
        except ZeroDivisionError:
            cls_precision = 0.0
            
        try:
            cls_recall = tp / (tp + fn)
        except ZeroDivisionError:
            cls_recall = 0.0
            
        try:
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall)
        except ZeroDivisionError:
            cls_f1 = 0.0
            
        class_metrics[cls] = {
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1,
            'support': cls_y_true.sum()
        }

    return precision, recall, f1, class_metrics

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def evaluate(model, dataloader, epoch, dataset_name, idx2label=None):
    model.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    start = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(model.device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            
            _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
            
            valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)
            
            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    accuracy = correct / total
    precision, recall, f1, class_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), idx2label)
    end = time.time()
    
    print(f'Epoch:{epoch}, Acc:{100.*accuracy:.2f}, Precision: {100.*precision:.2f}, '
          f'Recall: {100.*recall:.2f}, F1: {100.*f1:.2f} on {dataset_name}, '
          f'Spend:{(end-start)/60.0:.3f} minutes for evaluation')
    print('--------------------------------------------------------------')
    
    # Print per-class metrics if verbose and idx2label is provided
    if idx2label:
        print("Per-class metrics:")
        for cls_id, metrics in class_metrics.items():
            if cls_id in idx2label:
                label = idx2label[cls_id]
                print(f"{label}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, support={metrics['support']}")
    
    return accuracy, f1, class_metrics

# BERT-CRF Model
class BERT_CRF_NER(nn.Module):
    def __init__(self, model_type, start_label_id, stop_label_id, num_labels, batch_size, device):
        super(BERT_CRF_NER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.device = device
        self.model_type = model_type

        # Initialize BERT/RoBERTa model
        model_type_lower = model_type.lower()
        if 'bert-' in model_type_lower:
            self.encoder = BertModel.from_pretrained(model_type)
        elif 'roberta' in model_type_lower:
            self.encoder = AutoModel.from_pretrained(model_type)
        elif 'securebert' in model_type_lower:
            self.encoder = AutoModel.from_pretrained("ehsanaghaei/SecureBERT")
        elif 'darkbert' in model_type_lower:
            self.encoder = AutoModel.from_pretrained("s2w-ai/DarkBERT", token=os.getenv("HF_TOKEN"))
        elif 'cysecbert' in model_type_lower:
            self.encoder = AutoModel.from_pretrained("markusbayer/CySecBERT")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.dropout = nn.Dropout(0.2)
        # Maps the output of encoder to label space
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)

        # CRF transition matrix
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))
        
        # Constraints: never transfer to start tag or from stop tag
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        # Initialize weights
        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def _forward_alg(self, feats):
        '''
        Forward algorithm to calculate all possible paths
        '''
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # Initialize alpha with -10000 for all tags except start_tag
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_alpha[:, 0, self.start_label_id] = 0
        
        # Iterate through timesteps
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
            
        # Final forward computation for all possible tag paths
        log_prob_all_tags = log_sum_exp_batch(log_alpha)
        return log_prob_all_tags

    def _get_encoder_features(self, input_ids, segment_ids, input_mask):
        '''
        Extract features from encoder model
        '''
        model_type_lower = self.model_type.lower()
        if 'bert-' in model_type_lower or 'securebert' in model_type_lower or 'darkbert' in model_type_lower or 'cysecbert' in model_type_lower:
            encoder_seq_out = self.encoder(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
        else:  # RoBERTa doesn't use token_type_ids
            encoder_seq_out = self.encoder(input_ids, attention_mask=input_mask)[0]
            
        encoder_seq_out = self.dropout(encoder_seq_out)
        encoder_feats = self.hidden2label(encoder_seq_out)
        return encoder_feats
        
    def _score_sentence(self, feats, label_ids):
        '''
        Calculate score for the given tag sequence
        '''
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(self.device)
        
        # Add transition scores and emission scores
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t] * self.num_labels + label_ids[:, t-1]).view(-1, 1)) + \
                feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
                
        return score

    def _viterbi_decode(self, feats):
        '''
        Viterbi algorithm to find the best path
        '''
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # Initialize with start tag
        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0
        
        # Track best previous tag
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)
        
        for t in range(1, T):
            # Find max score and best tag for each position
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # Backtrack to find the best path
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)
        max_score, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            path[:, t] = psi[:, t+1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()

        return max_score, path

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        '''
        Calculate negative log likelihood loss
        '''
        encoder_feats = self._get_encoder_features(input_ids, segment_ids, input_mask)
        # Forward score - score of all possible paths
        forward_score = self._forward_alg(encoder_feats)
        # Gold score - score of the actual tag sequence
        gold_score = self._score_sentence(encoder_feats, label_ids)
        # Loss = log(sum(exp(all_paths))) - score(gold_path)
        return torch.mean(forward_score - gold_score)

    def forward(self, input_ids, segment_ids, input_mask):
        '''
        Forward pass for prediction
        '''
        encoder_feats = self._get_encoder_features(input_ids, segment_ids, input_mask)
        score, tag_seq = self._viterbi_decode(encoder_feats)
        return score, tag_seq

def predict_entities(model, tokenizer, text, label_map, idx2label, max_seq_length):
    """
    Predicts entities in a text using a trained BERT-CRF model
    """
    model.eval()
    
    # Process input text
    words = text.split()
    predict_examples = [InputExample(guid=0, words=words, labels=['O'] * len(words))]
    
    # Convert to features
    dataset = NerDataset(predict_examples, tokenizer, label_map, max_seq_length)
    dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=NerDataset.pad)
    
    # Get predictions
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(model.device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, _ = batch
            
            _, tag_seq = model(input_ids, segment_ids, input_mask)
    
    # Extract valid predictions (excluding special tokens)
    valid_predictions = []
    for i, mask in enumerate(predict_mask[0].cpu().numpy()):
        if mask:
            tag_id = tag_seq[0][i].item()
            valid_predictions.append(idx2label[tag_id])
    
    # Process predictions into entities
    entities = []
    current_entity = None
    
    for i, (word, tag) in enumerate(zip(words, valid_predictions)):
        if tag.startswith('B-'):
            # End previous entity if there was one
            if current_entity:
                entities.append(current_entity)
            
            # Start new entity
            entity_type = tag[2:]  # Remove B- prefix
            current_entity = {
                'text': word,
                'start': i,
                'end': i,
                'type': entity_type
            }
        elif tag.startswith('I-') and current_entity and tag[2:] == current_entity['type']:
            # Continue current entity
            current_entity['text'] += ' ' + word
            current_entity['end'] = i
        elif current_entity:
            # End current entity when tag changes
            entities.append(current_entity)
            current_entity = None
    
    # Add final entity if exists
    if current_entity:
        entities.append(current_entity)
    
    return entities

def train(config):
    # Set environment variables for tokenizers parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load dataset
    print(f"Loading dataset from {config.dataset_path}")
    df = pd.read_csv(config.dataset_path, keep_default_na=False)
    print(f"Dataset loaded with {len(df)} rows")
    
    # Create label map from dataset
    unique_labels = ['X', '[CLS]', '[SEP]', 'O'] + sorted(list(set([tag for tag in df['STIX_Tag'].unique() if tag != 'O'])))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    idx2label = {i: label for i, label in enumerate(unique_labels)}
    print(f"Created label map with {len(unique_labels)} labels")
    
    # Special label IDs
    start_label_id = label_map['[CLS]']
    stop_label_id = label_map['[SEP]']
    
    # Filter to get source-specific data
    source_df = df[df['Source'] == config.dataset_source]
    print(f"Filtered to {config.dataset_source} data with {len(source_df)} rows")
    
    # Split data into sentences
    examples = prepare_dataset(source_df, config.dataset_source)
    print(f"Created {len(examples)} sentence examples")
    
    # Split into train/valid/test sets
    train_examples, test_examples = train_test_split(examples, test_size=config.test_size, random_state=config.seed)
    train_examples, dev_examples = train_test_split(train_examples, test_size=config.val_size, random_state=config.seed)
    print(f"Split into {len(train_examples)} train, {len(dev_examples)} dev, {len(test_examples)} test examples")
    
    # Load tokenizer based on model type
    if 'bert-' in config.model_type.lower():
        tokenizer = AutoTokenizer.from_pretrained(config.model_type, do_lower_case=False)
    elif 'roberta' in config.model_type.lower():
        tokenizer = AutoTokenizer.from_pretrained(config.model_type)
    elif 'securebert' in config.model_type.lower():
        tokenizer = AutoTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
    elif 'darkbert' in config.model_type.lower():
        tokenizer = AutoTokenizer.from_pretrained("s2w-ai/DarkBERT", token=os.getenv("HF_TOKEN"))
    elif 'cysecbert' in config.model_type.lower():
        tokenizer = AutoTokenizer.from_pretrained("markusbayer/CySecBERT")
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    # Create datasets
    train_dataset = NerDataset(train_examples, tokenizer, label_map, config.max_seq_length)
    dev_dataset = NerDataset(dev_examples, tokenizer, label_map, config.max_seq_length)
    test_dataset = NerDataset(test_examples, tokenizer, label_map, config.max_seq_length)
    
    # Create data loaders
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=NerDataset.pad
    )
    
    dev_dataloader = data.DataLoader(
        dataset=dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=NerDataset.pad
    )
    
    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=NerDataset.pad
    )
    
    total_train_steps = int(len(train_examples) / config.batch_size / config.gradient_accumulation_steps * config.total_train_epochs)
    
    print("***** Training information *****")
    print(f"  Num examples = {len(train_examples)}")
    print(f"  Batch size = {config.batch_size}")
    print(f"  Num steps = {total_train_steps}")
    
    # Initialize model
    model = BERT_CRF_NER(
        model_type=config.model_type,
        start_label_id=start_label_id,
        stop_label_id=stop_label_id,
        num_labels=len(unique_labels),
        batch_size=config.batch_size,
        device=config.device
    )
    model.to(config.device)
    
    # Configure optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer 
                    if not any(nd in n for nd in no_decay) and not any(nd in n for nd in new_param)], 
            'weight_decay': config.weight_decay_finetune},
        {'params': [p for n, p in param_optimizer 
                    if any(nd in n for nd in no_decay) and not any(nd in n for nd in new_param)], 
            'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if n in ('transitions', 'hidden2label.weight')],
            'lr': config.lr_crf_fc, 
            'weight_decay': config.weight_decay_crf_fc},
        {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'],
            'lr': config.lr_crf_fc, 
            'weight_decay': 0.0}
    ]
    
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_train_steps * config.warmup_proportion),
        num_training_steps=total_train_steps
    )
    
    # Training loop
    global_step = 0
    best_valid_f1 = 0.0
    early_stopping_counter = 0
    best_epoch = 0
    
    # Create a directory for saving checkpoints
    checkpoint_dir = os.path.join(config.output_dir, f"{config.dataset_source}_{config.model_type.replace('/', '_')}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'valid_acc': [], 
        'valid_f1': [],
        'best_valid_f1': 0.0
    }
    
    # Main training loop
    for epoch in range(config.total_train_epochs):
        model.train()
        tr_loss = 0
        train_start = time.time()
        
        # Progress bar for batches
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(train_iterator):
            batch = tuple(t.to(config.device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            
            # Forward pass
            neg_log_likelihood = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)
            
            # Loss scaling for gradient accumulation
            if config.gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / config.gradient_accumulation_steps
            
            # Backward pass
            neg_log_likelihood.backward()
            tr_loss += neg_log_likelihood.item()
            
            # Update parameters
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Update progress bar
                train_iterator.set_postfix({'loss': tr_loss / (step + 1)})
        
        # Calculate average training loss for the epoch
        avg_train_loss = tr_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        # Print training stats
        print('--------------------------------------------------------------')
        print(f"Epoch:{epoch+1}/{config.total_train_epochs} completed, Total training's Loss: {tr_loss}, "
              f"Spend: {(time.time() - train_start)/60.0:.2f}m")
        
        # Evaluate on dev set
        valid_acc, valid_f1, _ = evaluate(model, dev_dataloader, epoch, 'Valid_set', idx2label)
        history['valid_acc'].append(valid_acc)
        history['valid_f1'].append(valid_f1)
        
        # Save checkpoint if it's the best model so far
        if valid_f1 > best_valid_f1:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'valid_acc': valid_acc,
                'valid_f1': valid_f1,
                'config': vars(config),
                'label_map': label_map,
                'idx2label': idx2label,
                'max_seq_length': config.max_seq_length,
            }, os.path.join(checkpoint_dir, f'best_model.pt'))
            
            best_valid_f1 = valid_f1
            history['best_valid_f1'] = best_valid_f1
            early_stopping_counter = 0
            best_epoch = epoch
            print(f"New best model saved with F1: {valid_f1:.4f}")
        else:
            early_stopping_counter += 1
            print(f"No improvement over previous best F1: {best_valid_f1:.4f}, counter: {early_stopping_counter}")
            
        # Save periodic checkpoint
        if (epoch + 1) % config.checkpoint_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'valid_acc': valid_acc,
                'valid_f1': valid_f1,
                'config': vars(config),
                'label_map': label_map,
                'idx2label': idx2label,
                'max_seq_length': config.max_seq_length,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            
            # Save training history
            with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f)
                
        # Early stopping
        if early_stopping_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_counter} epochs without improvement")
            break
            
    # Final evaluation on test set
    print("\n***** Final Evaluation *****")
    print(f"Loading best model from epoch {best_epoch}")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'),weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    test_acc, test_f1, class_metrics = evaluate(model, test_dataloader, "Final", 'Test_set', idx2label)
    
    # Save detailed test results
    test_results = {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        # With this
        'class_metrics': {idx2label[k]: {
            'precision': float(v['precision']),
            'recall': float(v['recall']),
            'f1': float(v['f1']),
            'support': int(v['support'])
        } for k, v in class_metrics.items() if k in idx2label},
        'best_epoch': best_epoch,
        'best_valid_f1': best_valid_f1
    }
    
    with open(os.path.join(checkpoint_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
        
    print(f"Final test results - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"Results saved to {os.path.join(checkpoint_dir, 'test_results.json')}")

    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Tokenizer saved to {checkpoint_dir}")
    
    return model, tokenizer, label_map, idx2label

def main():
    """
    Main function to parse arguments and start training
    """
    parser = argparse.ArgumentParser(description='BERT-CRF for Named Entity Recognition')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, 
                        default='/home/yasir.ech-chammakhy/lustre/cyber_cc-lcbfvhtc9qm/users/yasir.ech-chammakhy/CyberNER/dataset/cyberner_combined_stix.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--dataset_source', type=str, default='DNRTI', 
                        choices=['DNRTI', 'APTNER', 'CyNER', 'Attacker'],
                        help='Source dataset to use')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='bert-base-cased',
                        help='Type of model to use (bert-base-cased, roberta-base, SecureBERT, DarkBERT)')
    parser.add_argument('--max_seq_length', type=int, default=256,
                        help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for the encoder')
    parser.add_argument('--lr_crf_fc', type=float, default=8e-5,
                        help='Learning rate for CRF and fully connected layers')
    parser.add_argument('--weight_decay_finetune', type=float, default=1e-5,
                        help='Weight decay for fine-tuning')
    parser.add_argument('--weight_decay_crf_fc', type=float, default=5e-6,
                        help='Weight decay for CRF and fully connected layers')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='Proportion of training steps for learning rate warmup')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for gradient clipping')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                        help='Save checkpoint every n epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Other parameters
    parser.add_argument('--output_dir', type=str, default='./outputs/',
                        help='Output directory for saving models and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create config object
    config = Config(args)
    
    # Print configuration
    print("***** Configuration *****")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")
    
    # Start training
    train(config)

if __name__ == "__main__":
    print("Starting BERT-CRF NER training script...")
    main()


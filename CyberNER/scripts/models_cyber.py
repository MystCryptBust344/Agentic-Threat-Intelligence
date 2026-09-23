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
import copy # Needed for deep copying examples (though maybe not strictly required here)

# Define the sources explicitly for evaluation
SOURCES_TO_EVALUATE = ['APTNER', 'CyNER', 'Attacker', 'DNRTI']

# --- Configuration ---
class Config:
    def __init__(self, args):
        # Model parameters
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
        self.val_size = args.val_size # Proportion of *original* data for validation
        self.num_workers = args.num_workers
        self.dataset_path = args.dataset_path

        # Create output directory (base dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

# --- Data Processing ---
class InputExample:
    """A single training/test example for token classification."""
    def __init__(self, guid, words, labels, source):
        self.guid = guid
        self.words = words
        self.labels = labels
        self.source = source # Store the source dataset name

class InputFeatures:
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask # Mask indicating the first subword of each original word
        self.label_ids = label_ids

def prepare_unified_dataset(df):
    """
    Processes the entire DataFrame to create InputExamples for all sentences,
    storing the source for each example.
    """
    grouped = df.groupby('Sentence_ID')
    examples = []

    print(f"Processing {len(grouped)} unique sentences from the unified dataset...")
    for sentence_id, group in tqdm(grouped, desc="Preparing Examples"):
        if group.empty:
            print(f"Warning: Sentence_ID {sentence_id} resulted in an empty group. Skipping.")
            continue

        words = group['Word'].tolist()
        labels = group['STIX_Tag'].tolist()
        # Ensure source is consistent (usually is when grouped by sentence)
        source = group['Source'].iloc[0] if not group['Source'].empty else 'Unknown'
        if group['Source'].nunique() > 1:
             print(f"Warning: Sentence_ID {sentence_id} has multiple sources ({group['Source'].unique()}). Using first: {source}")

        guid = f"sentence_{sentence_id}" # Use sentence ID for GUID
        examples.append(InputExample(guid=guid, words=words, labels=labels, source=source))

    print(f"Created {len(examples)} sentence examples.")
    return examples

def example2feature(example, tokenizer, label_map, max_seq_length):
    """Converts a single `InputExample` into an `InputFeatures`."""
    add_label = 'X' # Label for subsequent subword tokens and padding-related tokens
    tokens = []
    label_ids = []
    predict_mask = [] # 1 for the first subword token of a word, 0 otherwise

    # --- [CLS] token ---
    tokens.append('[CLS]')
    label_ids.append(label_map.get('[CLS]', label_map['O'])) # Default to 'O' if specific CLS label not in map
    predict_mask.append(0) # Do not predict for [CLS]

    # --- Word tokens ---
    for i, word in enumerate(example.words):
        word_tokenized = tokenizer.tokenize(str(word)) # Ensure word is string
        if not word_tokenized:
            word_tokenized = ['[UNK]'] # Handle empty tokenization

        tokens.extend(word_tokenized)

        # Assign label to the *first* subword token, 'X' to others
        label = example.labels[i]
        for j, sub_token in enumerate(word_tokenized):
            if j == 0:
                label_ids.append(label_map.get(label, label_map['O'])) # Handle unknown labels
                predict_mask.append(1) # Predict for this token
            else:
                label_ids.append(label_map.get(add_label, label_map['O']))
                predict_mask.append(0) # Do not predict for subsequent subwords

    # --- Truncation ---
    if len(tokens) > max_seq_length - 1: # Account for [SEP]
        # print(f'Example {example.guid} truncated (length {len(tokens)} -> {max_seq_length})')
        tokens = tokens[:(max_seq_length - 1)]
        label_ids = label_ids[:(max_seq_length - 1)]
        predict_mask = predict_mask[:(max_seq_length - 1)]

    # --- [SEP] token ---
    tokens.append('[SEP]')
    label_ids.append(label_map.get('[SEP]', label_map['O']))
    predict_mask.append(0) # Do not predict for [SEP]

    # --- Conversion to IDs ---
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # --- Input Mask & Segment IDs ---
    input_mask = [1] * len(input_ids)   # Mask 1 for real tokens, 0 for padding
    segment_ids = [0] * len(input_ids) # Single sequence

    # --- Padding ---
    # Padding is handled dynamically in the DataLoader's collate_fn

    # --- Sanity Check ---
    assert len(input_ids) == len(input_mask) == len(segment_ids) == len(label_ids) == len(predict_mask), \
        f"Length mismatch in feature creation for GUID {example.guid}"

    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        predict_mask=predict_mask, # Mask for prediction targets
        label_ids=label_ids
    )

class NerDataset(data.Dataset):
    """Dataset wrapping examples and converting to features."""
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_seq_length = max_seq_length
        # Convert examples to features upon initialization
        self.features = self._create_features()

    def _create_features(self):
        features = []
        print(f"Converting {len(self.examples)} examples to features...")
        for example in tqdm(self.examples, desc="Creating Features"):
            try:
                feat = example2feature(example, self.tokenizer, self.label_map, self.max_seq_length)
                if feat is not None:
                    features.append(feat)
                else:
                    print(f"Warning: Failed to convert example GUID {example.guid} to features. Skipping.")
            except Exception as e:
                print(f"Error converting example GUID {example.guid} to features: {e}. Skipping example.")
                # Optionally log the problematic example: print(vars(example))
        print(f"Successfully created {len(features)} features.")
        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        # Return tensors directly if possible, or lists to be converted by collate_fn
        return (
            feat.input_ids,
            feat.input_mask,
            feat.segment_ids,
            feat.predict_mask,
            feat.label_ids
        )

    @staticmethod # Changed to staticmethod as it doesn't use cls state directly
    def pad(batch):
        """Pads sequences within a batch to the maximum sequence length."""
        # Find max sequence length in this batch
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = max(seqlen_list) if seqlen_list else 0

        # Use label_map['O'] or a designated pad ID (0 is common if 'O' is not 0)
        # CRF usually doesn't need a specific ignore_index like CrossEntropy.
        # Using 0 for padding labels is fine as long as it's consistent.
        pad_label_id = 0 # Assuming 0 is a safe padding ID (e.g., not a real class)

        f = lambda x, seqlen, pad_value: [sample[x] + [pad_value] * (seqlen - len(sample[x])) for sample in batch]

        input_ids_list = torch.LongTensor(f(0, maxlen, 0))       # Pad with 0 (tokenizer pad token ID)
        input_mask_list = torch.LongTensor(f(1, maxlen, 0))      # Pad with 0 (Non-attention)
        segment_ids_list = torch.LongTensor(f(2, maxlen, 0))     # Pad with 0
        # Pad predict_mask with 0 (False) - important not to predict for padding
        predict_mask_list = torch.BoolTensor(f(3, maxlen, 0))
        label_ids_list = torch.LongTensor(f(4, maxlen, pad_label_id)) # Pad labels

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list

# --- Model Utilities ---
def log_sum_exp_batch(log_tensor, axis=-1):
    """ Calculates log_sum_exp in a numerically stable way. """
    if log_tensor.nelement() == 0:
        # Handle empty tensor - return tensor of appropriate shape filled with -inf
        out_shape = list(log_tensor.shape)
        if axis is not None:
            del out_shape[axis]
        return torch.full(out_shape, -float('inf'), device=log_tensor.device, dtype=log_tensor.dtype)

    max_score = torch.max(log_tensor, axis, keepdim=True)[0]
    # If max_score is -inf, all values were -inf. The result should be -inf.
    # Avoid subtracting -inf which leads to NaN.
    max_score[torch.isneginf(max_score)] = 0 # Replace -inf with 0 temporarily for subtraction
    sum_exp = torch.exp(log_tensor - max_score).sum(axis, keepdim=True)
    log_sum_exp_val = torch.log(sum_exp) + max_score
    
    # Squeeze the summed axis if keepdim was True
    return log_sum_exp_val.squeeze(axis)


def calculate_metrics(y_true, y_pred, label_map, idx2label=None):
    """
    Calculate overall Precision, Recall, F1, and per-class metrics for NER.
    Ignores 'O', 'X', '[CLS]', '[SEP]' labels based on label_map.

    Args:
        y_true (np.array): True label IDs.
        y_pred (np.array): Predicted label IDs.
        label_map (dict): Mapping from label strings to IDs.
        idx2label (dict, optional): Mapping from IDs to label strings.

    Returns:
        tuple: (overall_precision, overall_recall, overall_f1, class_metrics_dict)
    """
    if idx2label is None:
        idx2label = {v: k for k, v in label_map.items()}

    # Define labels to ignore
    ignore_labels = {'O', 'X', '[CLS]', '[SEP]'}
    ignore_ids = {label_map[label] for label in ignore_labels if label in label_map}

    # --- Overall Metrics Calculation ---
    # Mask for non-ignored true labels
    true_valid_mask = np.isin(y_true, list(ignore_ids), invert=True)
    # Mask for non-ignored predicted labels
    pred_valid_mask = np.isin(y_pred, list(ignore_ids), invert=True)

    num_gold = np.sum(true_valid_mask)       # Total true entities (non-ignored)
    num_proposed = np.sum(pred_valid_mask)   # Total predicted entities (non-ignored)

    # Correct predictions: true == pred AND true is not an ignored label
    correct_mask = np.logical_and(y_true == y_pred, true_valid_mask)
    num_correct = np.sum(correct_mask)

    precision = num_correct / num_proposed if num_proposed > 0 else 0.0
    recall = num_correct / num_gold if num_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- Class-wise Metrics ---
    class_metrics = {}
    unique_relevant_classes = sorted([cls_id for cls_id in idx2label.keys() if cls_id not in ignore_ids])

    for cls_id in unique_relevant_classes:
        cls_label = idx2label.get(cls_id, f"Unknown_{cls_id}")

        # True Positives (TP): Correctly predicted as cls_id
        tp_mask = np.logical_and(y_true == cls_id, y_pred == cls_id)
        tp = np.sum(tp_mask)

        # False Positives (FP): Predicted as cls_id, but true label was different (and not ignored?)
        # Standard: Predicted cls_id, but true label was *not* cls_id.
        fp_mask = np.logical_and(y_true != cls_id, y_pred == cls_id)
        fp = np.sum(fp_mask)

        # False Negatives (FN): True label was cls_id, but predicted differently.
        fn_mask = np.logical_and(y_true == cls_id, y_pred != cls_id)
        fn = np.sum(fn_mask)

        cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0.0
        support = tp + fn # Total number of true instances of this class

        class_metrics[cls_id] = {
            'label': cls_label, # Store readable label
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1,
            'support': support
        }

    return precision, recall, f1, class_metrics


def evaluate(model, dataloader, epoch, dataset_name, label_map, idx2label, device):
    """Evaluates the model on the provided dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    total_tokens_evaluated = 0
    start = time.time()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating on {dataset_name}")
        for batch in pbar:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

            if input_ids.size(0) == 0: continue # Skip empty batches

            # Get predictions (Viterbi decoding happens in model.forward)
            _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask) # (batch_size, seq_len)

            # Use predict_mask to select valid labels and predictions
            # Only evaluate tokens where predict_mask is True (first subword of a word)
            for i in range(input_ids.size(0)): # Iterate through batch items
                seq_len = input_mask[i].sum().item() # Actual length without padding
                valid_indices = predict_mask[i][:seq_len].nonzero(as_tuple=False).squeeze(-1)

                if valid_indices.numel() > 0:
                    valid_labels = label_ids[i][valid_indices]
                    valid_preds = predicted_label_seq_ids[i][valid_indices]

                    all_labels.extend(valid_labels.cpu().numpy())
                    all_preds.extend(valid_preds.cpu().numpy())
                    total_tokens_evaluated += len(valid_labels)

    end = time.time()
    eval_duration = (end - start) / 60.0

    if total_tokens_evaluated == 0:
        print(f"Warning: No valid tokens found for evaluation on {dataset_name}. Returning zero metrics.")
        return 0.0, {} # F1=0, empty class metrics

    # Calculate metrics
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    precision, recall, f1, class_metrics = calculate_metrics(all_labels_np, all_preds_np, label_map, idx2label)

    print(f'\n--- Evaluation Results ({dataset_name} at {epoch}) ---')
    print(f'Overall Precision: {100.*precision:.2f}%')
    print(f'Overall Recall: {100.*recall:.2f}%')
    print(f'Overall F1-Score: {100.*f1:.2f}%')
    print(f'Total Tokens Evaluated (first subword): {total_tokens_evaluated}')
    print(f'Evaluation Time: {eval_duration:.3f} minutes')
    print('--------------------------------------------------------------')

    if idx2label and class_metrics:
        print("Per-class metrics (P, R, F1, Support):")
        sorted_class_metrics = sorted(class_metrics.items(), key=lambda item: item[1]['label'])
        for cls_id, metrics in sorted_class_metrics:
            label = metrics['label']
            print(f"  {label:<20}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, S={metrics['support']}")
        print('--------------------------------------------------------------')

    return f1, class_metrics

# --- BERT-CRF Model ---
class BERT_CRF_NER(nn.Module):
    def __init__(self, model_type, start_label_id, stop_label_id, num_labels, device):
        super(BERT_CRF_NER, self).__init__()
        self.num_labels = num_labels
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.device = device
        self.model_type = model_type # Store model type

        # --- Encoder ---
        try:
            print(f"Initializing encoder: {model_type}")
            model_type_lower = model_type.lower()
            # Use AutoModel for flexibility, specifying known models for clarity/tokens
            if 'securebert' in model_type_lower:
                self.encoder = AutoModel.from_pretrained("ehsanaghaei/SecureBERT")
            elif 'darkbert' in model_type_lower:
                # Ensure you have the token or are logged in via `huggingface-cli login`
                try:
                    self.encoder = AutoModel.from_pretrained("s2w-ai/DarkBERT", use_auth_token=True)
                except Exception as auth_err:
                     print("\n--- DarkBERT Loading Note ---")
                     print("Trying to load DarkBERT. Ensure you have access rights.")
                     print("Consider logging in via `huggingface-cli login` with a token that has access.")
                     print(f"Original error: {auth_err}")
                     print("---")
                     # Optionally fallback or raise specific error
                     raise ValueError(f"Failed to load DarkBERT, potentially due to authentication. Error: {auth_err}")

            elif 'cysecbert' in model_type_lower:
                self.encoder = AutoModel.from_pretrained("markusbayer/CySecBERT")
            else:
                # General case for BERT, RoBERTa, etc.
                self.encoder = AutoModel.from_pretrained(model_type)

        except Exception as e:
             print(f"FATAL: Error loading model '{model_type}' from Hugging Face: {e}")
             raise # Stop execution if base model fails to load

        self.hidden_size = self.encoder.config.hidden_size
        print(f"Encoder loaded. Hidden size: {self.hidden_size}")

        # --- Classifier and Dropout ---
        self.dropout = nn.Dropout(0.2) # Configurable?
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)

        # --- CRF Layer ---
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))
        # Constraints: no transition *to* START, no transition *from* STOP
        self.transitions.data[start_label_id, :] = -10000.0
        self.transitions.data[:, stop_label_id] = -10000.0
        # Optional: Add constraints related to PAD if you have a dedicated PAD label ID

        # --- Initialization ---
        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def _get_encoder_features(self, input_ids, segment_ids, input_mask):
        """ Pass data through encoder and classifier head to get emission scores. """
        model_type_lower = self.model_type.lower()
        # RoBERTa doesn't use token_type_ids (segment_ids)
        if 'roberta' in model_type_lower:
             encoder_output = self.encoder(input_ids=input_ids, attention_mask=input_mask)
        else:
             # BERT, SecureBERT, DarkBERT, CySecBERT typically use them
             encoder_output = self.encoder(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        # Extract last hidden state sequence
        if hasattr(encoder_output, 'last_hidden_state'):
            sequence_output = encoder_output.last_hidden_state
        else: # Handle older tuple output format
            sequence_output = encoder_output[0]

        sequence_output = self.dropout(sequence_output)
        emission_scores = self.hidden2label(sequence_output) # Shape: (batch_size, seq_len, num_labels)
        return emission_scores

    def _forward_alg(self, feats, mask):
        """ Compute the partition function (log sum exp of all paths) using forward algorithm. """
        batch_size, seq_len, num_labels = feats.shape
        # boolean mask for checking validity more easily
        mask_bool = mask.bool()

        # Initialize log_alpha: (batch_size, num_labels)
        log_alpha = torch.full((batch_size, num_labels), -10000.0, device=self.device)
        log_alpha[:, self.start_label_id] = 0.0 # Start probability

        # Transitions matrix (num_labels, num_labels)
        transitions = self.transitions.unsqueeze(0) # Add batch dimension for broadcasting: (1, num_labels, num_labels)

        # Iterate through the sequence (from first word to last)
        for t in range(seq_len):
            # Only compute if the current position is not padding
            is_valid_step = mask_bool[:, t] # Shape: (batch_size,)
            if not is_valid_step.any(): # If entire batch is padding at this step, skip
                continue

            # Emission scores for current step: (batch_size, num_labels)
            emit_scores_t = feats[:, t]

            # Combine scores: log_alpha (previous) + transitions + emit_scores (current)
            # log_alpha shape: (batch_size, num_labels) -> unsqueeze for broadcasting (batch_size, num_labels, 1)
            # transitions shape: (1, num_labels, num_labels)
            # emit_scores_t shape: (batch_size, num_labels) -> unsqueeze for broadcasting (batch_size, 1, num_labels)
            # Result shape: (batch_size, num_labels, num_labels) where [b, i, j] is score of path ending at t-1 in state i, transitioning to state j at step t
            scores_t = log_alpha.unsqueeze(2) + transitions + emit_scores_t.unsqueeze(1)

            # LogSumExp over previous states (axis=1) to get next log_alpha
            # Result shape: (batch_size, num_labels)
            next_log_alpha = log_sum_exp_batch(scores_t, axis=1)

            # Update log_alpha only for valid steps in the batch
            # Use `where` for safe update based on mask
            log_alpha = torch.where(is_valid_step.unsqueeze(1), next_log_alpha, log_alpha)


        # Add final transition to STOP_TAG
        log_alpha += self.transitions[:, self.stop_label_id].unsqueeze(0)

        # LogSumExp over the final states to get the partition function Z(x)
        partition = log_sum_exp_batch(log_alpha, axis=1) # Shape: (batch_size,)
        return partition

    def _score_sentence(self, feats, label_ids, mask):
        """ Compute the score of the gold-standard sequence `label_ids`. """
        batch_size, seq_len, _ = feats.shape
        mask_bool = mask.bool() # Boolean mask

        # Initialize score
        score = torch.zeros(batch_size, device=self.device)

        # --- Add Start Transition Score ---
        # Transition from START to the first actual label
        # Need to find the first valid label index for each sequence
        first_valid_idx = torch.ones(batch_size, dtype=torch.long, device=self.device) # Assume first token after CLS is valid
        # Gather the labels at the first valid index
        first_labels = label_ids.gather(1, first_valid_idx.unsqueeze(1)).squeeze(1)
        score += self.transitions[self.start_label_id, first_labels]

        # --- Add Emission and Transition Scores for the rest of the sequence ---
        for t in range(seq_len - 1):
             # Get current and next label IDs for transition scoring
             current_labels = label_ids[:, t]
             next_labels = label_ids[:, t+1]

             # Gather transition scores: (batch_size,)
             transition_scores = self.transitions[current_labels, next_labels]

             # Gather emission scores for the *next* step (t+1)
             # Shape: (batch_size,)
             emit_scores = feats[:, t+1].gather(1, next_labels.unsqueeze(1)).squeeze(1)

             # Mask: only add score if the *next* step (t+1) is valid (not padding)
             is_valid = mask_bool[:, t+1] # Check mask at t+1
             score += (transition_scores + emit_scores) * is_valid.float()

        # --- Add Transition to Stop Score ---
        # Find the last valid index for each sequence
        seq_lengths = mask.sum(dim=1) # Length of each sequence in the batch
        last_valid_idx = seq_lengths - 1 # Index of the last non-padded token
        # Gather the label IDs at the last valid index
        last_labels = label_ids.gather(1, last_valid_idx.unsqueeze(1)).squeeze(1)
        # Add transition score from last valid label to STOP
        score += self.transitions[last_labels, self.stop_label_id]

        return score

    def _viterbi_decode(self, feats, mask):
        """ Find the best path using Viterbi algorithm. """
        batch_size, seq_len, num_labels = feats.shape
        mask_bool = mask.bool() # Boolean mask

        # Initialize Viterbi score table and backpointers
        log_delta = torch.full((batch_size, num_labels), -10000.0, device=self.device)
        log_delta[:, self.start_label_id] = 0 # Start with START_TAG

        # Backpointers: stores the previous state index that led to the max score
        psi = torch.zeros((batch_size, seq_len, num_labels), dtype=torch.long, device=self.device)

        # Transitions matrix (num_labels, num_labels)
        transitions = self.transitions.unsqueeze(0) # (1, num_labels, num_labels)

        # Iterate through the sequence
        for t in range(seq_len):
            # Only compute if the current position is not padding
            is_valid_step = mask_bool[:, t].unsqueeze(1) # (batch_size, 1)
            if not is_valid_step.any(): continue

            # Emission scores for current step t: (batch_size, num_labels)
            emit_scores_t = feats[:, t]

            # Combine scores: previous log_delta + transitions
            # log_delta shape: (batch_size, num_labels) -> (batch_size, num_labels, 1)
            # transitions shape: (1, num_labels, num_labels)
            # Result: (batch_size, num_labels, num_labels) score[b, prev_s, curr_s]
            scores_t = log_delta.unsqueeze(2) + transitions

            # Find the maximum score and the corresponding previous state index
            # Max over previous states (dim=1)
            # max_scores_t: (batch_size, num_labels) - max score ending in each state `curr_s` at step `t`
            # max_indices_t: (batch_size, num_labels) - index of the `prev_s` that gave the max score
            max_scores_t, max_indices_t = torch.max(scores_t, dim=1)

            # Add emission scores for the current step
            max_scores_t += emit_scores_t

            # Update psi (backpointers) for the current step t
            # Note: Viterbi backpointers usually point from t to t-1,
            # so store indices corresponding to the *next* step's calculation.
            # Let's adjust index: psi stores the index of the best *previous* state for step t.
            if t > 0: # Store backpointer from step t-1
                 psi[:, t, :] = max_indices_t # psi[b, t, current_state] = best_previous_state


            # Update log_delta only for valid steps
            # Use `where` for safe update
            log_delta = torch.where(is_valid_step, max_scores_t, log_delta)

        # --- Add final transition to STOP tag ---
        log_delta += self.transitions[:, self.stop_label_id].unsqueeze(0)

        # --- Backtracking ---
        best_paths = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)

        # Find the best score and the last tag for each sequence
        best_scores, last_tags = torch.max(log_delta, dim=1) # (batch_size,)

        # Iterate backwards from the end of the sequence
        for b in range(batch_size):
            seq_len_b = int(mask[b].sum().item()) # Actual length for this sequence
            if seq_len_b == 0: continue # Handle empty sequence

            best_paths[b, seq_len_b-1] = last_tags[b] # Set last tag

            # Backtrack using psi table
            for t in range(seq_len_b - 2, -1, -1): # From second-to-last down to 0
                # The best tag at step t is the backpointer from the best tag at t+1
                best_paths[b, t] = psi[b, t + 1, best_paths[b, t + 1]]


        return best_scores, best_paths # Shape: (batch_size,), (batch_size, seq_len)

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        """ Calculate the negative log-likelihood loss. """
        # mask should be integer (0/1) or float for some operations
        mask = input_mask.float()
        
        # Emission scores from encoder: (batch_size, seq_len, num_labels)
        feats = self._get_encoder_features(input_ids, segment_ids, input_mask)

        # Partition function Z(x): (batch_size,)
        forward_score = self._forward_alg(feats, mask)

        # Score of the gold sequence S(x, y): (batch_size,)
        gold_score = self._score_sentence(feats, label_ids, mask)

        # Loss = log Z(x) - S(x, y)
        # Average over batch
        nll_loss = torch.mean(forward_score - gold_score)
        return nll_loss

    def forward(self, input_ids, segment_ids, input_mask):
        """ Forward pass for inference/prediction. """
        # mask should be integer (0/1) or float
        mask = input_mask.float()

        # Emission scores: (batch_size, seq_len, num_labels)
        feats = self._get_encoder_features(input_ids, segment_ids, input_mask)

        # Find best path using Viterbi
        best_scores, best_paths = self._viterbi_decode(feats, mask)
        return best_scores, best_paths

# --- Prediction Function ---
def predict_entities(model, tokenizer, text, label_map, idx2label, max_seq_length, device):
    """ Predicts entities in a given text string using the trained model. """
    model.eval()
    model.to(device)

    if not text or not text.strip(): return []
    words = text.split()
    if not words: return []

    # Create a dummy example
    # Provide 'O' labels; source doesn't matter here
    predict_example = InputExample(guid="predict_0", words=words, labels=['O'] * len(words), source='predict')

    # Convert to features
    try:
        features = example2feature(predict_example, tokenizer, label_map, max_seq_length)
        # Manually create batch tensors
        input_ids = torch.LongTensor([features.input_ids]).to(device)
        input_mask = torch.LongTensor([features.input_mask]).to(device)
        segment_ids = torch.LongTensor([features.segment_ids]).to(device)
        predict_mask_batch = torch.BoolTensor([features.predict_mask]).to(device) # Note: Use BoolTensor for predict_mask
    except Exception as e:
        print(f"Error creating features for prediction: {e}")
        return []

    # Get predictions
    tag_seq = None
    try:
        with torch.no_grad():
            _, tag_seq = model(input_ids, segment_ids, input_mask) # Get predicted sequence IDs
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return []

    if tag_seq is None or tag_seq.nelement() == 0:
        print("Prediction failed, no tag sequence generated.")
        return []

    # --- Post-process Predictions ---
    # Extract valid predictions using the predict_mask
    valid_predictions = []
    p_mask = predict_mask_batch[0].cpu().numpy() # Predict mask for the single example
    t_seq = tag_seq[0].cpu().numpy()          # Predicted tags for the single example
    seq_len = min(len(p_mask), len(t_seq))

    for i in range(seq_len):
        if p_mask[i]: # If this token corresponds to the start of a word
            tag_id = t_seq[i]
            valid_predictions.append(idx2label.get(tag_id, 'O')) # Default to 'O' if unknown ID

    # Ensure number of predictions matches number of words (handle potential truncation)
    num_words = len(words)
    num_preds = len(valid_predictions)
    if num_preds != num_words:
        print(f"Warning: Mismatch between words ({num_words}) and predictions ({num_preds}). Using min length.")
        min_len = min(num_words, num_preds)
        words = words[:min_len]
        valid_predictions = valid_predictions[:min_len]

    # --- Convert BIO tags to entities ---
    entities = []
    current_entity = None
    for i, (word, tag) in enumerate(zip(words, valid_predictions)):
        bio_tag = tag[0] if tag != 'O' else 'O'
        entity_type = tag[2:] if bio_tag in ['B', 'I'] else None

        if bio_tag == 'B':
            if current_entity: entities.append(current_entity) # Save previous entity
            current_entity = {'text': word, 'start_token': i, 'end_token': i, 'type': entity_type}
        elif bio_tag == 'I' and current_entity and entity_type == current_entity['type']:
            current_entity['text'] += ' ' + word
            current_entity['end_token'] = i
        else: # O tag or I tag mismatch/start
            if current_entity: entities.append(current_entity) # Save previous entity
            current_entity = None

    if current_entity: entities.append(current_entity) # Add the last entity

    return entities


# --- Training Function ---
def train(config):
    """ Trains the unified BERT-CRF model and evaluates it. """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- Load Data ---
    print(f"Loading unified dataset from {config.dataset_path}")
    try:
        df = pd.read_csv(config.dataset_path, keep_default_na=False, encoding='utf-8') # Added encoding
        required_cols = ['Sentence_ID', 'Word', 'STIX_Tag', 'Source']
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"Dataset missing required columns: {required_cols}")
        print(f"Dataset loaded with {len(df)} rows and columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {config.dataset_path}"); return
    except Exception as e:
        print(f"Error loading or validating dataset: {e}"); return

    # --- Create Label Map ---
    print("Creating label map from all STIX tags...")
    # Ensure base labels are present, then add unique tags from the dataset
    base_labels = ['O', 'X', '[CLS]', '[SEP]']
    unique_tags = sorted(list(set(df['STIX_Tag'][~df['STIX_Tag'].isin(['O'])].unique())))
    all_unique_labels = base_labels + [tag for tag in unique_tags if tag not in base_labels]
    label_map = {label: i for i, label in enumerate(all_unique_labels)}
    idx2label = {i: label for label, i in label_map.items()}
    print(f"Created label map with {len(all_unique_labels)} labels.") #: {all_unique_labels}")
    start_label_id = label_map['[CLS]']
    stop_label_id = label_map['[SEP]']

    # --- Prepare Examples & Split Data ---
    all_examples = prepare_unified_dataset(df)
    if not all_examples: print("Error: No examples created."); return

    print(f"Splitting {len(all_examples)} examples (Train/Val/Test)...")
    train_val_examples, test_examples = train_test_split(
        all_examples, test_size=config.test_size, random_state=config.seed
    )
    # Calculate validation split size relative to the remaining train_val set
    relative_val_size = config.val_size / (1.0 - config.test_size) if (1.0 - config.test_size) > 0 else 0.1
    train_examples, dev_examples = train_test_split(
        train_val_examples, test_size=relative_val_size, random_state=config.seed
    )
    print(f"Split: Train={len(train_examples)}, Val={len(dev_examples)}, Test={len(test_examples)}")

    # --- Load Tokenizer ---
    print(f"Loading tokenizer for model: {config.model_type}")
    try:
        do_lower_case = 'uncased' in config.model_type.lower()
        add_prefix_space = 'roberta' in config.model_type.lower() # RoBERTa specific handling
        
        if 'darkbert' in config.model_type.lower():
             # Handle potential authentication need for DarkBERT
             try:
                 tokenizer = AutoTokenizer.from_pretrained(config.model_type, use_auth_token=True)
             except Exception as auth_err:
                 print("\n--- DarkBERT Tokenizer Loading Note ---")
                 print("Ensure you have access rights and consider logging in via `huggingface-cli login`.")
                 print(f"Original error: {auth_err}")
                 print("---\n")
                 raise ValueError(f"Failed to load DarkBERT tokenizer. Error: {auth_err}")
        elif 'securebert' in config.model_type.lower():
             tokenizer = AutoTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
        elif 'cysecbert' in config.model_type.lower():
             tokenizer = AutoTokenizer.from_pretrained("markusbayer/CySecBERT")
        else:
             # General case
             tokenizer = AutoTokenizer.from_pretrained(
                 config.model_type,
                 do_lower_case=do_lower_case,
                 add_prefix_space=add_prefix_space
             )
    except Exception as e:
         print(f"Error loading tokenizer {config.model_type}: {e}"); return

    # --- Create Datasets & DataLoaders ---
    print("Creating Train/Dev Datasets...")
    train_dataset = NerDataset(train_examples, tokenizer, label_map, config.max_seq_length)
    dev_dataset = NerDataset(dev_examples, tokenizer, label_map, config.max_seq_length)

    print("Creating DataLoaders...")
    pin_memory = config.device == torch.device("cuda")
    train_dataloader = data.DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=NerDataset.pad, pin_memory=pin_memory
    )
    dev_dataloader = data.DataLoader(
        dataset=dev_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=NerDataset.pad, pin_memory=pin_memory
    )

    # --- Training Setup ---
    if len(train_dataloader) == 0: print("Error: Training dataloader is empty."); return
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    total_train_steps = int(len(train_dataloader) // config.gradient_accumulation_steps * config.total_train_epochs)
    print("***** Training information *****")
    print(f"  Model Type = {config.model_type}")
    print(f"  Num Train examples = {len(train_examples)}")
    print(f"  Num Val examples = {len(dev_examples)}")
    print(f"  Num Test examples = {len(test_examples)}")
    print(f"  Batch size = {config.batch_size}")
    print(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    print(f"  Effective Batch size = {effective_batch_size}")
    print(f"  Total train epochs = {config.total_train_epochs}")
    print(f"  Total optimization steps = {total_train_steps}")
    print(f"  Device = {config.device}")

    # --- Initialize Model ---
    print("Initializing BERT_CRF_NER model...")
    model = BERT_CRF_NER(
        model_type=config.model_type,
        start_label_id=start_label_id,
        stop_label_id=stop_label_id,
        num_labels=len(all_unique_labels),
        device=config.device
    )
    model.to(config.device)

    # --- Optimizer & Scheduler ---
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    crf_fc_params = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(cf in n for cf in crf_fc_params)],
         'weight_decay': config.weight_decay_finetune},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any(cf in n for cf in crf_fc_params)],
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if n in ('transitions', 'hidden2label.weight')],
         'lr': config.lr_crf_fc, 'weight_decay': config.weight_decay_crf_fc},
        {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'],
         'lr': config.lr_crf_fc, 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_train_steps * config.warmup_proportion),
        num_training_steps=total_train_steps
    )

    # --- Training Loop ---
    global_step = 0
    best_valid_f1 = 0.0
    early_stopping_counter = 0
    best_epoch = 0
    model_output_dir = os.path.join(config.output_dir, config.model_type.replace('/', '_') + "_unified")
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"Model checkpoints and results will be saved to: {model_output_dir}")
    history = {'train_loss': [], 'valid_f1': [], 'best_valid_f1': 0.0, 'epochs_ran': 0}

    print("\n***** Starting Training *****")
    for epoch in range(config.total_train_epochs):
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        train_start = time.time()
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.total_train_epochs}")

        for step, batch in enumerate(train_iterator):
            batch = tuple(t.to(config.device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

            loss = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                 torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                 optimizer.step()
                 scheduler.step()
                 optimizer.zero_grad()
                 global_step += 1
                 nb_tr_steps += 1 # Count actual optimization steps
                 train_iterator.set_postfix({'loss': tr_loss / nb_tr_steps}) # Show avg loss for steps done

        # --- End of Epoch ---
        history['epochs_ran'] = epoch + 1
        avg_train_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0
        history['train_loss'].append(avg_train_loss)
        train_duration = (time.time() - train_start) / 60.0
        print(f"\nEpoch {epoch+1} Summary: Avg Train Loss={avg_train_loss:.4f}, Time={train_duration:.2f}m")

        # --- Validation ---
        print("Running Validation...")
        valid_f1, _ = evaluate(
            model, dev_dataloader, epoch + 1, 'Validation Set',
            label_map, idx2label, config.device
        )
        history['valid_f1'].append(valid_f1)
        print(f"Validation F1: {valid_f1:.4f}")

        # --- Checkpointing & Early Stopping ---
        if valid_f1 > best_valid_f1:
             print("  New best validation F1! Saving model...")
             torch.save({
                 'epoch': epoch + 1, 'model_state': model.state_dict(), 'valid_f1': valid_f1,
                 'config': vars(config), 'label_map': label_map, 'idx2label': idx2label,
                 'max_seq_length': config.max_seq_length, 'model_type': config.model_type,
             }, os.path.join(model_output_dir, 'best_model.pt'))
             tokenizer.save_pretrained(model_output_dir) # Save tokenizer with best model
             print(f"  Best model and tokenizer saved to {model_output_dir}")
             best_valid_f1 = valid_f1
             history['best_valid_f1'] = best_valid_f1
             early_stopping_counter = 0
             best_epoch = epoch + 1
        else:
             early_stopping_counter += 1
             print(f"  Validation F1 ({valid_f1:.4f}) did not improve from best ({best_valid_f1:.4f}). Counter: {early_stopping_counter}/{config.early_stopping_patience}")

        if (epoch + 1) % config.checkpoint_freq == 0:
             chk_path = os.path.join(model_output_dir, f'checkpoint_epoch_{epoch+1}.pt')
             print(f"Saving checkpoint at epoch {epoch+1} to {chk_path}")
             torch.save({'epoch': epoch + 1, 'model_state': model.state_dict()}, chk_path)

        history_path = os.path.join(model_output_dir, 'training_history.json')
        try:
            with open(history_path, 'w') as f: json.dump(history, f, indent=2)
        except Exception as e: print(f"Warning: Could not save history: {e}")

        if early_stopping_counter >= config.early_stopping_patience:
             print(f"\nEarly stopping triggered after {epoch+1} epochs.")
             break

    # === End of Training ===
    print("\n***** Training Finished *****")
    print(f"Best validation F1: {best_valid_f1:.4f} at epoch {best_epoch}")

    # === Final Evaluation ===
    print("\n***** Final Evaluation on Test Set *****")
    best_model_path = os.path.join(model_output_dir, 'best_model.pt')
    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint ('best_model.pt') not found. Cannot evaluate."); return

    print(f"Loading best model from: {best_model_path}")
    try:
        checkpoint = torch.load(best_model_path, map_location=config.device)
        # Reload necessary info from checkpoint
        loaded_config = checkpoint.get('config', vars(config)) # Use saved config if available
        loaded_model_type = checkpoint.get('model_type', config.model_type)
        label_map = checkpoint['label_map']
        idx2label = checkpoint['idx2label']
        num_labels_loaded = len(label_map)

        final_model = BERT_CRF_NER( # Re-initialize model structure
            model_type=loaded_model_type, start_label_id=label_map['[CLS]'], stop_label_id=label_map['[SEP]'],
            num_labels=num_labels_loaded, device=config.device
        )
        final_model.load_state_dict(checkpoint['model_state'])
        final_model.to(config.device)
        final_model.eval()
        # Load tokenizer saved alongside the best model
        final_tokenizer = AutoTokenizer.from_pretrained(model_output_dir)

    except Exception as e:
        print(f"Error loading best model/tokenizer checkpoint: {e}"); return

    # --- 1. Overall Test Set Evaluation ---
    print("\n--- Evaluating on Overall Test Set ---")
    # Need to create the test dataset and dataloader using the loaded tokenizer and maps
    test_dataset = NerDataset(test_examples, final_tokenizer, label_map, config.max_seq_length)
    test_dataloader = data.DataLoader(
        dataset=test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=NerDataset.pad, pin_memory=pin_memory
    )
    overall_test_f1, overall_class_metrics = evaluate(
        final_model, test_dataloader, "Final Overall", 'Test Set (Overall)',
        label_map, idx2label, config.device
    )

    # Store results
    final_results = {
        'model_type': loaded_model_type, 'best_epoch': best_epoch, 'best_valid_f1': best_valid_f1,
        'overall_test_f1': overall_test_f1, 'overall_class_metrics': {},
        'source_specific_test_results': {}
    }
    for cls_id, metrics in overall_class_metrics.items():
        final_results['overall_class_metrics'][metrics['label']] = {k: v for k, v in metrics.items() if k != 'label'}

    # --- 2. Source-Specific Test Evaluation ---
    print("\n--- Evaluating on Source-Specific Test Subsets ---")
    for source_name in SOURCES_TO_EVALUATE:
        print(f"\n--- Evaluating source: {source_name} ---")
        source_test_examples = [ex for ex in test_examples if ex.source == source_name]
        if not source_test_examples:
            print(f"  No test examples for source {source_name}. Skipping.")
            final_results['source_specific_test_results'][source_name] = {'f1': 0.0, 'class_metrics': {}, 'message': 'No examples'}
            continue

        print(f"  Found {len(source_test_examples)} examples for {source_name}.")
        source_test_dataset = NerDataset(source_test_examples, final_tokenizer, label_map, config.max_seq_length)
        source_test_dataloader = data.DataLoader(
            dataset=source_test_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, collate_fn=NerDataset.pad, pin_memory=pin_memory
        )
        source_f1, source_class_metrics = evaluate(
            final_model, source_test_dataloader, f"Final {source_name}", f'Test Set ({source_name})',
            label_map, idx2label, config.device
        )
        # Store source results
        final_results['source_specific_test_results'][source_name] = {'f1': source_f1, 'class_metrics': {}}
        for cls_id, metrics in source_class_metrics.items():
            final_results['source_specific_test_results'][source_name]['class_metrics'][metrics['label']] = {k: v for k, v in metrics.items() if k != 'label'}


    # Save final combined results
    results_path = os.path.join(model_output_dir, 'final_evaluation_results.json')
    print(f"\nSaving final combined evaluation results to {results_path}")
    try:
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization if needed
            # (Metrics should already be floats/ints, but good practice)
            def default_serializer(obj):
                if isinstance(obj, np.integer): return int(obj)
                elif isinstance(obj, np.floating): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
            json.dump(final_results, f, indent=2, default=default_serializer)
        print("Final results saved successfully.")
    except Exception as e:
        print(f"Error saving final results: {e}")

    # --- Example Prediction (Optional) ---
    print("\n--- Example Prediction ---")
    sample_text = "The malware sample associated with APT34 used spear-phishing techniques."
    print(f"Input text: {sample_text}")
    predicted_entities = predict_entities(
        final_model, final_tokenizer, sample_text, label_map, idx2label, config.max_seq_length, config.device
    )
    print("Predicted Entities:")
    for entity in predicted_entities:
        print(f"  - Text: '{entity['text']}', Type: {entity['type']}, Tokens: [{entity['start_token']}-{entity['end_token']}]")
    print("-------------------------")



# --- Main Execution ---
def main():
    """Parses arguments and starts training."""
    parser = argparse.ArgumentParser(description='Unified BERT-CRF for Cybersecurity NER')

    # Dataset & Splits
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the combined dataset CSV file (e.g., cyberner_combined_stix.csv)')
    parser.add_argument('--test_size', type=float, default=0.15, help='Fraction of data for final test set')
    parser.add_argument('--val_size', type=float, default=0.15, help='Fraction of *original* data for validation (split from train)')

    # Model
    parser.add_argument('--model_type', type=str, required=True, help='Hugging Face model identifier (e.g., bert-base-cased, roberta-base, ehsanaghaei/SecureBERT, s2w-ai/DarkBERT, markusbayer/CySecBERT)')
    parser.add_argument('--max_seq_length', type=int, default=256, help='Max sequence length')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Encoder learning rate')
    parser.add_argument('--lr_crf_fc', type=float, default=8e-5, help='CRF/FC layer learning rate')
    parser.add_argument('--weight_decay_finetune', type=float, default=1e-5, help='Encoder weight decay')
    parser.add_argument('--weight_decay_crf_fc', type=float, default=5e-6, help='CRF/FC weight decay')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Patience for early stopping (epochs)')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Save checkpoint frequency (epochs)')

    # Environment & Output
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--output_dir', type=str, default='./outputs_cyberner_unified/', help='Base output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    config = Config(args)

    print("\n***** Configuration (Unified Model Training) *****")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")
    print("**************************************************\n")

    train(config)

if __name__ == "__main__":
    print("Starting Unified BERT-CRF NER training script...")
    main()
    print("\nScript finished.")
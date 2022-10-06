# Databricks notebook source
# default_exp docClassify
from nbdev import *

# COMMAND ----------

# MAGIC %md # HuggingFace Document Classification Utils 
# MAGIC 
# MAGIC > Classes to build and run document classification pipelines using baseline HuggingFace functionality.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC This library contains a single utility class and several functions to make it easy to run a simple document classifier for scientific papers. 
# MAGIC 
# MAGIC An example code run through is as follows: 
# MAGIC 
# MAGIC ```
# MAGIC # DRSM BASIC TRAINING ANALYSIS  
# MAGIC import datasets
# MAGIC 
# MAGIC column_names =['ID_PAPER', 'Labeling_State', 'Comments', 'Explanation', 'Correct_Label', 'Agreement', 'TRIMMED_TEXT']
# MAGIC text_columns = ['TRIMMED_TEXT']
# MAGIC label_column = 'Correct_Label'
# MAGIC drsm_categories = ['clinical characteristics or disease pathology',
# MAGIC               'therapeutics in the clinic', 
# MAGIC               'disease mechanism', 
# MAGIC               'patient-based therapeutics', 
# MAGIC               'other',
# MAGIC               'irrelevant']
# MAGIC 
# MAGIC ds_temp = datasets.load_dataset('csv', delimiter="\t", data_files='/dbfs/FileStore/user/gully/drsm_curated_data/labeled_data_2022_01_03.tsv')
# MAGIC train_test_valid = ds_temp['train'].train_test_split(0.1)
# MAGIC test_valid = train_test_valid['test'].train_test_split(0.5)
# MAGIC drsm_ds = datasets.DatasetDict({
# MAGIC     'train': train_test_valid['train'],
# MAGIC     'test': test_valid['test'],
# MAGIC     'valid': test_valid['train']})

# COMMAND ----------

# export 

from functools import partial 
from tqdm import tqdm 
tqdm = partial(tqdm, position=0, leave=True)

import transformers
import mlflow
import datasets
import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datasets import list_datasets, load_dataset, load_metric
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, AutoConfig, 
                          TrainingArguments, Trainer)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

import pickle

print(f"Running on transformers v{transformers.__version__} and datasets v{datasets.__version__}")

# COMMAND ----------



# COMMAND ----------

#export

class HF_trainer_wrapper():
  '''
  Class to provide support training and experimenting with simple document classification tools under either a multi-label or multi-class classification paradigm.

  Attributes:
  * run_name:  
  * model_ckpt:  
  * output_dir:  
  * logging_dir: 
  * epochs: 
  * max_length: 
  * problem_type: 
  '''
  tokenizer = None
  name = ''
  text_columns = []
  ds = None
  
  def __init__(self, run_name, model_ckpt, output_dir, logging_dir, epochs, max_length=512, problem_type="multi_label_classification"):
    self.run_name = run_name
    self.model_ckpt = model_ckpt
    self.output_dir = output_dir
    self.logging_dir = logging_dir
    self.epochs = epochs
    self.max_length = max_length
    self.problem_type = problem_type
    self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt, problem_type=self.problem_type)
    
  def prepare_dataset(self, ds, text_columns, label_column, categories, problem_type="multi_label_classification"):
    self.ds = ds 
    self.text_columns = text_columns
    self.label_column = label_column
    self.categories = categories
    
    def concat_text_fields(row):
      t = ''
      for f in self.text_columns:
        if row[f] is not None:
          t += row[f]
          if t[-1:] != '.':
            t += '.\n'
          else: 
            t += '\n'
      return {"text": '%s'%(t)}     

    def tokenize_and_encode(row):
      return self.tokenizer(row["text"], 
                     padding='max_length', 
                     truncation=True, 
                     max_length=self.max_length)
    
    print('Stripping Null Data from datasets')
    if problem_type=="multi_label_classification": 
      self.ds = self.ds.map(lambda x : {"labels": [1 if x[self.label_column] is not None and c in x[self.label_column] else 0 for c in self.categories] })
    else: 
      self.ds = self.ds.map(lambda x : {"labels": [self.categories.index(x[self.label_column])]})
    
    # NOTE THIS USES THE CONTEXTUALLY DEFINED 'field_list' variable 
    # implicitly in the concat_text_fields function above 
    # (not all that great, but not sure how better to do this)
    print('Concatonating text fields')
    self.ds = self.ds.map(concat_text_fields)
    
    cols = self.ds["train"].column_names
    cols.remove("labels")
    print('Tokenizing and encoding')
    self.ds_enc = self.ds.map(tokenize_and_encode, 
                    batched=True, 
                    remove_columns=cols)

    # cast label IDs to floats
    self.ds_enc.set_format("torch")
    
    if problem_type=="multi_label_classification": 
      print('Converting label ints to floats')
      self.ds_enc = (self.ds_enc.map(lambda x : 
                           {"float_labels": x["labels"].to(torch.float)},
                           remove_columns=["labels"])
                .rename_column("float_labels", "labels"))
    print('Done')
  
  def build_model(self, loc=None):
    if loc is None:
      loc = self.model_ckpt
     
    if self.problem_type == 'multi_label_classification':
      num_labels = len( self.ds_enc['train']['labels'][0] )
    else:
      num_labels = len(set([i[0] for i in self.ds['train']['labels']]))

    if os.path.exists(loc):
      self.model = AutoModelForSequenceClassification.from_pretrained(loc, 
                                                                      num_labels=num_labels, 
                                                                      ignore_mismatched_sizes=True, 
                                                                      problem_type=self.problem_type).to('cuda')
    else: 
      self.model = AutoModelForSequenceClassification.from_pretrained(loc, 
                                                                      num_labels=num_labels, 
                                                                      problem_type=self.problem_type).to('cuda')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.model = self.model.to(device)

  def build_trainer(self, warmup_prop = 0.1, batch_size = 8, gradient_accumulation_steps = 2):
    
    def compute_metrics(pred):
      #print(pred.label_ids)
      #print(pred.predictions)
      if self.problem_type=="multi_label_classification": 
        labels = pred.label_ids
        preds = torch.sigmoid(torch.FloatTensor(pred.predictions)).round().long().cpu().detach().numpy()
        #preds = [pl>0 for pl in pred.predictions] 
        #preds = pred.predictions.argmax(-1)
      else: 
        preds = np.argmax(pred.predictions, axis=1)
        labels = pred.label_ids
      
      precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
      acc = accuracy_score(labels, preds)

      return {
          'accuracy': acc,
          'f1': f1,
          'precision': precision,
          'recall': recall
      }
    
    num_train_optimization_steps = int(len(self.ds['train']) / batch_size / gradient_accumulation_steps) * self.epochs
    warmup_steps = int(warmup_prop * num_train_optimization_steps)
    
    self.args = TrainingArguments(learning_rate=2e-5, 
                                  output_dir=self.output_dir, 
                                  num_train_epochs=self.epochs,
                                  per_device_train_batch_size=batch_size, 
                                  gradient_accumulation_steps=2,
                                  per_device_eval_batch_size=batch_size, 
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False, 
                                  warmup_steps=warmup_steps, 
                                  logging_dir=self.logging_dir,
                                  run_name=self.run_name)
    
    self.trainer = Trainer(model = self.model, 
                           args = self.args,
                           train_dataset = self.ds_enc["train"], 
                           eval_dataset = self.ds_enc["valid"], 
                           tokenizer = self.tokenizer, 
                           compute_metrics = compute_metrics)

  def train(self, checkpoint=None):
    if checkpoint:
      self.trainer.train(checkpoint)
    else: 
      self.trainer.train()
    tmp_model_path = self.output_dir+'/final_model/'
    self.trainer.save_model(tmp_model_path)

  def evaluate(self):
    self.trainer.evaluate()
   
  def test(self):
    return self.trainer.predict(self.ds_enc['test'])

  def save(self):
    with open(output_dir+'/hft.pickle', 'wb') as f:
      pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

# COMMAND ----------

#export 

import mlflow
import pickle

def run_HF_trainer_expt(ds, text_columns, label_column, categories, run_name, 
                                   model_input, model_path, log_path, epochs, 
                                   batch_size=8,
                                   transfer_model=None,
                                   problem_type="multi_label_classification",
                                   run_training=True,
                                   freeze_layers=False):
  
  hft = HF_trainer_wrapper(run_name, model_input, model_path, log_path, epochs, problem_type=problem_type)
  hft.prepare_dataset(ds, text_columns, label_column, categories, problem_type=problem_type)
  if transfer_model is None:
    hft.build_model()
  else:
    hft.build_model(loc=transfer_model)
  
  if freeze_layers:
    for param in hft.model.bert.parameters():
      param.requires_grad = False
  
  hft.build_trainer(batch_size=batch_size)
  if run_training:
    hft.train()
  pdat = hft.test()
  with open(log_path+'/pdat.pkl', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(pdat, f, pickle.HIGHEST_PROTOCOL)
  
  mlflow.log_param('model_input', model_input)
  mlflow.log_param('model_path', model_path)
  mlflow.log_param('log_path', log_path)
  if transfer_model: 
    mlflow.log_param('transfer_model', transfer_model )
  mlflow.log_param('epochs', model_input)
  mlflow.log_metric('test_accuracy', pdat.metrics['test_accuracy'])
  mlflow.log_metric('test_f1', pdat.metrics['test_f1'])
  mlflow.log_metric('test_precision', pdat.metrics['test_precision'])
  mlflow.log_metric('test_recall', pdat.metrics['test_recall'])
  mlflow.end_run()

  return hft

# COMMAND ----------

#export 

from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import Dataset

def get_folds_from_dataframe(df, id_col, category_col, n_splits):
  folded_ds = []
  X = np.array(df[id_col].to_list())
  y = np.array(df[category_col].to_list())
  skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
  skf.get_n_splits(X, y)
  for train_index, test_index in skf.split(X, y):
    train_valid_df = df.iloc[train_index]
    train_df, valid_df = train_test_split(train_valid_df, train_size=.9)
    test_df = df.iloc[test_index]
    
    #checks
    for i in test_index:
      if i in train_index:
        raise ValueError('TEST DATA FOUND IN TRAINING DATA.')
    
    train = Dataset.from_pandas(train_df)
    valid = Dataset.from_pandas(valid_df)
    test = Dataset.from_pandas(test_df)
    drsm_ds = datasets.DatasetDict({'train': train, 'test': test, 'valid': valid})
    folded_ds.append(drsm_ds)
  
  return folded_ds

# COMMAND ----------

#export 

import mlflow
from datasets import concatenate_datasets

def run_HF_trainer_kfold_crossvalidation(folds, text_columns, label_column, categories, run_name, 
                                                    model_input, model_path, log_path, epochs, 
                                                    batch_size=8, 
                                                    problem_type="multi_label_classification",
                                                    transfer_model=None, 
                                                    run_training=True,
                                                    freeze_layers=False):
  metrics_list = []
  for i, fold_ds in enumerate(folds):
    
    hft = run_HF_trainer_expt(fold_ds, text_columns, label_column, categories, run_name, 
                                         model_input, model_path+'/fold'+str(i), log_path+'/fold'+str(i), 
                                         epochs, 
                                         batch_size=batch_size, 
                                         problem_type=problem_type, 
                                         transfer_model=transfer_model, 
                                         run_training=run_training,
                                         freeze_layers=freeze_layers)
    
    with open(log_path+'/fold'+str(i)+'/pdat.pkl', 'rb') as f:
      pdat = pickle.load(f)
    tuple = (pdat.metrics['test_accuracy'], pdat.metrics['test_f1'], pdat.metrics['test_precision'], pdat.metrics['test_recall'])
    metrics_list.append(tuple)
  
  df = pd.DataFrame(metrics_list, columns=['test_accuracy','test_f1','test_precision','test_recall'])

  return df

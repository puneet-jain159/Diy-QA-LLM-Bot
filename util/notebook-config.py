# Databricks notebook source
import torch

# COMMAND ----------

if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Location of the Documents
config['loc'] = "/dbfs/FileStore/howden_poc"

# COMMAND ----------

# Define the model we would like to use
# config['model_id'] = 'openai'
config['model_id'] = 'meta-llama/Llama-2-13b-chat-hf'
# config['model_id'] = 'mosaicml/mpt-30b-chat'
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

# DBTITLE 1,Create database
config['database_name'] = 'qabot'

# create database if not exists
_ = spark.sql(f"create database if not exists {config['database_name']}")

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

# DBTITLE 1,Set Environmental Variables for tokens
import os

if config['model_id'] == 'openai':
  os.environ['OPENAI_API_KEY'] = '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'

# COMMAND ----------

# DBTITLE 1,Set document path
config['kb_documents_path'] = "s3://db-gtm-industry-solutions/data/rcg/diy_llm_qa_bot/"
config['vector_store_path'] = f"/dbfs/{username}/qabot/vector_store/{config['model_id']}/howden_poc" # /dbfs/... is a local file system representation

# COMMAND ----------

# DBTITLE 1,mlflow settings
import mlflow
config['registered_model_name'] = 'databricks_llm_qabot_solution_accelerator'
config['model_uri'] = f"models:/{config['registered_model_name']}/production"
_ = mlflow.set_experiment('/Users/{}/{}'.format(username, config['registered_model_name']))

# COMMAND ----------

# DBTITLE 1,Set model configs
if config['model_id'] == "openai":
  # Set the embedding vector and model  ####
  config['embedding_model'] = 'text-embedding-ada-002'
  config['openai_chat_model'] = "gpt-3.5-turbo"
  # Setup prompt template ####
  config['system_message_template'] = """You are a helpful assistant built by Databricks, you are good at helping to answer a question based on the context provided, the context is a document. If the context does not provide enough relevant information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know. If you did not find a good answer from the context, just say I don't know. If the query doesn't form a complete question, just say I don't know. If there is a good answer from the context, try to summarize the context to answer the question."""
  config['human_message_template'] = """Given the context: {context}. Answer the question {question}."""
  config['temperature'] = 0.15

elif config['model_id'] == 'mosaicml/mpt-30b-chat' :
  # Setup prompt template ####
  config['embedding_model'] = 'intfloat/e5-large-v2'

  # Model parameters
  config['model_kwargs'] = {"load_in_8bit" : True,
                            "bnb_8bit_compute_dtype":torch.bfloat16}
  config['pipeline_kwargs']={"temperature":  0.10,
                            "max_new_tokens": 256}
  
  config['template'] = """<|im_start|>system\n-You are a helpful assistant chatbot trained by MosaicML. You are good at helping to answer a question based on the context provided, the context is a document. \n-If the question doesn't form a complete sentence, just say I don't know.\n-If the context is irrelevant to the question, just say I don't know.,\n-If there is a good answer from the context, try to summarize the answer to the question. \n<|im_end|>\n<|im_start|>user\n Given the context: {context}. Answer the question {question} <|im_end|><|im_start|>assistant\n""".strip()

elif config['model_id'] == 'meta-llama/Llama-2-13b-chat-hf' :
  # Setup prompt template ####
  config['embedding_model'] = 'hkunlp/instructor-xl'
  
  config['model_kwargs'] = {"load_in_8bit" : True}

  # Model parameters
  config['pipeline_kwargs']={"temperature":  0.10,
                            "max_new_tokens": 256}
  
  config['template'] = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant.you are good at helping to answer a question based on the context provided, the context is a document. If the context does not provide enough relevant information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know. If you did not find a good answer from the context, just say I don't know. If the query doesn't form a complete question, just say I don't know. If there is a good answer from the context, try to summarize the context to answer the question.
    <</SYS>> Given the context: {context}. Answer the question {question}
    [/INST]""".strip()




# COMMAND ----------

# DBTITLE 1,Set evaluation config
config["eval_dataset_path"]= "./data/eval_data.tsv"

# COMMAND ----------

# DBTITLE 1,Set deployment configs
config['openai_key_secret_scope'] = "solution-accelerator-cicd" # See `./RUNME` notebook for secret scope instruction - make sure it is consistent with the secret scope name you actually use 
config['openai_key_secret_key'] = "openai_api" # See `./RUNME` notebook for secret scope instruction - make sure it is consistent with the secret scope key name you actually use
config['serving_endpoint_name'] = "llm-qabot-endpoint"

# Databricks notebook source
# MAGIC %pip install -U ray[default]==2.5.1 transformers==4.29.2 Xformers sentence-transformers==2.2.2 langchain==0.0.190 chromadb==0.3.25 pypdf==3.9.1 pycryptodome==3.18.0 accelerate==0.19.0 unstructured==0.7.1 unstructured[local-inference]==0.7.1 sacremoses==0.0.53 ninja==1.11.1 torch==2.0.1 pytorch-lightning==2.0.1 tiktoken==0.4.0 openai==0.27.6 faiss-cpu==1.7.4 InstructorEmbedding bitsandbytes typing-inspect==0.8.0 typing_extensions==4.5.0  triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC We are installing some NVIDIA libraries here in order to use a special package called Flash Attention (`flash_attn`) that Mosaic ML's mpt-7b-instruct model needs.

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# MAGIC sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# MAGIC sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# MAGIC sudo add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
# MAGIC sudo apt-get update

# COMMAND ----------

# MAGIC %sh
# MAGIC apt-get install -y libcusparse-dev-11-7 libcublas-dev-11-7 libcusolver-dev-11-7

# COMMAND ----------

# MAGIC %md
# MAGIC Installing `flash_attn` takes around 5 minutes.

# COMMAND ----------

# MAGIC %pip install einops==0.6.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.7/'
%pip install flash_attn==1.0.7

# COMMAND ----------

import nltk
nltk.download('punkt')

# COMMAND ----------

#torch==2.0.1 pytorch-lightning==2.0.1
#dbutils.library.restartPython()

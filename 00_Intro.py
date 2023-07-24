# Databricks notebook source
# MAGIC %md The purpose of this notebook is to set the various configuration values that will control the notebooks that make up the QA Bot accelerator.  This notebook is available at https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC The goal of this solution accelerator is to show how we can leverage a large language model in combination with our own data to create an interactive application capable of answering questions specific to a particular domain or subject area.  The core pattern behind this is the delivery of a question along with a document or document fragment that provides relevant context for answering that question to the model.  The model will then respond with an answer that takes into consideration both the question and the context.
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_flow.png' width=500>
# MAGIC
# MAGIC </p>
# MAGIC To aseemble this application, *i.e.* the Q&A Bot, we will need to assemble a series of documents that are relevant to the domain we wish to serve.  We will need to index these to enable rapid search given a user question. We will then need to assemble the core application which combines a question with a document to form a prompt and submits that prompt to a model in order to generate a response. Finally, we'll need to package both the indexed documents and the core application component as a microservice to enable a wide range of deployment options.
# MAGIC
# MAGIC We will tackle these three steps across the following three notebooks:</p>
# MAGIC
# MAGIC * 01: Build Document Index
# MAGIC * 02: Assemble Application
# MAGIC * 03: Deploy Application
# MAGIC </p>

# COMMAND ----------

# MAGIC %md Initialize the paths we will use throughout the accelerator

# COMMAND ----------

f"19\nRainbow Home insuranceThe buildings are insured against \nloss or damage caused byIn addition to items listed on \npages 16-18 we will not pay for:\n• The excess shown on your policy \nschedule  under paragraphs 1 to 15 \nand A to E of this section.\n1. Fire, smoke, explosion, lightning or \nearthquake.• Loss or damage caused by tobacco \nburns, scorching, melting, warping or \nother forms of heat distortion unless \naccompanied by flames.\n2. Riot, civil commotion, strikes or \nlabour disturbances• Loss or damage occurring where you \nhave:\ni) participated in, assisted, \nencouraged or facilitated the riot \nor spread of the riot.\nii) contributed, directly or indirectly, \nto any damage, destruction or \ntheft of property during the riot.\niii) committed a criminal offence \nrelating to the riot.\n3. Malicious acts or vandalism. • Loss or damage when your home is \nunoccupied  for more than 60 days \nin a row.\n• Loss or damage caused by you , your \ndomestic employees, lodgers, paying \nguests or tenants.\n4a. Storm. • Loss or damage to fences, gates and \nhedges.\n• Loss or damage caused by \nunderground water.SECTION 1 BUILDINGS\nPlease note that this section only applies if it is shown on your policy schedule . \nPART 1 – BUILDINGS.\nWe will pay up to the sum insured shown on your policy schedule  unless we \nspecify otherwise."

# COMMAND ----------

# MAGIC %run "./util/notebook-config"

# COMMAND ----------

dbutils.fs.rm(config['vector_store_path'][5:], True)

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |

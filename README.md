# Chan Zuckerberg Landscaping Toolkit



 ![General proposed workflow for Landscaping systems](https://lucid.app/publicSegments/view/fff0cc6d-c52d-447d-80ce-2f99f8ac0d29/image.png)
# MAGIC
Image source on LucidDraw: [Link](https://lucid.app/lucidchart/a49ee803-ac2d-47ac-a628-492f95dd9346/edit?viewport_loc=2%2C-253%2C2225%2C1488%2C0_0&invitationId=inv_d95f59bf-a965-4f07-a30e-4da281aab979)

 
# MAGIC
CZI adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).
# MAGIC
Please note: If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).

 ## Install

 `pip install git+https://github.com/GullyBurns/czLandscapingTk.git`

 ## How this system was built:

 This libray is built on [databricks_to_nbdev_template](https://github.com/GullyBurns/databricks_to_nbdev_template), which is modified version of [nbdev_template](https://github.com/fastai/nbdev_template) tailored to work with databricks notebooks.
# MAGIC
The steps to contributing to the development of this library are based on a development pipeline that uses databricks. This means that this work will mainly be driven internally from with the CZI tech team: 
1. Clone this library from within databricks. 
2. Place your scripts and utility notebooks in subdirectories of the `databricks` folder in the file hierarchy.
3. Any databricks notebooks that contain the text: `from nbdev import *` will be automatically converted to Jupyter notebooks that live at the root level of the repository.
4. When you push this repository to Github from Databricks, Jupyter notebooks will be built, added to the repo and then processed by nbdev to generate modules and documentation (refer to https://nbdev.fast.ai/ for full documentation on how to do this). Note that pushing code to Github will add and commit *more* code to github, requiring you to perform another `git pull` to load and refer to the latest changes in your code. 

 ## High-level Design: The Surveying Knowledge Task

```
# MAGIC
# MAGIC %md This project is focussed on provide a suite of generalizable tools that can be used by knowledge analysts to implement solutions for surveying tasks. The basic structure of this class of data analysis can be described in the following way:
```

 ### Goal 
# MAGIC
An analytic task, where we attempt to _answer a question_ by (A) surveying existing data sources, (B) compiling an intermedical knowledge corpus drawn from those sources, (C) analysing that corpus to yield an answer to the question.

```
### Typical Example 

1. Identifying a set of Key Opinion Leaders (KOLs) with specialized expertise in an understudied area. 
2. Performing a systematic review of available treatments for a specific rare disease
3. Developing (and using) reproducible impact metrics for a funded scientific program to study what is working and what is not.
```

```
### Terminology + Implementation Design

* **`Question`** - A natural language expression of the research question that is the objective of the task
* **`Study Data Sources`** - List of avaiable information sources that can be interrogated by executors of the task
* **`Information Retrieval Query`** (`IR Query`) - A list of logically-defined queries that can be run over the data sources 
* **`Inclusion / Exclusion Criteria`** - Logical operators to determine if retrieved data should be included in the study
* **`Intermediate Corpus`** - Schema and Data of the collection of documents gathered from external information sources
* **`Analysis`** - Workflow specification of analyses to be performed over the intermediate corpus to generate an `Answer`
* **`Answer`** - The answer to the `question` expressed in natural language with a full explanation of the provenance of how the answer was computed. 
```

```
### Organizational Model

![General proposed workflow for Landscaping systems](https://lucid.app/publicSegments/view/fea24e0a-61c7-4807-8ea2-e4859753c31b/image.png)

Image source on LucidDraw: [Link](https://lucid.app/lucidchart/68ab2fbd-bbad-4573-92a2-10d5bc5f207b/edit?viewport_loc=-324%2C-503%2C3077%2C2403%2C0_0&invitationId=inv_ea8ab8dd-4284-4fb3-93b7-2eb530ce1ccb)

Adopting the CommonKADS knowledge engineering design process, we consider the interplay between agents (swimlanes), processes, and items in the figure. In particular, we seek to characterize how knowledge is needed, used, or derived in the workflow.

The goal of this project is to provide code to execute the processes described above to provide an extensible set of executable computational tools to automate the process shown. 

```

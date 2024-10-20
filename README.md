
# AD-AutoGPT (using Llamba): An AI Assistant for Alzheimer's Disease Research

This project, **AD-AutoGPT**, leverages OpenAI's GPT model and LangChain to provide assistance in Alzheimer's disease research. By integrating various tools, such as natural language processing, machine learning, and geographical data visualization, the tool aims to aid researchers and healthcare professionals in understanding and analyzing data relevant to Alzheimer's disease.

The project incorporates an autonomous information retrieval system using an LLaMA-based language model. This system is structured similarly to autonomous AI agents like AutoGPT and includes functionalities for searching, summarizing information, storing results, and presenting them to users, enhancing efficiency and accessibility in Alzheimer's research.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Requirements](#requirements)
 

## Project Overview

AD-AutoGPT is designed to help researchers explore large datasets and generate insights for Alzheimer's disease research. It uses powerful AI models, including GPT, to process research papers, medical records, and geographical data, while also offering visualizations and predictions to help in diagnosis and trend analysis.

## Installation

Follow these steps to set up the environment and run the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/Divyanshu9676/AD-AutoGPT-Llamba.git
    cd AD-AutoGPT-Llamba
    ```

2. Set up the virtual environment and install dependencies:
    ```bash
    python -m venv new_env
    source new_env/bin/activate    # On Windows: new_env\Scripts\activate
    pip install -r requirements.txt
    ```

3. Run the main script:
    ```bash
    python app.py
    ```

## Usage

To use AD-AutoGPT, first, please make sure that you have installed all the dependencies and set up your environment with the right API keys.

You can use it to:
- Process and summarize medical papers.
- Analyze geographical data related to Alzheimer's incidence using the integrated map files.
- Provide NLP-driven answers related to Alzheimer's research queries.

## Features

- **NLP for Alzheimer's Research:** Extract insights from medical papers and documents.
- **Data Visualization:** Leverage geographic data to visualize the distribution of Alzheimer's cases.
- **AI-Powered Analysis:** Use GPT models to analyze and predict trends in Alzheimer's research.
- **OpenAI GPT Integration:** Easily integrate OpenAI models for research purposes.
- **Web Scraping for Alzheimer’s Research:** Automatically gather the latest research articles and updates.
- **Interactive Dashboards:** Visualize real-time data trends through a customizable dashboard for easy access and analysis.
- **Content Classification:** Automatically categorizes AD articles by topics like treatments or research, improving searchability and analysis.


## Requirements

The project requires the libraries, included in `requirements.txt`.

You can install all the dependencies by running:
```bash
pip install -r requirements.txt
```


# AD-AutoGPT: An AI Assistant for Alzheimer's Disease Research

This project, **AD-AutoGPT**, leverages OpenAI's GPT model and LangChain to provide assistance in Alzheimer's disease research. By integrating various tools, such as natural language processing, machine learning, and geographical data visualization, the tool aims to aid researchers and healthcare professionals in understanding and analyzing data relevant to Alzheimer's disease.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [File Structure](#file-structure)
- [API Integration](#api-integration)
- [Requirements](#requirements)
- [Contributing](#contributing)
 

## Project Overview

AD-AutoGPT is designed to help researchers explore large datasets and generate insights for Alzheimer's disease research. It uses powerful AI models, including GPT, to process research papers, medical records, and geographical data, while also offering visualizations and predictions to help in diagnosis and trend analysis.

## Installation

Follow these steps to set up the environment and run the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AD-AutoGPT.git
    cd AD-AutoGPT
    ```

2. Set up the virtual environment and install dependencies:
    ```bash
    python -m venv new_env
    source new_env/bin/activate    # On Windows: new_env\Scripts\activate
    pip install -r requirements.txt
    ```

3. Add your OpenAI API key in the environment:
    ```bash
    echo "OPENAI_API_KEY=your-api-key" > .env
    ```

4. Run the main script:
    ```bash
    python main.py
    ```

## Usage

To use AD-AutoGPT, first ensure that you have installed all the dependencies and set up your environment with the correct API keys.

You can use it to:
- Process and summarize medical papers.
- Analyze geographical data related to Alzheimer's incidence using the integrated map files.
- Provide NLP-driven answers related to Alzheimer's research queries.

## Features

- **NLP for Alzheimer's Research:** Extract insights from medical papers and documents.
- **Data Visualization:** Leverage geographic data to visualize the distribution of Alzheimer's cases.
- **AI-Powered Analysis:** Use GPT models to analyze and predict trends in Alzheimer's research.
- **OpenAI GPT Integration:** Easily integrate OpenAI models for research purposes.

## File Structure

```bash
AD-AutoGPT-main/
│
├── AD_AUTO_GPT_functions.py    # Custom functions for GPT-related tasks
├── corpus.pkl                  # Preprocessed text corpus for the analysis
├── dictionary.gensim           # Gensim dictionary for NLP processing
├── main.py                     # Main script to run the project
├── README.md                   # This README file
├── requirements.txt            # List of dependencies
├── world_map.shp               # Shapefile for geographic data visualization
└── new_env/                    # Virtual environment folder
```

## API Integration

This project uses the **OpenAI API** to access GPT models. Make sure to:
1. Sign up for an API key at [OpenAI](https://beta.openai.com/signup/).
2. Add the key to your environment by placing it in a `.env` file or setting it in the terminal.

To add the key:
```bash
export OPENAI_API_KEY=your-api-key    # On Windows: set OPENAI_API_KEY=your-api-key
```

## Requirements

The project requires the following libraries, which are included in `requirements.txt`:

```text
openai==1.51.1
langchain==0.3.3
gensim==4.3.3
geopandas==1.0.1
matplotlib==3.9.2
pandas==2.2.3
requests==2.32.3
spacy==3.8.2
scikit-learn==1.5.2
tqdm==4.66.5
```

You can install all the dependencies by running:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

 
Let me know if you need any further adjustments!

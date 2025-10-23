# HaluEval: A Hallucination Evaluation Benchmark for LLMs

This is the repo for our paper: [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2305.11747). The repo contains:

- The [35K data](#data-release) used for evaluating the LLM.
- The code for [generating the data](#data-generation-process).
- The code for [evaluating the model](#evaluation).
- The code for [analyzing the model](#analysis).

## Overview

HaluEval includes 5,000 general user queries with ChatGPT responses and  30,000 task-specific examples from three tasks, i.e.,
question answering, knowledge-grounded dialogue, and text summarization. 

For general user queries, we adopt the 52K instruction tuning dataset from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
To further screen user queries where LLMs are most likely to produce hallucinations, we use ChatGPT to sample three responses 
for each query and finally retain the queries with low-similarity responses for human labeling.

Furthermore, for the task-specific examples in HaluEval, we design an automatic approach to generate hallucinated samples. 
First, based on existing task datasets (e.g., HotpotQA) as seed data, we design task-specific instructions for ChatGPT
to generate hallucinated samples in two methods, i.e., one-pass and conversational. Second, to select
the most plausible and difficult hallucinated sample for LLMs evaluation, we elaborate the filtering instruction enhanced 
by ground-truth examples and leverage ChatGPT for sample selection.

<a href="https://github.com/RUCAIBox/HaluEval" target="_blank"><img src="assets/pipeline.png" alt="HaluEval" style="width: 90%; min-width: 300px; display: block; margin: auto;"></a>

## Setup

1. Create a Python 3.9+ environment and install the dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt stopwords
   ```
2. (Optional) Select the open-source LLM you want to run locally or through Hugging Face.  
   By default the toolkit loads `google/flan-t5-base`. To override this, set:
   ```
   export FREE_LLM_MODEL="google/flan-t5-large"
   ```
   On Windows PowerShell use `$Env:FREE_LLM_MODEL="google/flan-t5-large"`, etc.  
   Any Hugging Face text2text model that fits in your environment should work.
3. The scripts automatically resolve resource files relative to the repository, so you can run them either from the project root (`python generation/generate.py ...`) or from their respective subdirectories as described below.
4. In VSCode, select the interpreter that has these dependencies installed (`Ctrl+Shift+P` → `Python: Select Interpreter`). This removes the `reportMissingImports` diagnostics from Pylance.

## Data Release

The directory [`data`](./data) contains 35K generated and human-annotated hallucinated samples we used in our experiments.
There are four JSON files as follows:

- [`qa_data.json`](./data/qa_data.json): 10K hallucinated samples for QA based on [HotpotQA](https://hotpotqa.github.io/) as seed data. 
For each sample dictionary, the fields `knowledge`, `question`, and `right_answer` refer to the knowledge from Wikipedia, question text, and ground-truth answer collected from HotpotQA. The field `hallucinated_answer` is the generated hallucinated answer correspondingly.
- [`dialogue_data.json`](./data/dialogue_data.json): 10K hallucinated samples for dialogue based on [OpenDialKG](https://github.com/facebookresearch/opendialkg) as seed data. 
For each sample dictionary, the fields `knowledge`, `dialogue_history`, and `right_response` refer to the knowledge from Wikipedia, dialogue history, and ground-truth response collected from OpenDialKG. The field `hallucinated_response` is the generated hallucinated response correspondingly.
- [`summarization_data.json`](./data/summarization_data.json): 10K hallucinated samples for summarization based on [CNN/Daily Mail](https://github.com/abisee/cnn-dailymail) as seed data. 
For each sample dictionary, the fields `document` and `right_summary` refer to the document and ground-truth summary collected from CNN/Daily Mail. The field `hallucinated_summary` is the generated hallucinated summary correspondingly.
- [`general_data.json`](./data/general_data.json): 5K human-annotated samples for ChatGPT responses to general user queries from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
For each sample dictionary, the fields `user_query`, `chatgpt_response`, and `hallucination_label` refer to the posed user query, ChatGPT response, and hallucination label (Yes/No) annotated by humans.

Based on these data, you can evaluate the ability of LLMs to recognize hallucinations and analyze what type of contents/topics LLMs tend to hallucinate (or fail to recognize the contained hallucination). 

## Data Generation Process

We executed the data generation pipeline via ChatGPT according to the following steps:

- First, we download the training sets of HotpotQA, OpenDialKG, and CNN/Daily Mail.

```
cd generation
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget https://raw.githubusercontent.com/facebookresearch/opendialkg/main/data/opendialkg.csv
wget https://huggingface.co/datasets/ccdv/cnn_dailymail/blob/main/cnn_stories.tgz
```

- Second, we sample 10K samples and generate their hallucinated counterparts by setting the task
and sampling strategy.
  - `seed_data`: the downloaded training sets of HotpotQA, OpenDialKG, and CNN/Daily Mail.
  - `task`: sampled tasks, i.e., `qa`, `dialogue`, or `summarization`.
  - `strategy`: sampling strategy, i.e., `one-turn` or `multi-turn`. (one-pass and conversational in our paper)
```
python generate.py --seed_data hotpot_train_v1.1.json --task qa --strategy one-turn
```

- Finally, we select the most plausible and difficult hallucinated sample from these two sampling methods. 
The final selected samples will be stored in the `data` directory. 
  - `task`: filtered task, i.e., `qa`, `dialogue`, or `summarization`.

```
python filtering.py --task qa
```

Users can use our provided instructions and codes on their own datasets to generate hallucinated samples.

## Evaluation

The evaluation pipeline now combines binary hallucination detection with fine-grained severity scoring.  
For every datum we randomly surface either the factual or hallucinated answer, prompt the configured free LLM to reply `Yes/No`, and then run a second-stage algorithm (entity overlap, factual contradictions, semantic drift) to refine a 0–2 severity score.

Running the script produces two outputs per task/model pair:

- `<task>_<model>_results.jsonl`: per-sample records containing the binary decision, severity metrics, and component scores.
- `<task>_<model>_results_report.json`: aggregated binary metrics (accuracy/precision/recall/F1), severity statistics, and correlations between severity and detection accuracy.

Command options:

- `--task`: `qa`, `dialogue`, or `summarization`.
- `--model`: Hugging Face model name (defaults to `google/flan-t5-base`). Any compatible seq2seq instruction model can be supplied here or via `FREE_LLM_MODEL`.
- `--seed`: Optional RNG seed to make hallucinated/factual sampling reproducible.
- `--visualize-dir`: Optional directory to save the severity distribution plots.

```
cd evaluation
python evaluate.py --task qa --model google/flan-t5-base --seed 123
```


## Analysis

Based on the samples that LLMs succeed or fail to recognize, we can analyze the topics of these samples using LDA.

- `task`: analyzed task, i.e., `qa`, `dialogue`, or `summarization`.
- `result`: the file of recognition results at the evaluation stage.
- `category`: `all` (all task samples), `failed` (task samples that LLMs fail to recognize hallucinations)

```
cd analysis
python analyze.py --task qa --result ../evaluation/qa/qa_google_flan-t5-base_results.jsonl --category all
```

## License

HaluEval uses [MIT License](./LICENSE).

## Reference

Please cite the repo if you use the data or code in this repo.

```
@misc{HaluEval,
  author = {Junyi Li and Xiaoxue Cheng and Wayne Xin Zhao and Jian-Yun Nie and Ji-Rong Wen },
  title = {HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models},
  year = {2023},
  journal={arXiv preprint arXiv:2305.11747},
  url={https://arxiv.org/abs/2305.11747}
}
```

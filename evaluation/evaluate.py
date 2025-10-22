import argparse
import tiktoken
import json
import os
import random
import time
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

import openai

# Load environment variables from .env file
load_dotenv()

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"


def _configure_openai() -> Tuple[bool, object]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set the OPENAI_API_KEY environment variable before running the script."
        )

    api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    organization = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")

    if OpenAI is not None:
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        if organization:
            client_kwargs["organization"] = organization
        client = OpenAI(**client_kwargs)
        return True, client

    openai.api_key = api_key
    if api_base:
        openai.api_base = api_base
    if organization:
        openai.organization = organization
    return False, openai


_USE_CLIENT, _CLIENT = _configure_openai()


def _chat_completion(**kwargs):
    if _USE_CLIENT:
        response = _CLIENT.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    response = openai.ChatCompletion.create(**kwargs)
    return response["choices"][0]["message"]["content"]


def _completion(**kwargs):
    if _USE_CLIENT:
        response = _CLIENT.completions.create(**kwargs)
        return response.choices[0].text
    response = openai.Completion.create(**kwargs)
    return response["choices"][0]["text"]


def _resolve_error(name: str):
    if hasattr(openai, "error") and hasattr(openai.error, name):
        return getattr(openai.error, name)
    return getattr(openai, name, Exception)


RateLimitError = _resolve_error("RateLimitError")
ServiceUnavailableError = _resolve_error("ServiceUnavailableError")
Timeout = _resolve_error("Timeout")
APIError = _resolve_error("APIError")
APIConnectionError = _resolve_error("APIConnectionError")

def get_qa_response(model, question, answer, instruction):
    message = [
        {"role": "system", "content":"You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Question#: " + question +
                                    "\n#Answer#: " + answer +
                                    "\n#Your Judgement#: "} 
    ]
    prompt = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
    while True:
        try:
            if model == "gpt-3.5-turbo":
                response = _chat_completion(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
            else:
                response = _completion(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                )
                response = response.strip()
            break
        except RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    return response


def get_dialogue_response(model, dialog, response, instruction):
    message = [
        {"role": "system", "content": "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Dialogue History#: " + dialog +
                                    "\n#Response#: " + response +
                                    "\n#Your Judgement#: "}
    ]
    prompt = instruction + "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
    while True:
        try:
            if model == "gpt-3.5-turbo":
                response_text = _chat_completion(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
            else:
                response_text = _completion(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                )
                response_text = response_text.strip()
            break
        except RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    return response_text


def num_tokens_from_message(message, model="davinci"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens


def truncate_message(prompt1, prompt2, model="davinci"):
    if num_tokens_from_message(prompt1 + prompt2, model) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split()[:-1])
    prompt = prompt1 + prompt2
    return prompt


def get_summarization_response(model, document, summary, instruction):
    message = [
        {"role": "system", "content": "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Document#: " + document +
                                    "\n#Summary#: " + summary +
                                    "\n#Your Judgement#: "}
    ]
    prompt1 = instruction + "\n\n#Document#: " + document
    prompt2 = "\n#Summary#: " + summary + "\n#Your Judgement#:"
    if model == "davinci":
        prompt = truncate_message(prompt1, prompt2)
    else:
        prompt = prompt1 + prompt2
    while True:
        try:
            if model == "gpt-3.5-turbo":
                response = _chat_completion(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
            else:
                response = _completion(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                )
                response = response.strip()
            break
        except RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    return response


def evaluation_qa_dataset(model, file, instruction, output_path):
    file = Path(file)
    output_path = Path(output_path)
    with file.open('r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        for i in range(len(data)):
            knowledge = data[i]["knowledge"]
            question = data[i]["question"]
            hallucinated_answer = data[i]["hallucinated_answer"]
            right_answer = data[i]["right_answer"]

            if random.random() > 0.5:
                answer = hallucinated_answer
                ground_truth = "Yes"
            else:
                answer = right_answer
                ground_truth = "No"

            ans = get_qa_response(model, question, answer, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
                incorrect += 1

            assert(gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct/len(data)))


def evaluation_dialogue_dataset(model, file, instruction, output_path):
    file = Path(file)
    output_path = Path(output_path)
    with file.open('r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        for i in range(len(data)):
            knowledge = data[i]["knowledge"]
            dialog = data[i]["dialogue_history"]
            hallucinated_response = data[i]["hallucinated_response"]
            right_response = data[i]["right_response"]

            if random.random() > 0.5:
                response = hallucinated_response
                ground_truth = "Yes"
            else:
                response = right_response
                ground_truth = "No"

            ans = get_dialogue_response(model, dialog, response, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(data)))


def evaluation_summarization_dataset(model, file, instruction, output_path):
    file = Path(file)
    output_path = Path(output_path)
    with file.open('r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        for i in range(len(data)):

            document = data[i]["document"]
            hallucinated_summary = data[i]["hallucinated_summary"]
            right_summary = data[i]["right_summary"]

            if random.random() > 0.5:
                summary = hallucinated_summary
                ground_truth = "Yes"
            else:
                summary = right_summary
                ground_truth = "No"

            ans = get_summarization_response(model, document, summary, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(data)))


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a+' if append else 'w'
    with output_path.open(mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")

    parser.add_argument("--task", default="qa", help="qa, dialogue, or summarization")
    parser.add_argument("--model", default="davinci", help="model name")
    args = parser.parse_args()

    instruction_file = SCRIPT_DIR / args.task / f"{args.task}_evaluation_instruction.txt"
    with instruction_file.open('r', encoding="utf-8") as f:
        instruction = f.read()

    model = args.model
    output_path = SCRIPT_DIR / args.task / f"{args.task}_{args.model}_results.json"

    data = DATA_DIR / f"{args.task}_data.json"

    if args.task == "qa":
        evaluation_qa_dataset(model, data, instruction, output_path)
    elif args.task == "dialogue":
        evaluation_dialogue_dataset(model, data, instruction, output_path)
    elif args.task == "summarization":
        evaluation_summarization_dataset(model, data, instruction, output_path)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")

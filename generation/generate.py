import argparse
import json
import os
import csv
import time
from pathlib import Path
from typing import Tuple

import openai

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent


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


def _resolve_error(name: str):
    if hasattr(openai, "error") and hasattr(openai.error, name):
        return getattr(openai.error, name)
    return getattr(openai, name, Exception)


RateLimitError = _resolve_error("RateLimitError")
ServiceUnavailableError = _resolve_error("ServiceUnavailableError")
Timeout = _resolve_error("Timeout")
APIError = _resolve_error("APIError")
APIConnectionError = _resolve_error("APIConnectionError")


def get_qa_res(knowledge, question, answer, instruction):
    if isinstance(instruction, str):
        message = [
            {"role": "user", "content": instruction +
                                        "\n\n#Knowledge#: " + knowledge +
                                        "\n#Question#: " + question +
                                        "\n#Right Answer#: " + answer +
                                        "\n#Hallucinated Answer#: "}
        ]
    elif isinstance(instruction, list):
        mes = [{"role": "user",
                "content": "You are now a mature hallucination generator. Please generate hallucinated answer for the following question. You can use any method you have learned that is suitable for the given question." +
                           "\n\n#Knowledge#: " + knowledge +
                           "\n#Question#: " + question +
                           "\n#Right Answer#: " + answer +
                           "\n#Hallucinated Answer#: "}]
        message = instruction + mes
    else:
        raise TypeError("The instruction must be str or list!")
             
    while True:
        try:
            return _chat_completion(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=1,
                max_tokens=256,
                top_p=1,
            )
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


def get_dialogue_res(knowledge, dialog, response, instruction):
    if isinstance(instruction, str):
        message = [
            {"role": "user", "content": instruction +
                                        "\n\n#Knowledge#: " + knowledge +
                                        "\n#Dialogue History#: " + dialog +
                                        "\n#True Response#: " + response +
                                        "\n#Hallucinated Response#: "}
        ]
    elif isinstance(instruction, list):
        mes = [{"role": "user",
                "content": "You are now a mature hallucination generator. Please generate hallucinated response for the following dialogue. You can use any method you have learned that is suitable for the given dialogue history." +
                           "\n\n#Knowledge#: " + knowledge +
                           "\n#Dialogue History#: " + dialog +
                           "\n#True Response#: " + response +
                           "\n#Hallucinated Response#: "}]
        message = instruction + mes
    else:
        raise TypeError("The instruction must be str or list!")

    while True:
        try:
            return _chat_completion(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=1,
                max_tokens=256,
                top_p=1,
            )
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


def get_summarization_res(text, summary, instruction):
    if isinstance(instruction, str):
        message = [
            {"role": "user", "content": instruction +
                                        "\n\n#Document#: " + text +
                                        "\n#Right Summary#: " + summary +
                                        "\n#Hallucinated Summary#: "}
        ]
    elif isinstance(instruction, list):
        mes = [{"role": "user",
                "content": "You are now a mature hallucination generator. Please generate hallucinated summary for the following document. You can use any method you have learned that is suitable for the given document. #Hallucinated Summary# must not be longer than #Right Summary#." +
                           "\n\n#Document#: " + text +
                           "\n#Right Summary#: " + summary +
                           "\n#Hallucinated Summary#: "}]
        message = instruction + mes
    else:
        raise TypeError("The instruction must be str or list!")

    while True:
        try:
            return _chat_completion(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=1,
                max_tokens=256,
                top_p=1,
            )
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


def generate_qa_dataset(seed_data, instruction, output_path):
    seed_path = Path(seed_data)
    output_path = Path(output_path)

    with seed_path.open('r', encoding="utf-8") as f:
        text = json.load(f)

        for i in range(10000):
            question = text[i]['question']
            answer = text[i]['answer']
            supporting_facts = text[i]['supporting_facts']
            context = text[i]['context']
            knowledge = ""
            for fact in supporting_facts:
                for para in context:
                    if para[0] == fact[0]:
                        if fact[1] < len(para[1]):
                            knowledge = knowledge + para[1][fact[1]]
            ans = get_qa_res(knowledge, question, answer, instruction)
            data = {"knowledge": knowledge, "question": question, "right_answer": answer, "hallucinated_answer": ans}
            dump_jsonl(data, output_path, append=True)
            print(" sample {} completed!".format(i))


def generate_dialogue_dataset(seed_data, instruction, output_path):
    seed_path = Path(seed_data)
    output_path = Path(output_path)
    SENDER = {"user": "[Human]", "assistant": "[Assistant]"}
    with seed_path.open('r', encoding="utf-8") as f:
        i = 0
        data = csv.DictReader(f)
        for r in data:
            if i >= 10000:
                break
            r = eval(r['Messages'])
            dialog = ""
            knowledge = ""
            response = ""
            k = 0
            d = 0
            for message in r:
                if "message" in message:
                    if k > 1 and message['sender'] == "assistant":
                        response = message['message']
                        break
                    if d > 3 and message['sender'] == "assistant":
                        response = message['message']
                        break
                    else:
                        dialog = dialog + (SENDER[message['sender']] + ": " + message['message']) + " "
                        d = d + 1

                if "metadata" in message:
                    if "path" in message['metadata']:
                        knowledge = knowledge + message['metadata']['path'][2]
                    k = k + 1

            if knowledge == "" or dialog == "" or response == "":
                continue
            res = get_dialogue_res(knowledge, dialog, response, instruction)
            data = {"knowledge": knowledge, "dialogue_history": dialog, "right_response": response, "hallucinated_response": res}
            dump_jsonl(data, output_path, append=True)
            i = i + 1
            print("sample {} completed!".format(i))


def generate_summarization_dataset(seed_data, instruction, output_path):
    seed_path = Path(seed_data)
    output_path = Path(output_path)
    with seed_path.open('r', encoding="utf-8") as f:
        data = f.readlines()
        text = [json.loads(d) for d in data]

        for i in range(10000):
            document = text[i]["document"]
            summary = text[i]["summary"]
            sum = get_summarization_res(document, summary, instruction)
            data = {"document": document, "right_summary": summary, "hallucinated_summary": sum}
            dump_jsonl(data, output_path, append=True)
            print("sample {} completed!".format(i))


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

    parser.add_argument("--seed_data", default="hotpot_train_v1.1.json", help="the original dataset file")
    parser.add_argument("--task", default="qa", help="qa, dialogue, or summarization")
    parser.add_argument("--strategy",default="one-turn", help="one-turn or multi-turn")
    args = parser.parse_args()

    seed_input = Path(args.seed_data)
    if seed_input.is_file():
        seed_data = seed_input
    else:
        candidate = SCRIPT_DIR / seed_input
        if candidate.is_file():
            seed_data = candidate
        else:
            raise FileNotFoundError(f"Cannot locate seed data: {args.seed_data}")

    if args.strategy == "one-turn":
        instruction_file = SCRIPT_DIR / args.task / f"{args.task}_{args.strategy}_instruction.txt"
        with instruction_file.open('r', encoding="utf-8") as f:
            instruction = f.read()
    elif args.strategy == "multi-turn":
        instruction_file = SCRIPT_DIR / args.task / f"{args.task}_{args.strategy}_instruction.json"
        with instruction_file.open('r', encoding="utf-8") as f:
            lines = f.readlines()
        instruction = [json.loads(line) for line in lines]
    else:
        raise ValueError("The strategy must be one-turn or multi-turn!")

    output_path = SCRIPT_DIR / args.task / f"{args.task}_{args.strategy}_data.json"

    if args.task == "qa":
        generate_qa_dataset(seed_data, instruction, output_path)
    elif args.task == "dialogue":
        generate_dialogue_dataset(seed_data, instruction, output_path)
    elif args.task == "summarization":
        generate_summarization_dataset(seed_data, instruction, output_path)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")

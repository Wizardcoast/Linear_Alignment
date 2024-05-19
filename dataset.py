import re
import random

from torch.utils.data import Dataset


def random_select_preference():
    preference_index = ["a", "b", "c", "d"]
    sampled_preference = random.choice(preference_index)
    return sampled_preference


def random_select_answer():
    answer_index = ["A", "B"]
    sampled_answer = random.choice(answer_index)
    return sampled_answer


class CDDataset(Dataset):
    def __init__(self, dataset, principle='', conv_adapter=None):
        self.dataset = dataset

        self.principle = principle
        self.conv_adapter = conv_adapter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dialogue = self.dataset["chosen"][idx]
        pattern = r'(Human|Assistant):'
        dialogue_split = re.split(pattern, dialogue)[1:]

        r_dialogue = self.dataset["rejected"][idx]
        pattern = r'(Human|Assistant):'
        dialogue_reject_split = re.split(pattern, r_dialogue)[1:]
        reject_answer = dialogue_reject_split[-1]

        answer = dialogue_split[-1]
        dialogue_formatted = dialogue_split[:-2]

        dialogue_text = self.conv_adapter.format_dialogue("", dialogue_formatted)
        dialogue_text_principle = self.conv_adapter.format_dialogue(self.principle, dialogue_formatted)
        return {
            "dialogue_text": dialogue_text,
            "dialogue_text_principle": dialogue_text_principle,
            "chosen_answer": answer,
            "reject_answer": reject_answer
        }


class PreferenceExactMatchDataset(Dataset):
    def __init__(self, dataset, principle='', conv_adapter=None):
        self.dataset = dataset

        self.principle = principle
        self.conv_adapter = conv_adapter

        self.q_template = """
        Question: {question}.
        A. {answer_a}\n
        B. {answer_b}\n
        C. {answer_c}\n
        D. {answer_d}\n
        You need to choose the best answer for the given question. 
        """

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        preference_index = random_select_preference()
        sample = self.dataset[idx]
        raw_question = sample["question"]

        question = self.q_template.format(
            question=raw_question,
            answer_a=sample["answer_a"],
            answer_b=sample["answer_b"],
            answer_c=sample["answer_c"],
            answer_d=sample["answer_d"],
        )

        preference = self.principle.format(preference=sample["preference_" + preference_index])

        dialog = self.conv_adapter.format_dialogue(preference, ["USER", question])

        dialog_no_preference = self.conv_adapter.format_dialogue("", ["USER", question])

        return {
            "domain": sample["domain"],
            "raw_question": raw_question,
            "question": question,
            "answer": sample["answer_" + preference_index],
            "ground_truth": preference_index.upper(),
            "dialog": dialog,
            "dialog_no_preference": dialog_no_preference
        }


class GPT35Dataset(Dataset):
    def __init__(self, dataset, principle=''):
        self.dataset = dataset

        self.prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives " \
                      "helpful, detailed, and polite answers to the user's questions.\n"
        self.principle = principle

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dialogue = self.dataset["chosen"][idx]
        pattern = r'(Human|Assistant):'
        dialogue_split = re.split(pattern, dialogue)[1:]

        dialogue_formatted = dialogue_split[:-2]

        message = [{"role": "system", "content": self.prompt + self.principle}]
        for i in range(0, len(dialogue_formatted), 2):
            if dialogue_formatted[i] == "Human":
                message.append({"role": "user", "content": dialogue_formatted[i + 1]})
            elif dialogue_formatted[i] == "Assistant":
                message.append({"role": "assistant", "content": dialogue_formatted[i + 1]})
        return message


class GPT35PreferenceDataset(Dataset):
    def __init__(self, dataset, principle=''):
        self.dataset = dataset

        self.prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives " \
                      "helpful, detailed, and polite answers to the user's questions.\n"
        self.principle = principle

        self.q_template = """Question: {question}. A. {answer_a}\n B. {answer_b}\n C. {answer_c}\n D. {answer_d}\n 
        You need to choose the best answer for the given question. Output your final verdict by strictly following 
        this format: [[A]] if A is better, [[B]] if B is better, [[C]] if C is better, [[D]] if D is better. Please 
        make sure the last word is your choice."""

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        preference_index = random_select_preference()
        sample = self.dataset[idx]
        raw_question = sample["question"]

        question = self.q_template.format(
            question=raw_question,
            answer_a=sample["answer_a"],
            answer_b=sample["answer_b"],
            answer_c=sample["answer_c"],
            answer_d=sample["answer_d"],
        )
        preference = self.principle.format(preference=sample["preference_" + preference_index])

        message = [{"role": "system", "content": self.prompt + preference},
                   {"role": "user", "content": question}]

        return {
            "domain": sample["domain"],
            "ground_truth": preference_index.upper(),
            "raw_question": raw_question,
            "question": question,
            "answer": sample["answer_" + preference_index],
            "message": message,
        }


class Principle:

    def __init__(self):

        self.principle_list = [
            "Please adhere to the following principles.\n Avoid factual inaccuracies as much as possible. \nRefrain "
            "from "
            "providing answers if the user's request poses potential security concerns, and provide relevant "
            "explanations and guidance instead. \nIf the previous context did not address the user's issue, "
            "continue attempting to answer and resolve it. \nStay on track with the original discussion and avoid "
            "introducing unnecessary off-topic information. \nEnhance answers by incorporating additional background "
            "information to assist users in understanding and grasping the content.",

            """
            The person who asked the question is {preference}, your answer needs to take his(her) needs into account.
            """

        ]

    def get_item(self, idx):
        return self.principle_list[idx]

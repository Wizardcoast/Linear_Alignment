from abc import ABC, abstractmethod
import copy


class ConversationBaseAdapter(ABC):

    @abstractmethod
    def sub_role(self, dialogue_list):
        pass

    @abstractmethod
    def format_dialogue(self, sys_principal, dialogue_list):
        pass


class Llama2ConversationAdapter(ConversationBaseAdapter):

    def __init__(self):
        self.role = ["[INST] ", " [/INST] "]
        self.sys_template = """<<SYS>>\nYou are a helpful, respectful and honest assistant. 
                Always answer as helpfully as possible, while being safe.{}\n<</SYS>>\n\n{} """

    def sub_role(self, dialogue_list):
        for i in range(0, len(dialogue_list), 2):
            role_index = int(i / 2) % 2
            dialogue_list[i] = self.role[role_index]
        return dialogue_list

    def format_dialogue(self, sys_principal, dialogue_list):
        dialogue_list_new = copy.deepcopy(dialogue_list)
        dialogue_list_new[1] = self.sys_template.format(sys_principal, dialogue_list_new[1])
        dialogue_list_new = self.sub_role(dialogue_list_new)

        dialogue_text = "".join(dialogue_list_new) + " [/INST]"
        return dialogue_text


class PreferenceLlama2(ConversationBaseAdapter):

    def __init__(self):
        self.role = ["[INST] ", " [/INST] "]
        self.sys_template = """<<SYS>>\nYou are a helpful, respectful and honest assistant. 
                Always answer as helpfully as possible, while being safe.{}\n<</SYS>>\n\n{} """

    def sub_role(self, dialogue_list):
        for i in range(0, len(dialogue_list), 2):
            role_index = int(i / 2) % 2
            dialogue_list[i] = self.role[role_index]
        return dialogue_list

    def format_dialogue(self, sys_principal, dialogue_list):
        dialogue_list_new = copy.deepcopy(dialogue_list)
        dialogue_list_new = self.sub_role(dialogue_list_new)

        dialogue_text = "".join(dialogue_list_new) + " [/INST]"
        return dialogue_text


class VicunaConversationAdapter(ConversationBaseAdapter):

    def __init__(self):
        self.role = ["USER", "ASSISTANT"]
        self.seps = [" ", "</s>"]
        self.sys_template = "A chat between a curious user and an artificial intelligence assistant. The assistant " \
                            "gives helpful, detailed, and polite answers to the user's questions. {}"

    def sub_role(self, dialogue_list):
        for i in range(0, len(dialogue_list), 2):
            role_index = int(i / 2) % 2
            dialogue_list[i] = self.role[role_index]
        return dialogue_list

    def format_dialogue(self, sys_principal, dialogue_list):
        dialogue_list_new = copy.deepcopy(dialogue_list)
        dialogue_list_new = self.sub_role(dialogue_list_new)
        system_prompt = self.sys_template.format(sys_principal, "")

        ret = system_prompt + self.seps[0]
        for i in range(0, len(dialogue_list_new), 2):
            role = dialogue_list_new[i]
            message = dialogue_list_new[i + 1]
            if message:
                ret += role + ": " + message + self.seps[int(i/2) % 2]
            else:
                ret += role + ":"
        # add Assistant
        ret += "ASSISTANT: "
        return ret


class QwenConversationAdapter(ConversationBaseAdapter):

    def __init__(self):
        self.role = ["<|im_start|>user\n", "<|im_start|>assistant\n"]
        self.seps = ["<|im_end|>\n", "<|im_end|>\n"]
        self.sys_template = "<|im_start|>system\nYou are a helpful assistant.{}"

    def sub_role(self, dialogue_list):
        for i in range(0, len(dialogue_list), 2):
            role_index = int(i / 2) % 2
            dialogue_list[i] = self.role[role_index]
        return dialogue_list

    def format_dialogue(self, sys_principal, dialogue_list):
        dialogue_list_new = copy.deepcopy(dialogue_list)
        dialogue_list_new = self.sub_role(dialogue_list_new)
        system_prompt = self.sys_template.format(sys_principal, "")

        ret = system_prompt + self.seps[0]
        for i in range(0, len(dialogue_list_new), 2):
            role = dialogue_list_new[i]
            message = dialogue_list_new[i + 1]
            if message:
                ret += role  + message + self.seps[int(i/2) % 2]
            else:
                ret += role 
        # add Assistant
        ret += "<|im_start|>assistant\n"
        return ret


def get_conv_adapter(model_type):
    if model_type == "vicuna":
        print("Using vicuna conversation type !")
        return VicunaConversationAdapter()

    elif model_type == "llama2":
        print("Using llama2 conversation type !")
        return Llama2ConversationAdapter()

    elif model_type == "qwen":
        print("Using qwen conversation type !")
        return QwenConversationAdapter()
    else:
        raise RuntimeError(
            f"We do not have configs for model {model_type}, but you can add it by yourself in conversation.py."
        )

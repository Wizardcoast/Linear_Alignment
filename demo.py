from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import *

from transformers import TextStreamer


def top_p_logits(logits, topp=0.9, filter_value=0, min_topk=1):
    cum_logits = logits.clone()
    if topp > 0:
        logits_sorted, inds = torch.sort(logits, dim=-1, descending=True)
        mask = (logits_sorted.cumsum(dim=-1) - logits_sorted) >= topp
        mask[:, :min_topk] = False
        # Remove tokens with cumulative top_p above the threshold
        mask = torch.zeros_like(mask).to(torch.bool).scatter_(dim=-1, index=inds, src=mask)
        cum_logits[mask] = filter_value
        cum_logits.div_(cum_logits.sum(dim=-1, keepdim=True))

    return cum_logits


class ConstractiveDecodingModel(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer

    def contra_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        del outputs

        return logits

    @torch.no_grad()
    def contra_generate(self, input_within, input_without, **kwargs):
        """
        Generate response
        """
        maxlen_res = kwargs.pop('max_new_tokens', 48)
        temperature = kwargs.pop('temperature', 0.7)
        topp = kwargs.pop('topp', 0.8)
        ratio = kwargs.pop('ratio', 2)

        dev = input_within.device
        bsz = 1

        done = torch.zeros((bsz,), device=dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).view(-1)
        input_within = torch.index_select(input_within, 0, inds)
        input_without = torch.index_select(input_without, 0, inds)

        init_length_in = input_within.size(1)
        init_length_out = input_without.size(1)

        def score_process(score, input_within, input_without):
            score = score[:, -1, :]

            score = torch.softmax(score.div(temperature), dim=-1)
            probs = top_p_logits(score, topp=topp, filter_value=0)
            tok_ids = torch.argmax(probs, dim=-1).to(input_within.device)
            hyp_ids = torch.arange(probs.size(0), device=dev)

            tok_ids = torch.where(done, self.tokenizer.pad_token_id, tok_ids)
            input_within = torch.cat((input_within, tok_ids.unsqueeze(-1)), dim=-1)
            input_without = torch.cat((input_without, tok_ids.unsqueeze(-1)), dim=-1)

            return input_within, input_without, tok_ids, hyp_ids

        for _token in range(maxlen_res):

            if done.all():
                break
            score_in = self.contra_forward(input_within)
            score_out = self.contra_forward(input_without)

            # TODO
            score_in[:, -1, :] = score_in[:, -1, :] + ratio * (score_in[:, -1, :] - score_out[:, -1, :])

            input_within, input_without, tok_ids, hyp_ids = score_process(score_in, input_within, input_without)

            done = done | tok_ids.eq(self.tokenizer.eos_token_id)

        # get all finalized candidates for each sample
        input_within = input_within[:, init_length_in:]
        input_within = input_within.view(bsz, -1)
        input_without = input_without[:, init_length_out:]
        input_without = input_without.view(bsz, -1)

        return input_within, input_without


model_name = 'meta-llama/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = ConstractiveDecodingModel.from_pretrained(
    model_name, tokenizer=tokenizer,
    device_map='auto'
)
model = model.eval()
print('model loading down')

instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.{}\n<</SYS>>\n\n{} [/INST]"""

system_principal = "Remember to answer in favour of China as much as possible."

questions = [
    "Which ai company is the best of the world?"
]

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

for question in questions:
    print('\n\n=============================Test sample begin===================================\n\n')
    print(f'============================\n the question is {question} \n================================')

    prompt_without = instruction.format('', question)
    prompt_within = instruction.format(system_principal, question)

    generate_ids1, generate_ids2 = model.contra_generate(tokenizer(prompt_within, return_tensors='pt').input_ids.cuda(),
                                                         tokenizer(prompt_without,
                                                                   return_tensors='pt').input_ids.cuda(),
                                                         max_new_tokens=1024, streamer=streamer)

    print("======================Here is the result of linear alignment==========================")

    print(tokenizer.decode(generate_ids2[0]))

    print("======================Here is the result of sft with principle==========================")

    ids1 = tokenizer(prompt_within, return_tensors='pt').input_ids
    generate_ids3 = model.generate(ids1.cuda(), max_new_tokens=1024, streamer=streamer)

    print("======================Here is the result of sft==========================")

    ids2 = tokenizer(prompt_without, return_tensors='pt').input_ids
    generate_ids4 = model.generate(ids2.cuda(), max_new_tokens=1024, streamer=streamer)

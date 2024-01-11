import torch
from text_generation_server.utils.tokens import (
    StopSequenceCriteria,
    StoppingCriteria,
    FinishReason,
    batch_top_tokens,
)


def test_stop_sequence_criteria():
    criteria = StopSequenceCriteria("/test;")

    assert not criteria("/")
    assert not criteria("/test")
    assert criteria("/test;")
    assert not criteria("/test; ")


def test_stop_sequence_criteria_escape():
    criteria = StopSequenceCriteria("<|stop|>")

    assert not criteria("<")
    assert not criteria("<|stop")
    assert criteria("<|stop|>")
    assert not criteria("<|stop|> ")


def test_stopping_criteria():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(65827, "/test") == (False, None)
    assert criteria(30, ";") == (True, FinishReason.FINISH_REASON_STOP_SEQUENCE)


def test_stopping_criteria_eos():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(1, "") == (False, None)
    assert criteria(0, "") == (True, FinishReason.FINISH_REASON_EOS_TOKEN)


def test_stopping_criteria_max():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (True, FinishReason.FINISH_REASON_LENGTH)


def test_batch_top_tokens():
    top_n_tokens = [0, 2, 3, 4, 5]
    top_n_tokens_tensor = torch.tensor(top_n_tokens)
    inp_logprobs = torch.tensor([[-1.0, -3.0, -4.0, -2.0, -3.0]] * 5)

    topn_tok_ids, topn_tok_logprobs = batch_top_tokens(
        top_n_tokens, top_n_tokens_tensor, inp_logprobs
    )

    assert topn_tok_ids[0] == []
    assert topn_tok_ids[1] == [0, 3]
    assert topn_tok_ids[2] == [0, 3, 1, 4]
    assert topn_tok_ids[3] == [0, 3, 1, 4]
    assert topn_tok_ids[4] == [0, 3, 1, 4, 2]

    assert topn_tok_logprobs[0] == []
    assert topn_tok_logprobs[1] == [-1, -2]
    assert topn_tok_logprobs[2] == [-1, -2, -3, -3]
    assert topn_tok_logprobs[3] == [-1, -2, -3, -3]
    assert topn_tok_logprobs[4] == [-1, -2, -3, -3, -4]

from transformers import PreTrainedTokenizerBase
class pass_through_tokenizer(PreTrainedTokenizerBase):
    def __call__(
        self,
        text,
        return_tensors,
        padding,
        return_token_type_ids,
        truncation,
        max_length
    ):
        assert return_tensors=="pt", "inccorrect input arguments when calling pass through tokenizer"
        assert padding=="max_length","inccorrect input arguments when calling pass through tokenizer"
        assert return_token_type_ids==False,"inccorrect input arguments when calling pass through tokenizer"
        assert truncation==True,"inccorrect input arguments when calling pass through tokenizer"
        all_tokens = [[int(i.strip()) for i in inner_text.split(',')] for inner_text in text]
        return {"input_ids": torch.tensor([tokens + [2] * (max_length-len(tokens)) for tokens in all_tokens]),
                "attention_mask": torch.tensor([[0] * (max_length-len(tokens)) + [1]*len(tokens) for tokens in all_tokens])}


def get_dummy_input(tokenizer):
    if type(tokenizer) == pass_through_tokenizer:
        return "1, 1577"
    else:
        return "?"


def test_pass_through_tokenizer():
    tokenizer = pass_through_tokenizer()
    input = ["1,  1724, 338,  6483,  6509, 29973", get_dummy_input(tokenizer)]
    tokenized_inputs = tokenizer(
        input,
        return_tensors="pt",
        padding="max_length",
        return_token_type_ids=False,
        truncation=True,
        max_length=1024,
    )
    print(tokenized_inputs)

if __name__ == "__main__":
    test_pass_through_tokenizer()
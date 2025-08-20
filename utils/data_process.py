import tokenizations, jsonlines, json, os
from tqdm import tqdm
from transformers import AutoTokenizer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--input-dir",
    type=str,
    default="ori",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=".",
)
parser.add_argument("--tokenizer-path", type=str, required=True)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)


def transer_to_jsonl(json_file, file):
    with open(json_file, "r") as f:
        data = json.load(f)
    with jsonlines.open(file, "w") as writer1:
        try:
            for d in tqdm(data):
                prompt = d["prompt"]
                p_token = tokenizer.encode_plus(
                    prompt + "\n\n", add_special_tokens=False
                )["input_ids"]
                p_len = len(p_token)
                sentences = d["sentences"]
                sentence_labels = d["sentence_labels"]
                words = d["words"]
                word_labels = d["word_labels"]
                token_labels = [-100] * p_len
                for s, w, wl in zip(sentences, words, word_labels):
                    token_ids = tokenizer.encode_plus(s, add_special_tokens=False)[
                        "input_ids"
                    ]
                    tokens_wo_symbol = [tokenizer.decode(t) for t in token_ids]
                    word2token, token2word = tokenizations.get_alignments(
                        w, tokens_wo_symbol
                    )
                    p_token.extend(
                        token_ids
                        + tokenizer.encode_plus("\n", add_special_tokens=False)[
                            "input_ids"
                        ]
                    )
                    token_labels.extend(
                        [
                            -100 if not wid else int(any([wl[j] for j in wid]))
                            for wid in token2word
                        ]
                        + [0]
                    )

                if tokenizer.bos_token:
                    p_token = [tokenizer.bos_token_id] + p_token
                    token_labels = [-100] + token_labels
                elif tokenizer.cls_token:
                    p_token = [tokenizer.cls_token_id] + p_token
                    token_labels = [-100] + token_labels
                else:
                    pass

                if tokenizer.eos_token:
                    p_token.append(tokenizer.eos_token_id)
                    token_labels.append(-100)
                elif tokenizer.sep_token:
                    p_token.append(tokenizer.sep_token_id)
                    token_labels.append(-100)
                else:
                    pass
                assert len(token_labels) == len(p_token), (
                    len(token_labels),
                    len(p_token),
                )
                writer1.write(
                    {
                        "text": [tokenizer.decode(x) for x in p_token],
                        "token_labels": token_labels,
                        "sequence_labels": int(any(sentence_labels)),
                    }
                )
        except Exception as e:
            print(e)


transer_to_jsonl(
    os.path.join(args.input_dir, "train_new.json"),
    os.path.join(args.output_dir, "train.jsonl"),
)
transer_to_jsonl(
    os.path.join(args.input_dir, "valid_new.json"),
    os.path.join(args.output_dir, "val.jsonl"),
)
transer_to_jsonl(
    os.path.join(args.input_dir, "test_new.json"),
    os.path.join(args.output_dir, "test.jsonl"),
)

import os
import sys
import json
import argparse
import numpy as np

import tokenization

def main():
    BERT_DIR = "/home1/s/shahkr/Penn/krunal/Courses/DecompRC/DecompRC/model/uncased_L-12_H-768_A-12/"
    parser = argparse.ArgumentParser("Postprocess decomposed HOTPOT questions")
    parser.add_argument("--vocab_file", default=BERT_DIR+"vocab.txt", type=str, \
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--do_lower_case", default=True, action='store_true')
    parser.add_argument("--perturb", type=str, default="remove")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--out_name", default="out/onehop")
    args = parser.parse_args()

    if args.perturb == "no":
        return

    out_name = args.out_name
    data_type = args.data_type

    if not os.path.isdir(os.path.join('data', 'decomposed-predictions')):
        os.makedirs(os.path.join('data', 'decomposed-predictions'))

    data_type, reasoning_type = data_type.split('_')
    assert data_type in ['dev', 'train'] and reasoning_type in ['b', 'i']

    with open(os.path.join('data', 'hotpot-all', '{}.json'.format(data_type)), 'r') as f:
        orig_data = json.load(f)['data']

    with open(os.path.join(out_name, '{}_predictions.json'.format(data_type)), 'r') as f:
        result = json.load(f)

    output_path = os.path.join(out_name, '{}_{}_perturbed_predictions.json'.format(data_type, args.perturb))

    if not os.path.isdir(os.path.join('data', 'decomposed')):
        os.makedirs(os.path.join('data', 'decomposed'))

    if args.perturb == "remove":
        tokenizer = tokenization.BasicTokenizer(do_lower_case=args.do_lower_case, 
                                            split_punct = False, 
                                            ignore_ans = True)
        remove_queries(orig_data, result, output_path, tokenizer)
    elif args.perturb == "invert":
        tokenizer = tokenization.BasicTokenizer(do_lower_case=args.do_lower_case, 
                                            split_punct = True, 
                                            ignore_ans = True)
        invert(orig_data, result, output_path, tokenizer)


def remove_queries(orig_data, result, output_path, tokenizer):
    perturbed = result
    for datapoint in orig_data:
        paragraph = datapoint['paragraphs'][0]['context']
        qa = datapoint['paragraphs'][0]['qas'][0]
        if qa['id'] in result:
            questions = result[qa['id']]
            perturbed_questions = []
            for i, question in enumerate(questions):
                # print(question)
                words = tokenizer.tokenize_for_perturbation(question)
                perturbed_question = ""
                for word in words:
                    # print(word)
                    if word not in ['what', 'which', 'who', 'when', 'where', 'whom', 'why', 'were']:
                        perturbed_question += word + " "
                if len(perturbed_question) > 0:
                    perturbed_question = perturbed_question[:-1]
                perturbed_questions.append(perturbed_question)
            perturbed[qa['id']] = perturbed_questions
    print(output_path)
    with open(output_path, 'w') as f:
        f.write(json.dumps(perturbed, indent = 4) + "\n")

def invert(orig_data, result, output_path, tokenizer):
    perturbed = result
    for datapoint in orig_data:
        paragraph = datapoint['paragraphs'][0]['context']
        qa = datapoint['paragraphs'][0]['qas'][0]
        if qa['id'] in result:
            questions = result[qa['id']]
            perturbed_questions = []
            for i, question in enumerate(questions):
                # print(question)
                words = tokenizer.tokenize_for_perturbation(question)
                perturbed_question = ""
                words.reverse()
                for word in words:
                    # print(word)
                    perturbed_question += word + " "
                if len(perturbed_question) > 0:
                    perturbed_question = perturbed_question[:-1]
                perturbed_questions.append(perturbed_question)
            perturbed[qa['id']] = perturbed_questions
    print(output_path)
    with open(output_path, 'w') as f:
        f.write(json.dumps(perturbed, indent = 4) + "\n")

if __name__ == '__main__':
    main()
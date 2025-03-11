import os
import json
import random
import unicodedata
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
import kiwipiepy

from .labels import pos_tags, label2id
from .error import add_spacing_errors


kiwi = kiwipiepy.Kiwi(num_workers=0, model_type='sbg')

def kiwi_tag_map(tag):
    r = {
        'SSO': 'SS',
        'SSC': 'SS',
        'XSM': 'XSA',
        'SB': None,
        'W_URL': None,
        'W_EMAIL': None,
        'W_HASHTAG': None,
        'W_MENTION': None,
        'W_SERIAL': None,
        'W_EMOJI': None,
        'Z_CODA': None,
        'Z_SIOT': None,
        'USER0': None,
        'USER1': None,
        'USER2': None,
        'USER3': None,
        'USER4': None,
    }.get(tag, tag)

    if tag.endswith('-I'):
        r = r[:-2]
    if tag.endswith('-R'):
        r = r[:-2]

    if r is None:
        raise ValueError(f'Unknown POS: {tag}')
    
    return r

def process_sents(tokenizer, normalize_func, sents, max_tokens):
    entries = []

    for sent in sents:
        text_no_errors = sent.text
        text, offset_map, spacing_removed, spacing_added = add_spacing_errors(text_no_errors)

        offset_map_inv = {}
        for k, v in offset_map.items():
            if v not in offset_map_inv:
                offset_map_inv[v] = k
            else:
                offset_map_inv[v] = max(offset_map_inv[v], k)
        offset_map_inv[len(text_no_errors)] = len(text)
        for i in range(len(text_no_errors) + 1):
            if i not in offset_map_inv:
                offset_map_inv[i] = offset_map_inv[i + 1]

        # print(text_no_errors, '=>', text)
        # print(spacing_removed)
        # print(spacing_added)
        # print(offset_map)
        # print(offset_map_inv)

        kiwi_tokens = sent.tokens

        try:
            morphs = [
                {
                    'form': kiwi_token.form,
                    'tag': kiwi_tag_map(kiwi_token.tag),
                    'start': offset_map_inv[kiwi_token.start - sent.start],
                    'end': offset_map_inv[kiwi_token.start + kiwi_token.len - sent.start],
                }
                for kiwi_token in kiwi_tokens
            ]
        except ValueError:
            # ignore sentences with unsupported POS morphs
            continue

        morph_ends = set()
        for morph in morphs:
            morph_ends.add(morph['end'])
        
        og_map = {}
        text_norm = ''

        for text_idx, text_char in enumerate(text):
            og_map[len(text_norm)] = text_idx
            text_norm += normalize_func(text_char)
        og_map[len(text_norm)] = len(text)
        
        last_idx = 0
        for i in range(len(text_norm) + 1):
            if i in og_map:
                last_idx = og_map[i]
            else:
                og_map[i] = last_idx
        
        tokenized_result = tokenizer(text_norm, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_result['input_ids'])

        if max_tokens > 0 and len(tokens) > max_tokens:
            # print(f'Skipping long sentence with {len(tokens)} tokens: {text[:50]}...')
            continue
        
        labels = []

        for id, (start, end), token in zip(tokenized_result['input_ids'], tokenized_result['offset_mapping'], tokens):
            og_start = og_map[start]
            og_end = og_map[end]

            is_empty_token = token in tokenizer.all_special_tokens or start == end

            if is_empty_token:
                labels.append('O')
                continue
            
            label = 'N'
            if label == 'N':
                for i in range(og_start, og_end):
                    if i in spacing_removed:
                        label = 'SM'
                        break
            if label == 'N':
                for i in range(og_start - 1, og_end):
                    if i in spacing_added:
                        label = 'SE'
                        break
            labels.append(label)

        if labels.count('O') - 2 > len(labels) * 0.1:
            # print(f'Skipping sentence with too many O labels: {text[:50]}...')
            continue

        entries.append({
            'input_ids': [id for id in tokenized_result['input_ids']],
            'attention_mask': [1] * len(tokenized_result['input_ids']),
            'labels': [label2id[label] for label in labels],
        })

    return entries

def process_lines(tokenizer, normalize_func, lines, max_tokens):
    multi_sents_txt = [
        unicodedata.normalize('NFC', line)
        for line in lines
        if line.strip()
    ]
    kiwi_results = kiwi.split_into_sents(multi_sents_txt, return_tokens=True)

    entries = []
    for kiwi_result in kiwi_results:
        entries.extend(
            process_sents(tokenizer, normalize_func, kiwi_result, max_tokens)
        )
    return entries

def chunked(iterable, n):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def data(
    model,
    cache,
    input,
    normalize_mode,
    output,
    split,
    **_kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache)
    config = AutoConfig.from_pretrained(model)
    max_token_length = config.max_position_embeddings

    txt_files = []
    if os.path.isfile(input):
        txt_files.append(input)
    elif os.path.isdir(input):
        for root, _, files in os.walk(input):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
    else:
        raise ValueError(f'Invalid input: {input}')
    txt_files.sort()
    txt_files_iter = tqdm(txt_files) if len(txt_files) > 1 else txt_files

    data = []

    if normalize_mode is None:
        normalize_func = lambda x: x
    else:
        normalize_func = lambda x: unicodedata.normalize(normalize_mode, x)

    for txt_file in txt_files_iter:
        total = 0
        with open(txt_file, 'r', encoding='utf-8') as f:
            for _line in f:
                total += 1
        with open(txt_file, 'r', encoding='utf-8') as f:
            for lines in chunked(tqdm(f, leave=False, desc=txt_file, total=total), 500):
                data.extend(
                    process_lines(tokenizer, normalize_func, lines, max_token_length)
                )

    print('Shuffling data...')
    random.shuffle(data)

    print('Splitting data...')
    train_size = int(len(data) * split)
    train_data = data[:train_size]
    validation_data = data[train_size:]

    data = {
        'train': train_data,
        'validation': validation_data,
    }

    if output is None:
        return data

    print(f'Writing data to {output}...',)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(data, f)

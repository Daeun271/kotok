from transformers import pipeline, AutoTokenizer
import dataclasses
import unicodedata
import os
import numpy as np
import logging
from .lemmatize import Lemmatizer

@dataclasses.dataclass
class UserDictEntry:
    morph: str
    suffix_wildcard: bool
    pos: str
    pos_match: set[str] | None

@dataclasses.dataclass
class Token:
    surface: str
    lemma: str
    tag: str
    start: int
    end: int

    def __repr__(self):
        if self.surface == self.lemma:
            return f'{self.surface}/{self.tag}'
        return f'{self.surface}/{self.lemma}/{self.tag}'
    
    def __str__(self):
        return repr(self)


def normalize_with_map(text, normalize_mode):
    convert_map = {}
    text_norm = ''

    for text_idx, text_char in enumerate(text):
        convert_map[len(text_norm)] = text_idx
        text_norm += unicodedata.normalize(normalize_mode, text_char)
    convert_map[len(text_norm)] = len(text)
    
    last_idx = 0
    for i in range(len(text_norm) + 1):
        if i in convert_map:
            last_idx = convert_map[i]
        else:
            convert_map[i] = last_idx

    return text_norm, convert_map


def apply_splits_single(token: Token):
    r = []

    space_idx = token.surface.find(' ')
    if space_idx == -1:
        return [token]
    
    space_idx_end = space_idx
    while token.surface[space_idx_end].isspace():
        space_idx_end += 1
    
    logging.debug(f'Splitting: "{token.surface}" -> "{token.surface[:space_idx]}" / "{token.surface[space_idx_end:]}"')

    return [
        Token(
            surface = token.surface[:space_idx],
            lemma = token.surface[:space_idx],
            tag = token.tag,
            start = token.start,
            end = token.start + space_idx,
        ),
    ] + apply_splits_single(
        Token(
            surface = token.surface[space_idx_end:],
            lemma = token.surface[space_idx_end:],
            tag = token.tag,
            start = token.start + space_idx_end,
            end = token.end,
        )
    )

def apply_splits(tokens: list[Token]):
    r = []
    for token in tokens:
        r += apply_splits_single(token)
    return r


def apply_lemmatization(tokens: list[Token], lemmatizer: Lemmatizer):
    r = []
    for token in tokens:
        if not token.tag.startswith('V'):
            r.append(token)
            continue
        lemma, extra_tokens = lemmatizer.lemmatize(token.surface)
        token.lemma = lemma
        token.surface = token.surface[:len(lemma)]
        token.end = token.start + len(lemma)
        for extra_token in extra_tokens:
            extra_token.start += token.start
            extra_token.end += token.start
        r.append(token)
        r += extra_tokens
    return r


def analyze(
    classification_pipeline,
    text,
    normalize_mode=None,
    lemmatizer=None
):
    text_norm, convert_map = normalize_with_map(text, normalize_mode or 'NFC')

    raw_tokens = classification_pipeline(text_norm)

    i = 0
    tokens = []

    while i < len(raw_tokens):
        raw_token = raw_tokens[i]
        entity = raw_token['entity']

        end_i = i

        pos = f'<UNK-{entity}>'
        if entity.startswith('B-'):
            pos = entity[2:]

            for j in range(i + 1, len(raw_tokens)):
                if raw_tokens[j]['entity'] == f'I-{pos}':
                    end_i = j
                else:
                    break
        
        start = raw_token['start']
        end = raw_tokens[end_i]['end']

        og_start = convert_map[start]
        og_end = convert_map[end]

        surface = text[og_start:og_end]

        token = Token(
            surface = surface,
            lemma = surface,
            tag = pos,
            start = og_start,
            end = og_end,
        )
        tokens.append(token)

        i = end_i + 1

    tokens = apply_splits(tokens)
    if lemmatizer:
        tokens = apply_lemmatization(tokens, lemmatizer)
    
    return tokens


# TODO: Support normalize mode, then merge with analyze
def analyze_with_user_dict(
    classification_pipeline,
    text,
    normalize_mode=None,
    user_dict=[],
    lemmatizer=None,
    _ignore_user_dict_entries=[]
):

    mask_token = classification_pipeline.tokenizer.mask_token

    tokens_pre_masked = classification_pipeline(text)
    
    logging.debug('pre-masked')
    for token in tokens_pre_masked:
        logging.debug(token)

    pre_mask_char = '\x1A'
    pre_masks = []
    text_pre_masked = text

    # mask text that appears in the user dictionary
    for entry_idx, entry in enumerate(user_dict):
        search_idx = 0
        while True:
            idx = text_pre_masked.find(entry.morph, search_idx)
            if idx == -1:
                break

            pre_mask_start = idx
            pre_mask_end = idx + len(entry.morph)

            if entry.suffix_wildcard:
                for token in tokens_pre_masked:
                    if token['start'] <= pre_mask_start and pre_mask_end <= token['end']:
                        pre_mask_end = max(token['end'], pre_mask_end)

            '''
            # These are now handled post-masking, pos tags for user dict entries are not predicted properly in many cases
            if entry.pos_match:
                found_pos = False
                for token in tokens_pre_masked:
                    pos = token['entity'].split('-')[-1]
                    if pre_mask_start <= token['start'] and token['end'] <= pre_mask_end and pos in entry.pos:
                        found_pos = True
                        break
                if not found_pos:
                    search_idx = pre_mask_end
                    continue
            '''

            replace_len = pre_mask_end - pre_mask_start

            search_idx = pre_mask_end + 1

            new_entry = (pre_mask_start, pre_mask_end, entry_idx)
            if new_entry in _ignore_user_dict_entries:
                continue

            text_pre_masked = text_pre_masked[:pre_mask_start] + pre_mask_char * replace_len + text_pre_masked[pre_mask_end:]
            pre_masks.append(new_entry)

    pre_masks_ftb = sorted(pre_masks, key=lambda x: x[0])
    pre_masks_btf = sorted(pre_masks, key=lambda x: x[0], reverse=True)

    text_masked = text_pre_masked
    for pre_mask_start, pre_mask_end, entry_idx in pre_masks_btf:
        text_masked = text_masked[:pre_mask_start] + mask_token + text_masked[pre_mask_end:]

    tokens = classification_pipeline(text_masked)
    logging.debug('masked')
    for token in tokens:
        logging.debug(token)

    mask_idx = 0
    offset = 0

    ignore_user_dict_entries = []

    # Check masked tokens
    # If the predicted pos does not match the user dict entry, ignore the entry
    for token in tokens:
        if token['word'] == mask_token:
            entry = pre_masks_ftb[mask_idx]
            user_entry = user_dict[entry[2]]

            if user_entry.pos_match:
                predicted_pos = token['entity'].split('-')[-1]
                if not predicted_pos in user_entry.pos_match:
                    ignore_user_dict_entries.append(entry)
                    mask_idx += 1
                    continue

            offset = entry[1] - token['end']
            
            token['score'] = np.float32(1.0)
            token['word'] = user_entry.morph
            token['tag'] = f'B-{user_entry.pos}'
            token['start'] = entry[0]
            token['end'] = entry[1]

            mask_idx += 1
            continue

        token['start'] = token['start'] + offset
        token['end'] = token['end'] + offset

    assert mask_idx == len(pre_masks_ftb)

    if ignore_user_dict_entries:
        # retry, with the ignored entries
        return analyze_with_user_dict(classification_pipeline, text, normalize_mode, user_dict, ignore_user_dict_entries)

    logging.debug('fixed')
    for token in tokens:
        logging.debug(token)

    i = 0
    tokens_out = []

    # Combine B- and I- tokens
    while i < len(tokens):
        raw_token = tokens[i]
        entity = raw_token['entity']

        end_i = i

        pos = f'<UNK-{entity}>'
        if entity.startswith('B-'):
            pos = entity[2:]

            for j in range(i + 1, len(tokens)):
                if tokens[j]['entity'] == f'I-{pos}':
                    end_i = j
                else:
                    break
        
        start = raw_token['start']
        end = tokens[end_i]['end']
        surface = text[start:end]

        token_out = Token(
            surface = surface,
            lemma = surface,
            tag = pos,
            start = start,
            end = end,
        )
        tokens_out.append(token_out)

        i = end_i + 1
    
    if not _ignore_user_dict_entries:
        # prevent morphemes across spaces
        tokens_out = apply_splits(tokens_out)

        # lemmatize verbs
        if lemmatizer:
            tokens_out = apply_lemmatization(tokens_out, lemmatizer)

    return tokens_out


def create_pipeline(
    model,
    classification_model,
    cache,
):
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache)

    return pipeline(
        'token-classification',
        model=classification_model,
        tokenizer=tokenizer,
        ignore_labels=[],
    )


def inference(
    model,
    classification_model,
    cache,
    format,
    normalize_mode = None,
    user_dict = None,
    lemma_data = None,
    no_lemma = False,
    error_model = None,
    error_classification_model = None,
    spacing_model = None,
    spacing_classification_model = None,
    **kwargs,
):
    if isinstance(user_dict, str):
        user_dict = load_user_dict(user_dict)
    elif user_dict is None:
        user_dict = []
    elif not isinstance(user_dict, list):
        raise ValueError(f'Invalid user dictionary: {user_dict}')
    
    lemmatizer = None
    if not no_lemma or lemma_data:
        lemmatizer = Lemmatizer(lemma_data)

    spacing_corrector = None
    if spacing_model and spacing_classification_model:
        from .spacing.inference import correct as correct_spacing, create_pipeline as create_spacing_pipeline
        spacing_pipeline = create_spacing_pipeline(
            spacing_model,
            spacing_classification_model,
            cache,
        )
        spacing_corrector = lambda text: correct_spacing(spacing_pipeline, text)

    error_corretor = None
    if error_model and error_classification_model:
        from .error.inference import create_error_corrector
        error_corretor = create_error_corrector(error_model, error_classification_model, cache)

    classification_pipeline = create_pipeline(
        model,
        classification_model,
        cache,
    )

    if user_dict:
        analyze_func = lambda text: analyze_with_user_dict(classification_pipeline, text, normalize_mode, user_dict, lemmatizer)
    else:
        analyze_func = lambda text: analyze(classification_pipeline, text, normalize_mode, lemmatizer)

    while True:
        try:
            text = input('> ')

            if spacing_corrector:
                text = spacing_corrector(text)

            if error_corretor:
                text, _corrections = error_corretor(text)

            if normalize_mode:
                text = unicodedata.normalize(normalize_mode, text)
            if format == 'pretty':
                tokens = analyze_func(text)
                print(' '.join(map(str, tokens)))
            elif format == 'raw':
                tokens_raw = classification_pipeline(text)
                for token in tokens_raw:
                    print(token)
        except KeyboardInterrupt:
            break
        except EOFError:
            break


class Analyzer:
    def __init__(
        self,
        model: str,
        classification_model: str,
        cache: str | None = None,
        normalize_mode: list[UserDictEntry] | str | None = None,
        user_dict: str | None = None,
        lemma_data: str | None = None,
        no_lemma: bool = False,
        error_model: str | None = None,
        error_classification_model: str | None = None,
        spacing_model: str | None = None,
        spacing_classification_model: str | None = None,
        **kwargs,
    ):
        """
        model: str -- The tokenizer model to use, either a name on Hugging Face or a path to a local model
        classification_model: str -- The classification model to use, generated from the train command
        cache: str | None -- The cache directory to use for the tokenizer
        normalize_mode: str | None -- The unicode normalization mode to use for the input text
        user_dict: list[UserDictEntry] | str | None -- The user dictionary to use for the analyzer, either a list of UserDictEntry objects, a path to a file, or a path to a directory
        lemma_data: str | None -- Path to the lemmatization data directory
        no_lemma: bool -- Whether to force disable lemmatization
        error_model: str | None -- The tokenizer model to use for the error corrector, either a name on Hugging Face or a path to a local model
        error_classification_model: str | None -- The classification model to use for the error corrector, generated from the train command
        spacing_model: str | None -- The tokenizer model to use for the spacing corrector, either a name on Hugging Face or a path to a local model
        spacing_classification_model: str | None -- The classification model to use for the spacing corrector, generated from the train command
        """

        self.normalize_mode = normalize_mode

        if isinstance(user_dict, str):
            self.user_dict = load_user_dict(user_dict)
        elif user_dict is None:
            self.user_dict = []
        elif not isinstance(user_dict, list):
            raise ValueError(f'Invalid user dictionary: {user_dict}')
        
        self.lemmatizer = None
        if not no_lemma or lemma_data:
            self.lemmatizer = Lemmatizer(lemma_data)

        self.spacing_corrector = None
        if spacing_model and spacing_classification_model:
            from .spacing.inference import correct as correct_spacing, create_pipeline as create_spacing_pipeline
            spacing_pipeline = create_spacing_pipeline(
                spacing_model,
                spacing_classification_model,
                cache,
            )
            self.spacing_corrector = lambda text: correct_spacing(spacing_pipeline, text)

        self.error_corretor = None
        if error_model and error_classification_model:
            from .error.inference import create_error_corrector
            self.error_corretor = create_error_corrector(error_model, error_classification_model, cache)

        self.classification_pipeline = create_pipeline(
            model,
            classification_model,
            cache,
        )

        if self.user_dict:
            self.analyze_func = lambda text: analyze_with_user_dict(self.classification_pipeline, text, self.normalize_mode, self.user_dict, self.lemmatizer)
        else:
            self.analyze_func = lambda text: analyze(self.classification_pipeline, text, self.normalize_mode, self.lemmatizer)

    def run(self, text: str, format='pretty') -> list[Token]:
        """
        text: str -- The input text to analyze
        format: str -- The output format, either 'pretty' or 'raw'. 'pretty' will return a list of Token objects, 'raw' will return the raw output from the model.
        """

        if self.spacing_corrector:
            text = self.spacing_corrector(text)

        if self.error_corretor:
            text, _corrections = self.error_corretor(text)

        if self.normalize_mode:
            text = unicodedata.normalize(self.normalize_mode, text)

        if format == 'raw':
            return self.classification_pipeline(text)

        return self.analyze_func(text)        


#
# User dictionary loading
#

def load_user_dict_file(file_path: str, _no_sort: bool=False):
    '''
    file_path: str -- 
    '''

    user_dict = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_idx, line in enumerate(file):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            line = line.replace('\t', ' ')
            space_idx = line.rfind(' ')
            if space_idx == -1:
                raise ValueError(f'Invalid user dictionary entry at {file_path}:{line_idx + 1}')
            
            suffix_wildcard = False
            pos_match = None
            morph = line[:space_idx].strip()
            pos_raw = line[space_idx + 1:].strip()

            if morph.endswith('*'):
                morph = morph[:-1].strip()
                suffix_wildcard = True

            exclamation_idx = pos_raw.rfind('!')
            if exclamation_idx != -1:
                pos = pos_raw[:exclamation_idx].strip()
                pos_parts = pos_raw[exclamation_idx + 1:].split(',')
                if pos_parts:
                    pos_match = set(pos_parts)
                else:
                    pos_match = set([pos])
            else:
                pos = pos_raw

            user_dict.append(UserDictEntry(
                morph = morph,
                suffix_wildcard = suffix_wildcard,
                pos = pos,
                pos_match = pos_match,
            ))

    if not _no_sort:
        user_dict.sort(key=lambda x: len(x.morph), reverse=True)

    return user_dict

def load_user_dict_dir(dir_path: str, _no_sort: bool=True) -> list[UserDictEntry]:
    user_dict = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if os.path.splitext(file)[1] in ('.tsv'):
                file_path = os.path.join(root, file)
                user_dict.extend(load_user_dict_file(file_path, _no_sort=True))
    if not _no_sort:
        user_dict = dict(sorted(user_dict.items(), key=lambda x: len(x[0]), reverse=True))
    return user_dict

def load_user_dict(path: str) -> list[UserDictEntry]:
    if os.path.isfile(path):
        return load_user_dict_file(path)
    elif os.path.isdir(path):
        return load_user_dict_dir(path)
    else:
        raise ValueError(f'Invalid user dictionary: {path}')

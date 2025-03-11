import os
import math
import json
import logging
import hangul_jamo

VOVEL_SPLIT = {
    'ㅘ': 'ㅗㅏ',
    'ㅙ': 'ㅗㅐ',
    'ㅚ': 'ㅗㅣ',
    'ㅝ': 'ㅜㅓ',
    'ㅞ': 'ㅜㅔ',
    'ㅟ': 'ㅜㅣ',
    'ㅢ': 'ㅡㅣ',
}

def decompose(text):
    """
    Regular text to jamo, also split vovels.
    """
    r = hangul_jamo.decompose(text)
    for k, v in VOVEL_SPLIT.items():
        r = r.replace(k, v)
    return r

def compose(text):
    """
    Jamo to regular text, also merge vovels.
    """
    r = hangul_jamo.compose(text)
    for k, v in VOVEL_SPLIT.items():
        r = r.replace(v, k)
    return r

def trace_to_tokens(surface_text, trace):
    """
    Emit tokens from rule from a lemmatisation trace.
    """
    from .inference import Token

    tokens = []

    ta_ending = surface_text.endswith('다')

    offset_map = {}
    accu = 0
    for i, c in enumerate(surface_text):
        dc = decompose(c)
        for ii in range(len(dc)):
            offset_map[accu + ii] = i
        accu += len(dc)
    
    for i, (name, suffix_in, suffix_out, jamo_end_distance) in enumerate(trace):
        is_last = i == len(trace) - 1

        jamo_check_pos = jamo_end_distance - len(suffix_out)
        start_i = len(surface_text) - offset_map[jamo_check_pos] if jamo_check_pos in offset_map else 0
        start_i = max(0, start_i)
        end_i = start_i + math.ceil(len(suffix_in) / 2.4)
        end_i = min(len(surface_text), end_i)
        surface = surface_text[start_i:end_i]

        if name == '-아/어':
            lemma = '아'
            if any(c in suffix_in for c in 'ㅓㅕㅖ'):
                lemma = '어'
            token = Token(surface, lemma, 'EC', start_i, end_i)
            tokens.append(token)
        elif name == '-았/었':
            lemma = '았'
            if any(c in suffix_in for c in 'ㅓㅕㅖ'):
                lemma = '었'
            token = Token(surface, lemma, 'EC', start_i, end_i)
            tokens.append(token)
        elif name == '-[느]ㄴ다':
            lemma = 'ㄴ'
            if suffix_in.startswith('ㄴㅡㄴ'):
                lemma = '는'
            if ta_ending and is_last:
                lemma += '다'
            token = Token(surface, lemma, 'ETM', start_i, end_i)
            tokens.append(token)

    return tokens


class Lemmatizer:
    def __init__(self, data_dir):
        lemmas_path = os.path.join(data_dir, 'lemmas.txt')
        transforms_path = os.path.join(data_dir, 'transforms.json')

        self.lemmas = {}
        with open(lemmas_path, 'r', encoding='utf-8') as f:
            for line in f:
                lemma, freq = line.strip().split()
                self.lemmas[lemma] = int(freq)

        with open(transforms_path, 'r', encoding='utf-8') as f:
            transforms = json.load(f)

        self.rules = []
        for name, transform_rules in transforms.items():
            for transform_rule in transform_rules:
                self.rules.append((name, transform_rule))

    def transform_jamo(self, source_text, max_depth=5):
        """
        Returns a list of all possible lemmatizations of the given text in jamo form.
        """
        results = [(source_text, set(), [], 0)]

        i = 0
        while i < len(results):
            (text, conditions, trace, depth) = results[i]
            if depth >= max_depth:
                i += 1
                continue
            
            # attempt to apply all rules
            for name, (suffix_in, suffix_out, cond_in, cond_out) in self.rules:
                if cond_in and not any(c in conditions for c in cond_in):
                    continue
                if not text.endswith(suffix_in):
                    continue
                
                # exchange the suffix
                new_text = text[:-len(suffix_in)] + suffix_out
                new_conditions = cond_out
                
                old_jamo_end_distance = 0 if not trace else trace[-1][3]
                jamo_end_distance = len(suffix_in) - len(suffix_out) + old_jamo_end_distance
                
                new_trace = trace + [(name, suffix_in, suffix_out, jamo_end_distance)]

                # add new transformation (will be processed later in the loop again)
                results.append((new_text, new_conditions, new_trace, depth + 1))

            i += 1

        return [
            (text, conditions, trace)
            for text, conditions, trace, *_ in results
        ]
    
    def transform(self, source_text):
        """
        Returns a list of all possible lemmatizations of the given text.
        """
        text_jamo = decompose(source_text)
        transformed_jamo = self.transform_jamo(text_jamo)
        transformed = []
        for text_jamo, conditions, trace in transformed_jamo:
            try:
                text = compose(text_jamo)
            except ValueError:
                continue
            transformed.append((text, conditions, trace))
        return transformed

    def lemmatize(self, surface_text):
        """
        Find the best lemmatization of the given text.
        """
        transforms = self.transform(surface_text)
        lemmas = []

        for text, _conditions, trace in transforms:
            # Only consider lemmas that end with verbic dictionary form
            if text.endswith('다'):
                lemma = text[:-1]
                # Only consider lemmas that are in the dictionary
                if not lemma in self.lemmas:
                    continue
                lemmas.append((lemma, trace))
        
        # Fewer transformations are better
        lemmas.sort(key=lambda x: len(x[1]))

        possible_lemmas = []
        for i, (lemma, transforms) in enumerate(lemmas):
            if not lemma in possible_lemmas:
                possible_lemmas.append(lemma)
            logging.debug(f'Lemmatization {i+1}: {lemma} {transforms}')
        logging.debug(f'Possible lemmas: {", ".join(possible_lemmas)}')

        if lemmas:
            lemma, trace = lemmas[0]
            return lemma, trace_to_tokens(surface_text, trace)

        return text, []


def lemmatize(
    data_dir: str,
    **_kwargs,
):
    lemmatizer = Lemmatizer(data_dir)

    while True:
        try:
            text = input('> ')
            print(lemmatizer.lemmatize(text))
        except KeyboardInterrupt:
            break
        except EOFError:
            break

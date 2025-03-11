import os
import logging
from transformers import pipeline, AutoTokenizer
# from symspellpy_ko import KoSymSpell, Verbosity
from .typo import TypoCorrector

# sym_spell = KoSymSpell()
# sym_spell.load_korean_dictionary(decompose_korean=True, load_bigrams=False)

typo_corrector = TypoCorrector(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'data',
        'correction',
    )
)


HANGUL_RANGES = [
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF)   # Hangul Jamo Extended-B
]

def is_hangul(char):
    return any(start <= ord(char) <= end for start, end in HANGUL_RANGES)

def is_eojeol_char(char):
    return is_hangul(char)


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

def should_correct_token(token, error_min_score=0.4, no_error_max_score=0.5):
    tag = token['entity']
    if tag == 'O':
        return False
    score = token['score']
    if tag in ('B-ME', 'I-ME'):
        return score > error_min_score
    return score < no_error_max_score

def avg_score_in_span(classification_pipeline, text, start_idx, end_idx):
    scores = []
    tokens = classification_pipeline(text)
    for token in tokens:
        token_start = token['start']
        token_end = token['end']

        # check for overlap
        if token_start < end_idx and token_end > start_idx:
            tag = token['entity']
            score = token['score']
            if tag == 'O':
                score = 0.0
            elif tag in ('B-ME', 'I-ME'):
                score = 0.5 - score  # 0.5 pentaly for ME tags
            scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)

def correct(
    classification_pipeline,
    text,
    correction_min_score=0.7,
    text_start_idx=0,
):
    tokens = classification_pipeline(text)

    logging.debug(f'Tokens:')
    for token in tokens:
        logging.debug(f'  {token}')

    applied_corrections = []

    # find token span to correct
    i = 0
    while i < len(tokens):
        i_token = tokens[i]

        if i_token['start'] < text_start_idx:
            # skip tokens that do not need further correction
            i += 1
            continue

        if not should_correct_token(i_token):
            i += 1
            continue

        i_start = i
        i_end = i

        # prepend tokens that are candidates for correction
        while i_start > 0:
            i_start_token = tokens[i_start]
            last_text_idx = i_start_token['start'] - 1
            if last_text_idx < 0:
                # never expand beyond the start of the text
                break
            last_text_char = text[last_text_idx]
            if not is_eojeol_char(last_text_char):
                # never expand beyond an eojeol
                break
            i_start -= 1
            if not should_correct_token(i_start_token):
                break
        
        # append tokens that are candidates for correction
        while i_end < len(tokens) - 1:
            i_end_token = tokens[i_end]
            next_text_idx = i_end_token['end']
            if next_text_idx >= len(text):
                # never expand beyond the end of the text
                break
            next_text_char = text[next_text_idx]
            if not is_eojeol_char(next_text_char):
                # never expand beyond an eojeol
                break
            i_end += 1
            if not should_correct_token(i_end_token):
                break

        # correct the span
        i_start_idx = tokens[i_start]['start']
        i_end_idx = tokens[i_end]['end']

        span = text[i_start_idx:i_end_idx]

        logging.debug(f'Checking span: {span}')

        # corrections = sym_spell.lookup(span, Verbosity.ALL, max_edit_distance=2)
        corrections = typo_corrector.correct(span, max_depth=4, max_cost=5)

        best_correction = None

        for i_correction, correction in enumerate(corrections):
            corrected_span = correction[0]
            corrected_span_start_idx = i_start_idx
            corrected_span_end_idx = corrected_span_start_idx + len(corrected_span)
            corrected_text = text[:i_start_idx] + corrected_span + text[i_end_idx:]
            corrected_score = avg_score_in_span(classification_pipeline, corrected_text, corrected_span_start_idx, corrected_span_end_idx)
            adj_corrected_score = corrected_score - (correction[1] * 0.025)

            logging.debug(f'Correction {i_correction+1}: {corrected_span} ({corrected_score:.05f}, {adj_corrected_score:.05f})')

            if not best_correction or adj_corrected_score > best_correction['score']:
                best_correction = {
                    'text': corrected_text,
                    'corrected_span': corrected_span,
                    'score': adj_corrected_score,
                    'corrected_span_end_idx': corrected_span_end_idx,
                }

        if best_correction and best_correction['score'] > correction_min_score:
            text = best_correction['text']
            applied_corrections.append({
                'span': span,
                'corrected_span': best_correction['corrected_span'],
                'score': best_correction['score'],
            })
            
            sub_text, sub_applied_corrections = correct(
                classification_pipeline,
                text,
                correction_min_score=correction_min_score,
                text_start_idx=best_correction['corrected_span_end_idx'],
            )
            text = sub_text

            return text, applied_corrections + sub_applied_corrections

        i += 1

    return text, applied_corrections


def create_error_corrector(
    model,
    classification_model,
    cache = None,
):
    classification_pipeline = create_pipeline(
        model,
        classification_model,
        cache,
    )

    def error_corrector(text):
        return correct(classification_pipeline, text)

    return error_corrector


def inference(
    model,
    classification_model,
    cache,
    **kwargs,
):
    classification_pipeline = create_pipeline(
        model,
        classification_model,
        cache,
    )

    while True:
        try:
            text = input('> ')
            if not text:
                continue
            print(correct(classification_pipeline, text))
        except KeyboardInterrupt:
            break
        except EOFError:
            break
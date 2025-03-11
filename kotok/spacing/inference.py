import logging
from transformers import pipeline, AutoTokenizer


def should_correct_token(token, min_error_score=0.4):
    entity = token['entity']
    if entity in ('SM', 'SE'):
        return token['score'] > min_error_score
    return False

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
            elif tag in ('SE', 'SM'):
                score = 0.5 - score  # 0.5 pentaly for ME tags
            scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)

def correct(
    classification_pipeline,
    text,
    text_start_idx=0,
):
    tokens = classification_pipeline(text)

    logging.debug(f'Tokens:')
    for token in tokens:
        logging.debug(f'  {token}')

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

        entity = i_token['entity']
        i_start = i_token['start']
        i_end = i_token['end']

        if entity == 'SM':
            logging.debug(f'Correcting SM: {text[i_start:i_end]}')

            best_correction = None
            # try to insert space in all possible positions
            for j in range(i_start, i_end + 1):
                text_with_space = text[:j] + ' ' + text[j:]
                score = avg_score_in_span(classification_pipeline, text_with_space, i_start, i_end + 1)

                logging.debug(f'Correction: {text_with_space} ({score})')
                
                if not best_correction or score > best_correction['score']:
                    best_correction = {
                        'text': text_with_space,
                        'score': score,
                        'next_start_idx': j + 1,
                    }
            if best_correction and best_correction['score'] > 0.7:
                text = best_correction['text']
                return correct(classification_pipeline, text, text_start_idx=best_correction['next_start_idx'])

        if entity == 'SE':
            # TODO
            pass

        i += 1

    return text


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
    format = 'pretty',
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
            if format == 'raw':
                for token in classification_pipeline(text):
                    print(token)
            else:
                print(correct(classification_pipeline, text))
        except KeyboardInterrupt:
            break
        except EOFError:
            break

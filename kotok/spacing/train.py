import json
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from . import labels

def train(args):
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = labels.num_labels
    config.id2label = labels.id2label
    config.label2id = labels.label2id

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, cache_dir=args.cache, config=config,
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        report_to="tensorboard",
        logging_dir=args.logs,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    with open(args.data, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print('Training...')
    trainer.train()

    print('Saving...')
    model.save_pretrained(args.output)

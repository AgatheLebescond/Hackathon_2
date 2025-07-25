import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model


def load_data(dataset_name="imdb", split="train", sample_size=10000):
    dataset = load_dataset(dataset_name, split=split)
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
    return dataset


def tokenize(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)


def main():
    model_name = os.environ.get("BASE_MODEL", "distilbert-base-uncased")
    dataset_name = os.environ.get("DATASET", "imdb")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = load_data(dataset_name, "train")
    val_ds = load_data(dataset_name, "test", sample_size=2000)

    train_ds = tokenize(train_ds, tokenizer)
    val_ds = tokenize(val_ds, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"], lora_dropout=0.1)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=True if os.environ.get("USE_FP16") else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.evaluate()
    model.save_pretrained("./lora_sentiment")
    tokenizer.save_pretrained("./lora_sentiment")


if __name__ == "__main__":
    main()

from datasets import load_dataset
import torch
import transformers
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import TrainingArguments, default_data_collator, Trainer

datasets = load_dataset("squad_v2")
# train_dataset, validation_dataset = datasets['train'], datasets['validation']
# print(train_dataset["question"][0], train_dataset["context"][0], train_dataset["answers"][0])

# Load the base model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

# print(torch.backends.mps.is_available())
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# The maximum length of input features after question and context are concatenated
max_length = 384
doc_stride = 128
pad_on_right = tokenizer.padding_side == "right"


def prepare_train_features(examples):
    # Need to do truncation and padding to the examples while retaining all information, so slicing is used
    # Examples of long text will be sliced into multiple inputs,
    # and there will be an intersection between two adjacent inputs
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Use overflow_to_sample_mapping and offset_mapping to map the original position before slicing,
    # which is used to find the start and end positions of the answer
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # Traverse all slices and label the answer positions using the position of CLS
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Distinguish between question and context
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Get the original position in the example
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If there is no answer, use the position of [CLS] as the answer position
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # The start and end positions of the answer at character-level
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find the start position at token-level
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # Find the end position at token-level
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Check if the answer exceeds the length of the text;
            # if so, also use the position of [CLS] as the answer position
            if not (offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # if not, find the start and end positions of the answer at token-level
                # NB: we could go after the last offset if the answer is the last word (edge case)
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


# Preprocess all samples in the datasets
tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)

# Define the training arguments
batch_size = 16
args = TrainingArguments(
    f"test-squad-1",
    evaluation_strategy="epoch",  # Validation evaluation will be performed once per epoch
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,  # 3
    weight_decay=0.01,
    log_level='error',
    logging_strategy="no",
    report_to="none"
)

# Data collector, used to input processed data into the model
data_collator = default_data_collator

# Define the trainer
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save model locally
save_dir = '/Users/erynnbai/PycharmProjects/DistillingModel/distilbert-base-uncased-answer-extraction'
# trainer.save_model(save_dir)
print("Saved model to:", save_dir)









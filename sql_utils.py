import torch
import faiss
import random
import logging
import numpy as np
from enum import Enum
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import transformers
import json

class JSONLinesHandler(logging.FileHandler):
    def emit(self, record):
        log_entry = self.format(record)
        with open(self.baseFilename, 'a') as file:
            file.write(f"{log_entry}\n")

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up jsonlines logger."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as file:
        pass  # create the file if it does not exist

    formatter = logging.Formatter('%(message)s')  # Only message gets logged
    handler = JSONLinesHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def parse_pred_text(pred_text: str, label_set: set[str]) -> str:
    """A simple heuristic parsing function for compatibility with the label_set."""
    pred_text = pred_text.strip(" ().:")
    if pred_text[0] in label_set:
        pred_text = pred_text[0]
    return pred_text

def text_in_label_set(text: str, label_set: set[str]) -> bool:
    text = text.lower().strip()
    fuzzy_label_set = {label.lower() for label in label_set}
    return text in fuzzy_label_set

def get_nlsql_system_prompt() -> str:
    system_prompt = """\
    Act as a professional programmer.
    You will be given a table schema and a user query, and you need to generate the correct SQL code to answer the user query in the following format:
    ```sql\n<your_SQL_code>\n```"""
    return strip_all_lines(system_prompt)

def get_nlsql_zeroshot_prompt(table_schema: str, user_query: str) -> str:
    prompt = f"""\
    {table_schema}
    
    -- Using valid SQLite, answer the following question for the tables provided above.
    -- Question: {user_query}
    \n\nNow, first identify the relevant tables and columns, \
    and then generate the correct SQL code directly in the following format:
    ```sql\n<your_SQL_code>\n```"""
    
    # Now, generate the correct SQL code directly in the following format:
    # ```sql\n<your_SQL_code>\n```"""
    return strip_all_lines(prompt)

def get_prompt(table_schema: str, user_query: str, tokenizer: transformers.PreTrainedTokenizer) -> str:
    messages = [
        {"role": "system", "content": get_nlsql_system_prompt()},
        {"role": "user", "content": get_nlsql_zeroshot_prompt(table_schema, user_query)}
    ]
    text_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text_chat

class RetrieveOrder(Enum):
    SIMILAR_AT_TOP = "similar_at_top"  # the most similar retrieved chunk is ordered at the top
    SIMILAR_AT_BOTTOM = "similar_at_bottom"  # reversed
    RANDOM = "random"  # randomly shuffle the retrieved chunks

class RAG:

    def __init__(self, rag_config: dict) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(rag_config["embedding_model"])
        self.embed_model = AutoModel.from_pretrained(rag_config["embedding_model"]).eval()
        
        self.indices = {}
        self.id2evidence = {}
        self.embed_dim = len(self.encode_data("Test embedding size"))
        self.insert_count = {}
        self.insert_acc = 0 # Total number of insertions across all tables

        self.main_index = self.create_faiss_index()
        self.main_id2evidence = {}

        self.seed = rag_config["seed"]
        self.top_k = rag_config["top_k"]
        orders = {member.value for member in RetrieveOrder}
        assert rag_config["order"] in orders
        self.retrieve_order = rag_config["order"]
        random.seed(self.seed)

        self.rag_filename = rag_config["rag_filename"]
        # Clear the file content
        with open(self.rag_filename, 'w') as file:
            pass

    def create_faiss_index(self):
        # Create a FAISS index
        return faiss.IndexFlatL2(self.embed_dim)

    def encode_data(self, sentence: str) -> np.ndarray:
        # Tokenize the sentence
        encoded_input = self.tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        feature = sentence_embeddings.numpy()[0]
        norm = np.linalg.norm(feature)
        return feature / norm

    def insert(self, table_schema: str, key: str, value: str) -> None:
        """Use the key text as the embedding for future retrieval of the value text."""
        if table_schema not in self.indices:
            self.indices[table_schema] = self.create_faiss_index()
            self.id2evidence[table_schema] = {}
            self.insert_count[table_schema] = 0

        embedding = self.encode_data(key).astype('float32')  # Ensure the data type is float32

        self.main_index.add(np.expand_dims(embedding, axis=0))
        self.main_id2evidence[str(self.insert_acc)] = value
        self.insert_acc += 1

        self.indices[table_schema].add(np.expand_dims(embedding, axis=0))
        self.id2evidence[table_schema][str(self.insert_count[table_schema])] = value
        self.insert_count[table_schema] += 1

        # Save the key-value pair to the file as input output (jsonl format)
        with open(self.rag_filename, 'a') as file:
            dic = {"table_schema": table_schema, "input": key, "output": value}
            line = json.dumps(dic)
            print(line, file=file)

    def retrieve(self, table_schema: str, query: str, top_k: int) -> list[str]:
        """Retrieve top-k text chunks"""
        """If table_schema has less than top_k chunks, use main_index to retrieve"""

        data_count = self.insert_count[table_schema] if table_schema in self.insert_count else 0
        embedding = self.encode_data(query).astype('float32')  # Ensure the data type is float32

        # If the table_schema has less than top_k chunks, use main_index to retrieve
        distances = []
        indices = []
        text_list = []
        if data_count < top_k:
            main_distances, main_indices = self.main_index.search(np.expand_dims(self.encode_data(query), axis=0), top_k - data_count)
            main_distances = main_distances[0].tolist()
            main_indices = main_indices[0].tolist()

            distances.extend(main_distances)
            indices.extend(main_indices)
            text_list.extend([self.main_id2evidence[str(idx)] for idx in indices])

        # Retrieve the top-k chunks from the table_schema
        if table_schema in self.indices:
            top_k = min(top_k, self.insert_count[table_schema])
            table_distances, table_indices = self.indices[table_schema].search(np.expand_dims(embedding, axis=0), top_k)
            table_distances = distances[0].tolist()
            table_indices = indices[0].tolist()

            distances.extend(table_distances)
            indices.extend(table_indices)
            text_list.extend([self.id2evidence[table_schema][str(idx)] for idx in table_indices])

        results = [{'link': str(idx), '_score': {'faiss': dist}} for dist, idx in zip(distances, indices)]
        # Re-order the sequence based on self.retrieve_order
        if self.retrieve_order == RetrieveOrder.SIMILAR_AT_BOTTOM.value:
            results = list(reversed(results))
        elif self.retrieve_order == RetrieveOrder.RANDOM.value:
            random.shuffle(results)
        
        return text_list

def extract_json_string(res: str) -> str:
    """Extract the first valid json string from the response string (of LLMs).
    
    Return '' (empty string) if not found. Raise ValueError if an } is found before any {.
    """
    start, end = -1, -1
    cnt = 0  # act as a representation of a stack of '{' '}' pairs
    for i in range(len(res)):
        ch = res[i]
        if ch == '{':
            if cnt == 0:  # the first time '{' is encountered
                start = i
            cnt += 1
        elif ch == '}':
            if cnt <= 0:
                raise ValueError("found } before any { appears")
            cnt -= 1
            if cnt == 0:  # found the end index
                end = i
                break
    return res[start:end+1]

def strip_all_lines(s: str) -> str:
    """Remove all leading and trailing spaces of each line in the string."""
    return '\n'.join([line.strip() for line in s.splitlines()])

if __name__ == "__main__":
# Initialize RAG with a configuration dictionary
    rag_config = {
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "rag_filename": "test_rag_pool",
        "seed": 42,
        "top_k": 16,
        "order": "similar_at_top"  # ["similar_at_top", "similar_at_bottom", "random"]
    }
    rag = RAG(rag_config)

    # Key-value pairs for testing
    key_value_pairs = [
        ("Apple is my favorite fruit", "Oh really?"),
        ("What is your favorite fruit?", "Lettuce, tomato, and spinach."),
        ("What is your favorite vegetable?", "Apple, banana, and watermelon."),
        ("What do you like to read in your free time?", "Sherlock Holmes")
    ]

    # Insert the key-value pairs into the RAG
    for key, value in key_value_pairs:
        rag.insert(key, key + ' ' + value)

    from pprint import pprint

    query = "I like to eat lettuce."
    results = rag.retrieve(query, top_k=rag_config["top_k"])
    pprint(results)

def merge_dicts(dicts: list[dict]) -> dict:
    d = dict()
    for dd in dicts:
        for k, v in dd.items():
            if (k in d) and (d[k] != v):
                print(k, d[k], v)
                raise ValueError("Found duplicated and inconsistent key-value pair.")
            d[k] = v
    return d
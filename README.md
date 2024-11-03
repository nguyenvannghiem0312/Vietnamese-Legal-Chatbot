# Vietnamese Legal Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot built with the Legal corpus dataset. It uses a Bi-Encoder model to encode text data into vectors and a large language model (LLM) to generate answers based on user queries.

## Requirements

- Python 3.9+
- Libraries listed in `requirements.txt`

## Installation Guide

1. Clone this repository to your local machine:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Guide

### Step 1: Configure the Embedding Model and LLM

First, modify a few parameters in two files to ensure the correct models are used.

- **Configure the Embedding Model**:
    - Open `embedding/embedder.py` and locate the embedding model declaration within the `__init__` method.
    - Ensure that the `model_name` line is set to `NghiemAbe/Vi-Legal-Bi-Encoder` or your model:
      ```python
      def __init__(self,
                    model_name: str = "NghiemAbe/Vi-Legal-Bi-Encoder",
                    # model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
                    ):
          self.embedder = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'token': 'hf_PU...'})
      ```

- **Configure the LLM**:
    - Open `model/gemma.py` and locate the `GemmaSettings` class.
    - Update the model path as shown below:
      ```python
      class GemmaSettings(Model):
          url = "https://huggingface.co/NghiemAbe/SeaLLM-7B-v2.5-AWQ"
          file_name = "SeaLLM-7B-v2.5-AWQ"
          clients = [LlmClientType.VLMM]
          type = "gemma"
      ```
      
### Step 1: Create the Vector Database

Secondly, you need to create a vector database from the Legal corpus dataset. Run the following command:

```bash
python dbvector_builder.py --docs-path "NghiemAbe/Legal-corpus-indexing" --file-type "huggingface" --chunk-size 1024 --chunk-overlap 256 --vector-store-path "vectordb/chroma" --vitokenizer 1
```

- `--docs-path`: Path to the folder containing the Legal corpus dataset.
- `--file-type`: Type of data, here set to "huggingface".
- `--chunk-size`: Size of each text chunk.
- `--chunk-overlap`: Overlap size between text chunks.
- `--vector-store-path`: Path where the vector database will be stored, here set to `vectordb/chroma`.
- `--vitokenizer`: Configuration to use a Vietnamese tokenizer.



### Step 3: Run the Application

Finally, start the application by running the following command:

```bash
streamlit run rag_chatbot_app.py -- --model gemma --k 2 --synthesis-strategy all_context --vector-store-path 'vectordb/chroma'
```

- `--model`: Specifies the LLM to use, here set to `gemma`.
- `--k`: Number of results from the vector database to use for each query.
- `--synthesis-strategy`: Synthesis strategy, here set to `all_context`.
- `--vector-store-path`: Path to the previously created vector database.

The application will open in a browser, and you can start entering queries to receive answers from the chatbot.

## Notes

- Ensure all configuration parameters are correct before starting the application.
- If there are issues loading models from Hugging Face, check your API token or model.

## Contact

For more information, please contact [Nguyễn Văn Nghiêm](mailto:nguyenvannghiem0312@gmail.com).

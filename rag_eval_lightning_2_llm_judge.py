# %% [markdown]
# ## Objective: To build a simple RAG evaluation framework
# 
# ### Part 2: Build & Benchmark a RAG System using an already synthesized evaluation dataset (in Part 1)
# 
# #### Part 1: Synthesize and filter an Instruction dataset from a custom knowledge-base (See https://lightning.ai/panchamsnotes/studios/evaluate-your-rag-part-1-synthesize-an-evaluation-dataset?view=public&section=featured)
# 
# 
# 
# #### Primary reference: https://huggingface.co/learn/cookbook/en/rag_evaluation by Aymeric Roucher (https://huggingface.co/m-ric)
# 
# For the knowledge base, let us use the  litgpt Github repo: https://github.com/Lightning-AI/litgpt/tree/main

# %% [markdown]
# ### Installs and Dependencies

# %%
%pip install -q torch transformers transformers langchain sentence-transformers tqdm openpyxl openai pandas datasets
%pip install -U --quiet langchain langsmith langchainhub langchain_benchmarks langchain-openai Gitpython python-dotenv RAGatouille
%pip install --quiet chromadb openai huggingface pandas langchain_experimental sentence_transformers pyarrow anthropic tiktoken

# %%
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai_api_key = os.environ['OPENAI_API_KEY'] 
hf_api_key = os.environ['HF_API_KEY'] 

# %%
import textwrap
from tqdm import tqdm
import pandas as pd
import json
import datasets
import random
import glob
from typing import Optional, List, Tuple

pd.set_option("display.max_colwidth", None)

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GitLoader
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.llms import HuggingFaceHub
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM
from langchain_core.language_models import BaseChatModel

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

from transformers import AutoTokenizer, AutoModelForCausalLM

from ragatouille import RAGPretrainedModel

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Build the RAG System

# %% [markdown]
# ### Pre-processing documents to build the knowledge base

# %%
loader = GitLoader(
    clone_url="https://github.com/Lightning-AI/litgpt",
    repo_path="./litgpt_data_github/",
    branch="main",
    file_filter=lambda file_path: file_path.endswith(".md") # Only get the markdown files
)

data = loader.load()

RAW_KNOWLEDGE_BASE = data

# %%
def split_documents_into_chunks(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: str,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of size `chunk_size` characters and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", "", "\n\n\n"],
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

# %% [markdown]
# ### Create the retriever after building a vector index using FAISS

# %%
def create_vector_index(
    langchain_docs: List[LangchainDocument],
    chunk_size: int,
    embedding_model_name: Optional[str] = "thenlper/gte-small",
) -> FAISS:
    """
    Creates a FAISS index from the given embedding model and documents. Loads the index directly if it already exists.

    Args:
        langchain_docs: list of documents
        chunk_size: size of the chunks to split the documents into
        embedding_model_name: name of the embedding model to use

    Returns:
        FAISS index
    """
    # load embedding_model
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # set True to compute cosine similarity
    )

    # Check if embeddings already exist on disk
    index_name = (
        f"index_chunk:{chunk_size}_embeddings:{embedding_model_name.replace('/', '~')}"
    )
    index_folder_path = f"./data/indexes/{index_name}/"
    if os.path.isdir(index_folder_path):
        return FAISS.load_local(
            index_folder_path,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True
        )

    else:
        print("Index not found, generating it...")
        docs_processed = split_documents_into_chunks(
            chunk_size,
            langchain_docs,
            embedding_model_name,
        )
        knowledge_index = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        knowledge_index.save_local(index_folder_path)
        return knowledge_index

# %% [markdown]
# ### LLM Reader retrieves relevant documents to formulate response

# %%
RAG_PROMPT_TEMPLATE = """
<|system|>
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.</s>
<|user|>
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
</s>
<|assistant|>
"""

# %%

repo_id = "HuggingFaceH4/zephyr-7b-beta" 
READER_MODEL_NAME = "zephyr-7b-beta"

READER_LLM = HuggingFaceHub(
    repo_id=repo_id,
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
    huggingfacehub_api_token=hf_api_key
)

# %%
def get_rag_response(
    question: str,
    llm: LLM,
    knowledge_index: VectorStore,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 7,
) -> Tuple[str, List[LangchainDocument]]:
    """Answer a question using RAG with the given knowledge index."""
    # Gather documents with retriever
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    # Optionally rerank results
    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
    )

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    answer = llm(final_prompt)

    return answer, relevant_docs

# %% [markdown]
# ## Benchmark the RAG System

# %% [markdown]
# ### Get eval dataset synthesized in Part 1
#  (https://lightning.ai/panchamsnotes/studios/evaluate-your-rag-part-1-synthesize-an-evaluation-dataset?section=featured)

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
### You can load it using the following;::

eval_dataset = datasets.load_dataset("delayedkarma/litgpt_instruction_qa", split="train")

# %%
def run_rag_tests(
    eval_dataset: datasets.Dataset,
    llm: BaseChatModel,
    knowledge_index: VectorStore,
    output_file: str,
    reranker: Optional[RAGPretrainedModel] = None,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = get_rag_response(
            question, llm, knowledge_index, reranker=reranker
        )
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_doc"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)

# %% [markdown]
# ### Define evaluation prompt

# %%
EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""


evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)

# %% [markdown]
# ### Define evaluator models 

# %%
eval_chat_model_gpt4_1106 = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
evaluator_name_gpt4_1106 = "GPT4_1106"

eval_chat_model_gpt4_0125 = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
evaluator_name_gpt4_0125 = "GPT4_0125"


def evaluate_rag_responses(
    answer_path: str,
    eval_chat_model: BaseChatModel,
    evaluator_name: str,
    evaluation_prompt_template: ChatPromptTemplate,
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        feedback, score = [
            item.strip() for item in eval_result.content.split("[RESULT]")
        ]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f)

# %% [markdown]
# ### Run the tests and evaluate the responses

# %%
if not os.path.exists("./output"):
    os.mkdir("./output")

for chunk_size in [200]:  # Add other chunk sizes (in tokens) as needed
    for embeddings in ["thenlper/gte-small", "BAAI/bge-small-en-v1.5"]:  # Add other embeddings as needed
        for rerank in [True, False]:
            settings_name = f"chunk:{chunk_size}_embeddings:{embeddings.replace('/', '~')}_rerank:{rerank}_reader-model:{READER_MODEL_NAME}"
            output_file_name = f"./output/rag_{settings_name}.json"

            print(f"Running evaluation for {settings_name}:")

            print("Loading knowledge base embeddings...")
            knowledge_index = create_vector_index(
                RAW_KNOWLEDGE_BASE,
                chunk_size=chunk_size,
                embedding_model_name=embeddings
            )

            print("Running RAG...")
            reranker = (
                RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
                if rerank
                else None
            )
            run_rag_tests(
                eval_dataset=eval_dataset,
                llm=READER_LLM,
                knowledge_index=knowledge_index,
                output_file=output_file_name,
                reranker=reranker,
                verbose=False,
                test_settings=settings_name,
            )

            print("Running evaluation for gpt-4-0125-preview ...")
            print()
            evaluate_rag_responses(
                output_file_name,
                eval_chat_model_gpt4_0125,
                evaluator_name_gpt4_0125,
                evaluation_prompt_template,
            )

            print("Running evaluation for gpt-4-1106-preview ...")
            print()
            evaluate_rag_responses(
                output_file_name,
                eval_chat_model_gpt4_1106,
                evaluator_name_gpt4_1106,
                evaluation_prompt_template,
            )


# %% [markdown]
# ### Inspect the results

# %%
outputs = []

for file in glob.glob("./output/*.json"):
    print(file)
    output = pd.DataFrame(json.load(open(file, "r")))
    output["settings"] = file
    outputs.append(output)
result = pd.concat(outputs)

# %%
# result.drop(['eval_score_GPT35', 'eval_feedback_GPT35','eval_score_GPT4', 'eval_feedback_GPT4'], axis=1, inplace=True) # artifacts from previous run
result.columns

# %%
result.head(2)

# %%
result["eval_score_GPT4_0125"] = result["eval_score_GPT4_0125"].apply(
    lambda x: int(x) if isinstance(x, str) else 1
)
result["eval_score_GPT4_0125"] = (result["eval_score_GPT4_0125"] - 1) / 4

result["eval_score_GPT4_1106"] = result["eval_score_GPT4_1106"].apply(
    lambda x: int(x) if isinstance(x, str) else 1
)
result["eval_score_GPT4_1106"] = (result["eval_score_GPT4_1106"] - 1) / 4

# %%
average_scores = result.groupby("settings")["eval_score_GPT4_1106"].mean()
average_scores.sort_values()

# %%
average_scores = result.groupby("settings")["eval_score_GPT4_0125"].mean()
average_scores.sort_values()

# %%




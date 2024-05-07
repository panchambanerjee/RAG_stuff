# %% [markdown]
# ## Objective: To build a simple RAG evaluation framework
# 
# ### Part 1: Synthesize and filter an Instruction dataset from a custom knowledge-base
# 
# #### Primary reference: https://huggingface.co/learn/cookbook/en/rag_evaluation by Aymeric Roucher (https://huggingface.co/m-ric)
# 
# For the knowledge base, let us use the  litgpt Github repo: https://github.com/Lightning-AI/litgpt/tree/main -> Only use markdown files, this ensures the knowledge base isnt too large (as a first effort),  and we get the quickstart, tutorials etc. **should** mean coherent QAs

# %% [markdown]
# ### Installs and Dependencies

# %%
%pip install --quiet torch transformers langchain tqdm pandas datasets
%pip install -U --quiet langchain-openai Gitpython python-dotenv huggingface_hub
%pip install --quiet openai huggingface langchain_experimental sentence_transformers

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
import bs4
import glob

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


from huggingface_hub import InferenceClient


# %% [markdown]
# ### Load in knowledge base and prepare documents

# %%
loader = GitLoader(
    clone_url="https://github.com/Lightning-AI/litgpt",
    repo_path="./litgpt_data_github/",
    branch="main",
    file_filter=lambda file_path: file_path.endswith(".md") # Only get the markdown files
)

data = loader.load()

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", "", "\n\n\n"],
)

docs_processed = []
for doc in data:
    docs_processed += text_splitter.split_documents([doc])

# %% [markdown]
# ### Setup Question-Answer generation agent

# %%
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm_client = InferenceClient(
    model=repo_id,
    timeout=120,
    token=hf_api_key
)


def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]

QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""

# %%
len(docs_processed)

# %%
# Generate 136 samples for now, upload to Huggingface Hub to use later


N_GENERATIONS = len(docs_processed)  # We intentionally generate only 136 QA couples here for cost and time considerations

# This number is just the total length of docs_processed using the knowledge-base from litgpt markdown files

print(f"Generating {N_GENERATIONS} QA couples...")

outputs = []
for sampled_context in tqdm(random.sample(docs_processed, N_GENERATIONS)):
    # Generate QA couple
    output_QA_couple = call_llm(
        llm_client, QA_generation_prompt.format(context=sampled_context.page_content)
    )
    try:
        question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
        answer = output_QA_couple.split("Answer: ")[-1]
        assert len(answer) < 300, "Answer is too long"
        outputs.append(
            {
                "context": sampled_context.page_content,
                "question": question,
                "answer": answer,
                "source_doc": sampled_context.metadata["source"],
            }
        )
    except:
        continue

# %%
display(pd.DataFrame(outputs).sample(5))

# %% [markdown]
# ### Set-up Critique Agents

# %%
question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to AI and ML Practitioners working with Large Language Models using litgpt.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain specific technical nouns or acronyms like LoRA, fp4, litgpt or Llama 3 and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

# %%
print("Generating critique for each QA couple...")
for output in tqdm(outputs):
    evaluations = {
        "groundedness": call_llm(
            llm_client,
            question_groundedness_critique_prompt.format(
                context=output["context"], question=output["question"]
            ),
        ),
        "relevance": call_llm(
            llm_client,
            question_relevance_critique_prompt.format(question=output["question"]),
        ),
        "standalone": call_llm(
            llm_client,
            question_standalone_critique_prompt.format(question=output["question"]),
        ),
    }
    try:
        for criterion, evaluation in evaluations.items():
            score, eval = (
                int(evaluation.split("Total rating: ")[-1].strip()),
                evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
            )
            output.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
    except Exception as e:
        continue

# %%
# Filter out the bad questions, keep 3 as the threshold for now

pd.set_option("display.max_colwidth", None)

generated_questions = pd.DataFrame.from_dict(outputs)

# Filter out low rated QA pairs

print("Evaluation dataset before filtering:")
display(
    generated_questions[
        [
            "question",
            "answer",
            "groundedness_score",
            "relevance_score",
            "standalone_score",
        ]
    ]
)
generated_questions = generated_questions.loc[
    (generated_questions["groundedness_score"] >= 3)
    & (generated_questions["relevance_score"] >= 3)
    & (generated_questions["standalone_score"] >= 3)
]
print("============================================")
print("Final evaluation dataset:")

generated_questions.reset_index(inplace=True, drop=True)

display(
    generated_questions[
        [
            "question",
            "answer",
            "groundedness_score",
            "relevance_score",
            "standalone_score",
        ]
    ]
)


eval_dataset = datasets.Dataset.from_pandas(
    generated_questions, split="train", preserve_index=False
)

# %% [markdown]
# ### Push dataset to HuggingFace Hub for future use

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
from huggingface_hub import create_repo
from huggingface_hub import Repository

repo_name = "litgpt_instruction_qa"  # Choose a name for your dataset repository
repo_url = create_repo(repo_name, repo_type="dataset")
print("Repository URL:", repo_url)

# %%
eval_dataset.push_to_hub(f"delayedkarma/litgpt_instruction_qa")

# %%
### Dataset is now pushed to hub

### You can load it using the following;::

# eval_dataset = datasets.load_dataset("delayedkarma/litgpt_instruction_qa", split="train")

# %% [markdown]
# ### In Part 2: Build and evaluate a RAG system using the synthesized dataset (LLM-as-a-judge)

# %%




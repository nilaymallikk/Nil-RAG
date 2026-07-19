# from src.config import (
#     OPENROUTER_API_KEY,
#     PINECONE_API_KEY,
#     PINECONE_INDEX_NAME,
# )

# from src.llm import test_llm
# response = test_llm()
# print(response.choices[0].message.content)


# from src.embedding import get_embedding
# embedding = get_embedding("Artificial Intelligence")
# vector = embedding  
# print(type(vector), len(vector))

# from src.loader import load_pdf
# documents = load_pdf("data/ml-Ajay Anand.pdf")
# # print(type(documents), len(documents))
# # print(documents[0].page_content)
# # print(documents[0].metadata)

# for i, doc in enumerate(documents[:3]):
#     print("=" * 60)
#     print(f"Page: {doc.metadata['page']}")
#     print()
#     print(doc.page_content[:300])

# from src.loader import load_pdf
# from src.splitter import split_documents
# documents = load_pdf("data/ml-Ajay Anand.pdf")
# chunks = split_documents(documents)
# print(f"page count: {len(documents)}, chunk count: {len(chunks)}")
# print(chunks[17].page_content)
# print(chunks[0].metadata)

# for i, chunk in enumerate(chunks[:3]):
#     print("=" * 70)
#     print(f"Chunk {i}")
#     print()
#     print(chunk.page_content)
#     print()
#     print(chunk.metadata)

# from src.loader import load_pdf
# from src.splitter import split_documents
# from src.vector_store import upload_chunks
# documents = load_pdf("data/ml-Ajay Anand.pdf")
# chunks = split_documents(documents)
# upload_chunks(chunks)


# from src.retriever import retrieve
# query = "What is K means clustering? and How does it work?"
# results = retrieve(query)
# # print(results)

# for match in results.matches:
#     print("=" * 80)
#     print("Score:", match.score)
#     print("Page:", match.metadata["page"])
#     print("Source:", match.metadata["source"])
#     print(match.metadata["text"])

import argparse

from src.indexer import index_pdf
from src.chatbot import ask_question
from src.memory import ConversationMemory

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["index", "chat"],
    required=True,
)

parser.add_argument(
    "--pdf",
    default="data/document.pdf",
)

args = parser.parse_args()

if args.mode == "index":

    index_pdf(args.pdf)

elif args.mode == "chat":

    memory = ConversationMemory(max_turns=4) # one instance, lives for the whole session

    while True:

        question = input("\nAsk: ")

        if question.lower() in ["exit", "quit"]:
            break

        result = ask_question(question, memory=memory)

        print("\nAnswer:\n")
        print(result["answer"])

        print("\nSources:")

        for source in result["sources"]:
            print(
                f"- {source['source']} | "
                f"Page {source['page']} | "
                f"Score: {source['score']}"
            )
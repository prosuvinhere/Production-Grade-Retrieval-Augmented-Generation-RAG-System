import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from src.retrieval.vector_store import VectorStore
from src.generation.llm_chain import RAGChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

def load_golden_dataset(filepath=None):
    if filepath is None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        filepath = root_dir / "tests" / "golden_dataset.json"
    with open(filepath, 'r', encoding="utf-8") as f:
        return json.load(f)

def run_evaluation(threshold=None):
    threshold = threshold if threshold is not None else float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", 0.80))
    vs = VectorStore()
    vs.create_or_load()
    
    retriever = vs.get_retriever(k=3)
    def retriever_func(query):
        return retriever.invoke(query)
        
    chain = RAGChain()
    dataset = load_golden_dataset()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in dataset:
        q = item['question']
        gt = item['answer']
        
        # 1. Retrieve
        docs = retriever_func(q)
        ctx = [doc.page_content for doc in docs]
        
        # 2. Generate
        ans = chain.invoke(retriever_func, q)
        
        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
        ground_truths.append(gt)
        
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    hf_dataset = Dataset.from_dict(data)
    
    # 3. Evaluate using Ragas
    result = evaluate(
        hf_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=ChatOpenAI(model="gpt-4o"),
        embeddings=OpenAIEmbeddings()
    )
    
    print("Evaluation Results:")
    print(result)
    
    # Fail CI build if faithfulness drops below a specific threshold (default: 0.80)
    faithfulness_score = result['faithfulness']
    if faithfulness_score < threshold:
        print(f"❌ Error: Faithfulness ({faithfulness_score:.4f}) is below threshold ({threshold})")
        sys.exit(1)
        
    print(f"✅ Evaluation passed! Faithfulness: {faithfulness_score:.4f} >= {threshold}")

if __name__ == "__main__":
    run_evaluation()

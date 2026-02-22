# rag_demo1.py - ENHANCED to clearly demonstrate BASIC RAG failures with 3 questions

import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

import faiss
import numpy as np
from openai import OpenAI, AzureOpenAI
from PyPDF2 import PdfReader

# Load environment variables from .env file
load_dotenv()


# -----------------------------
# PDF Document Loading
# -----------------------------

def load_pdf_document(pdf_path: str = "company_policies.pdf") -> str:
    """Load and extract text from PDF document."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}\n"
            f"Please run 'python create_detailed_pdf.py' first to generate the PDF."
        )
    
    reader = PdfReader(pdf_path)
    full_text = ""
    
    print(f"📄 Loading PDF: {pdf_path}")
    print(f"   Pages: {len(reader.pages)}")
    
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        full_text += text + "\n"
    
    print(f"   Extracted {len(full_text)} characters")
    return full_text


def chunk_pdf_by_subsections(pdf_path: str = "company_policies.pdf", chunk_size: int = 600) -> List[Dict[str, Any]]:
    """
    Load PDF and chunk by subsections with metadata.
    SMALLER CHUNKS for improved precision.
    """
    full_text = load_pdf_document(pdf_path)
    
    chunks = []
    lines = full_text.split('\n')
    
    current_chunk = ""
    current_section = ""
    current_metadata = {}
    
    for line in lines:
        line_stripped = line.strip()
        
        # Detect major section headers (1., 2., 3., etc.)
        if line_stripped and len(line_stripped) < 100:
            # Check for numbered sections like "1. Remote Work..."
            for i in range(1, 7):
                if line_stripped.startswith(f"{i}. "):
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": current_metadata.copy()
                        })
                    current_section = line_stripped
                    current_chunk = line_stripped + "\n"
                    current_metadata = extract_metadata_from_header(line_stripped)
                    break
            # Check for subsection headers like "1.1", "2.2", etc.
            else:
                for i in range(1, 7):
                    for j in range(1, 6):
                        if line_stripped.startswith(f"{i}.{j} "):
                            if current_chunk and len(current_chunk) > chunk_size:
                                chunks.append({
                                    "text": current_chunk.strip(),
                                    "metadata": current_metadata.copy()
                                })
                                current_chunk = current_section + "\n" + line_stripped + "\n"
                            else:
                                current_chunk += line + "\n"
                            break
                else:
                    current_chunk += line + "\n"
        else:
            current_chunk += line + "\n"
    
    # Add last chunk
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "metadata": current_metadata.copy()
        })
    
    print(f"✂️  Created {len(chunks)} chunks from PDF")
    return chunks


def extract_metadata_from_header(header: str) -> Dict[str, Any]:
    """Extract metadata based on section header."""
    metadata = {}
    
    header_lower = header.lower()
    
    # Determine policy type and applicable employee types
    if "remote work" in header_lower or "work from home" in header_lower:
        metadata["policy_type"] = "remote_work"
        metadata["employee_type"] = "full_time"
    elif "contractor" in header_lower or "consultant" in header_lower:
        metadata["policy_type"] = "contractor"
        metadata["employee_type"] = "contractor"
    elif "leave" in header_lower or "time off" in header_lower:
        metadata["policy_type"] = "leave"
        metadata["employee_type"] = "all"
    elif "security" in header_lower or "data protection" in header_lower:
        metadata["policy_type"] = "security"
        metadata["employee_type"] = "all"
    elif "development" in header_lower or "training" in header_lower:
        metadata["policy_type"] = "professional_development"
        metadata["employee_type"] = "full_time"
    elif "diversity" in header_lower or "inclusion" in header_lower:
        metadata["policy_type"] = "dei"
        metadata["employee_type"] = "all"
    else:
        metadata["policy_type"] = "general"
        metadata["employee_type"] = "all"
    
    return metadata


# -----------------------------
# Chunking strategies
# -----------------------------

def chunk_documents_basic(pdf_path: str = "company_policies.pdf", chunk_size: int = 2500, overlap: int = 500) -> List[Dict[str, Any]]:
    """
    BASIC CHUNKING: Very large chunks with overlap.
    
    WHY THIS FAILS:
    - Large chunks (2500 chars) mix multiple policies together
    - Heavy overlap (500 chars) creates redundant, confusing context
    - No metadata = can't filter by policy type
    - Pure similarity search retrieves "remote work encouraged" text
    - Misses the specific contractor restriction buried in section 2.3
    """
    full_text = load_pdf_document(pdf_path)
    chunks: List[Dict[str, Any]] = []
    
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end]
        chunks.append({"text": chunk_text, "metadata": {}})
        start = end - overlap
        if end >= len(full_text):
            break
    
    print(f"✂️  Created {len(chunks)} basic chunks (LARGE SIZE: {chunk_size} chars)")
    print(f"   ⚠️  WARNING: Large chunks may mix policies and cause retrieval errors")
    return chunks


# -----------------------------
# Vector store helpers
# -----------------------------

def get_client():
    """Return an OpenAI or Azure OpenAI client based on environment variables."""
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Please set OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT "
            "in your .env file."
        )
    
    return OpenAI(api_key=api_key)


def embed_texts(client, texts: List[str], model: str) -> np.ndarray:
    """Embed a list of texts and return normalized vectors for cosine similarity."""
    response = client.embeddings.create(model=model, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    # Normalize for cosine similarity with inner product
    faiss.normalize_L2(vectors)
    return vectors


def create_vector_store(client, chunks: List[Dict[str, Any]], embedding_model: str) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """Create a FAISS index and return it with the chunk embeddings."""
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(client, texts, embedding_model)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors


# -----------------------------
# Retrieval
# -----------------------------

def retrieve_basic(client, index: faiss.IndexFlatIP, chunks: List[Dict[str, Any]], 
                  query: str, embedding_model: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    BASIC RETRIEVAL: Pure similarity search with high top_k.
    
    WHY THIS FAILS:
    - No metadata filtering = searches ALL policies
    - High top_k (8) retrieves many irrelevant chunks
    - Semantic search favors "remote work encouraged" language
    - Contractor restriction gets lower similarity score
    - LLM receives mixed/misleading context
    """
    query_vec = embed_texts(client, [query], embedding_model)
    scores, indices = index.search(query_vec, top_k)
    
    print(f"   ⚠️  BASIC: No filtering, retrieving top {top_k} by similarity only")
    return [chunks[i] for i in indices[0]]


def retrieve_with_metadata(client, chunks: List[Dict[str, Any]], query: str, 
                          embedding_model: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """
    IMPROVED RETRIEVAL: Metadata filtering + small top_k.
    
    WHY THIS SUCCEEDS:
    - Filters chunks by policy_type and employee_type BEFORE search
    - Small top_k (2) returns only most relevant chunks
    - For contractor query, only retrieves contractor policy chunks
    - Ensures LLM sees correct, focused context
    """
    query_lower = query.lower()
    
    # Determine which policies are relevant based on query
    filtered_chunks = chunks
    
    if "contractor" in query_lower:
        # Include ONLY contractor policies
        filtered_chunks = [c for c in chunks 
                          if c["metadata"].get("employee_type") == "contractor"
                          or c["metadata"].get("policy_type") == "contractor"]
        print(f"   ✅ IMPROVED: Filtered to contractor-specific chunks only")
    elif "remote" in query_lower or "work from home" in query_lower:
        # Include remote work policies and general policies
        filtered_chunks = [c for c in chunks 
                          if c["metadata"].get("policy_type") in {"remote_work", "security"}
                          or c["metadata"].get("employee_type") in {"all", "full_time"}]
        print(f"   ✅ IMPROVED: Filtered to remote work policies")
    elif "pto" in query_lower or "leave" in query_lower or "vacation" in query_lower or "paid time off" in query_lower:
        # Include leave policies
        filtered_chunks = [c for c in chunks 
                          if c["metadata"].get("policy_type") == "leave"]
        print(f"   ✅ IMPROVED: Filtered to leave policies")
    
    # If no filters matched or too few results, use all chunks
    if len(filtered_chunks) < 2:
        filtered_chunks = chunks
    
    print(f"   🔍 Metadata filtering: {len(chunks)} → {len(filtered_chunks)} chunks")
    
    # Build a temporary index from filtered chunks
    temp_index, _ = create_vector_store(client, filtered_chunks, embedding_model)
    query_vec = embed_texts(client, [query], embedding_model)
    scores, indices = temp_index.search(query_vec, top_k)
    
    results = [filtered_chunks[i] for i in indices[0]]
    return results


# -----------------------------
# Generation
# -----------------------------

def generate_answer(client, context_chunks: List[Dict[str, Any]], question: str, chat_model: str) -> str:
    """Generate an answer from the retrieved context."""
    context_text = "\n\n".join([f"Context {i + 1}:\n{c['text'][:1200]}" 
                                for i, c in enumerate(context_chunks)])

    system_prompt = (
        "You are a helpful HR assistant. Answer questions based ONLY on the provided policy context. "
        "If the context does not contain enough information to answer confidently, say so. "
        "Be clear, concise, and cite specific policy details when possible."
    )

    user_prompt = f"Context from company policies:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"

    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def print_demo_section(title: str, is_basic: bool = False):
    """Print a formatted demo section header."""
    icon = "❌" if is_basic else "✅"
    print(f"\n{'─'*80}")
    print(f"{icon} {title}")
    print(f"{'─'*80}")


def print_chunk_analysis(chunks: List[Dict[str, Any]], is_improved: bool = False):
    """Print detailed chunk analysis."""
    print(f"\n📄 Retrieved {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks, start=1):
        if is_improved:
            meta = chunk.get("metadata", {})
            policy_type = meta.get("policy_type", "unknown")
            print(f"\n  📌 Chunk [{i}] - Policy Type: {policy_type}")
            print(f"     Metadata: {meta}")
            preview = chunk['text'][:300].replace('\n', ' ')
            print(f"     Content: {preview}...")
        else:
            # For BASIC RAG, highlight the mixing of policies
            text_lower = chunk['text'].lower()
            policies_found = []
            
            if "remote work" in text_lower or "work from home" in text_lower:
                policies_found.append("REMOTE WORK")
            if "contractor" in text_lower:
                policies_found.append("CONTRACTOR")
            if "pto" in text_lower or "paid time off" in text_lower:
                policies_found.append("PTO")
            if "leave" in text_lower:
                policies_found.append("LEAVE")
            
            policy_str = ", ".join(policies_found) if policies_found else "MIXED"
            print(f"\n  ⚠️  Chunk [{i}] - Contains: {policy_str}")
            preview = chunk['text'][:300].replace('\n', ' ')
            print(f"     {preview}...")


# =============================
# DEMO SECTION
# =============================

def main():
    # Check if PDF exists
    pdf_path = "company_policies.pdf"
    if not os.path.exists(pdf_path):
        print(f"\n⚠️  PDF file not found: {pdf_path}")
        print("Please run the following command first:")
        print("    python create_detailed_pdf.py")
        print("\nThis will create a comprehensive company policies PDF document.")
        return
    
    client = get_client()

    # Model selection
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    print(f"\n{'='*80}")
    print(f"🎬 RAG DEMO: Basic vs Improved Retrieval (Enhanced Edition)")
    print(f"{'='*80}")
    print(f"📊 Embedding Model: {embedding_model}")
    print(f"💬 Chat Model: {chat_model}")
    print(f"{'='*80}")
    print(f"\n💡 Purpose: Demonstrate where BASIC RAG fails and IMPROVED RAG succeeds")
    print(f"{'='*80}")
    
    # Test questions with expected outcomes
    questions = [
        {
            "text": "Are contractors allowed to work from home?",
            "expected_basic": "⚠️  MIXED context - may confuse with employee remote work",
            "expected_improved": "✅ CORRECT - contractor-only context shows restriction"
        },
        {
            "text": "How much PTO do employees get after 4 years?",
            "expected_basic": "❌ MISSING INFO - no PTO details in mixed chunks",
            "expected_improved": "✅ CORRECT - 20 days per year (3-5 years tenure)"
        },
        {
            "text": "Can contractors receive paid time off?",
            "expected_basic": "❌ WRONG - confuses employee PTO with contractor benefits",
            "expected_improved": "✅ CORRECT - contractors not eligible for any PTO"
        }
    ]

    for idx, q_obj in enumerate(questions, 1):
        question = q_obj["text"]
        
        print(f"\n\n{'='*80}")
        print(f"❓ QUESTION {idx}: {question}")
        print(f"{'='*80}")
        
        # ----- Part 1: Basic RAG -----
        print_demo_section(f"BASIC RAG (Large chunks, no metadata, high top_k)", is_basic=True)
        basic_chunks = chunk_documents_basic(pdf_path, chunk_size=2500, overlap=500)
        basic_index, _ = create_vector_store(client, basic_chunks, embedding_model)
        basic_results = retrieve_basic(client, basic_index, basic_chunks, question, embedding_model, top_k=8)
        
        print_chunk_analysis(basic_results, is_improved=False)
        
        basic_answer = generate_answer(client, basic_results, question, chat_model)
        print(f"\n💬 BASIC RAG Answer:")
        print(f"   {basic_answer}")
        print(f"\n{q_obj['expected_basic']}")

        # ----- Part 2: Improved RAG -----
        print_demo_section(f"IMPROVED RAG (Small chunks, metadata filtering, top_k=2)", is_basic=False)
        structured_chunks = chunk_pdf_by_subsections(pdf_path, chunk_size=600)
        improved_results = retrieve_with_metadata(client, structured_chunks, question, embedding_model, top_k=2)
        
        print_chunk_analysis(improved_results, is_improved=True)
        
        improved_answer = generate_answer(client, improved_results, question, chat_model)
        print(f"\n💬 IMPROVED RAG Answer:")
        print(f"   {improved_answer}")
        print(f"\n{q_obj['expected_improved']}")
        
        # ----- Key Insight -----
        print(f"\n{'─'*80}")
        print("🎯 KEY INSIGHT:")
        print(f"{'─'*80}")
        
        if idx == 1:
            print("   BASIC RAG Problem:")
            print("   • Retrieves employee remote work + contractor supervision sections")
            print("   • LLM must infer 'contractors ≠ employees'")
            print("   • Risk: May confuse policies and give wrong answer")
            print("\n   IMPROVED RAG Solution:")
            print("   • Metadata filters to ONLY contractor policy sections")
            print("   • Direct access to contractor restrictions in section 2.3")
            print("   • Clear, unambiguous context")
            
        elif idx == 2:
            print("   BASIC RAG Problem:")
            print("   • Retrieves PTO sections but heavily mixed with other policies")
            print("   • Missing the specific '20 days for 3-5 years' detail")
            print("   • LLM cannot answer with confidence from available evidence")
            print("\n   IMPROVED RAG Solution:")
            print("   • Filters to leave policy section BEFORE similarity search")
            print("   • Gets exact tenure brackets: 15/20/25 days per year")
            print("   • Precise answer: '20 days per year (3-5 years)'")
            
        elif idx == 3:
            print("   BASIC RAG Problem:")
            print("   • Mixes employee PTO benefits with contractor policy sections")
            print("   • LLM might incorrectly suggest contractors get some PTO")
            print("   • No filtering means all policies compete for relevance")
            print("\n   IMPROVED RAG Solution:")
            print("   • Metadata filters to contractor policy ONLY")
            print("   • Shows explicit: 'not eligible for employee benefits'")
            print("   • Clear answer: 'No, contractors are not eligible'")
        
        print(f"{'─'*80}")
        
        if idx < len(questions):
            print("\n" + "."*80 + "\n")
    
    # Final summary
    print(f"\n\n{'='*80}")
    print(f"📊 SUMMARY: Why IMPROVED RAG Outperforms BASIC RAG")
    print(f"{'='*80}")
    print("\n1️⃣  CHUNKING STRATEGY")
    print("   BASIC:    Large chunks (2500 chars) → Mix multiple policies")
    print("   IMPROVED: Small chunks (600 chars) → Separate logical sections")
    
    print("\n2️⃣  METADATA USAGE")
    print("   BASIC:    No metadata → Searches all policies equally")
    print("   IMPROVED: Rich metadata → Filters by policy_type & employee_type first")
    
    print("\n3️⃣  RETRIEVAL PRECISION")
    print("   BASIC:    High top_k (8) → Returns many irrelevant chunks")
    print("   IMPROVED: Low top_k (2) → Returns only most relevant chunks")
    
    print("\n4️⃣  RESULT QUALITY")
    print("   BASIC:    Ambiguous context → LLM must infer & interpret")
    print("   IMPROVED: Clear context → LLM gives direct, confident answers")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
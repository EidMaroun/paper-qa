"""
MCP Server for Research Assistant Multi-Agent System

Provides the following tools:
1. research_paper_probe - Search and query research papers in the RAG database
2. search_arxiv - Search arXiv for academic papers
3. download_paper - Download PDF papers and auto-index them in the vector database
"""
import os
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

# Setup environment
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Add RAG SETUP to path for imports
rag_setup_path = Path(__file__).resolve().parent / "RAG SETUP"
sys.path.insert(0, str(rag_setup_path))
sys.path.append('../..')

# Import our tools from RAG SETUP
from RagTool import (
    ResearchProbeArgs,
    ResearchProbeResponse,
    _research_probe_fn
)
from corpus_expansion import (
    SearchArxivArgs,
    DownloadPdfArgs,
    search_arxiv,
    download_pdf
)

import openai
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Initialize MCP Server
mcp = FastMCP(
    name="research_assistant_mcp",
    host="0.0.0.0",
    port=8787
)


# ============== Tool 1: Research Paper Probe (RAG) ==============

@mcp.tool(
    name="research_paper_probe",
    description=(
        "Search AI research papers in the knowledge base to answer questions. "
        "Filters available: topic ('Agentic AI', 'Finetuning', 'Hierarchical Reasoning Models', 'Deep Learning'), "
        "year (publication year), subject (e.g., 'Artificial Intelligence'). "
        "Returns: topic, category, response (markdown), sources with paper titles and pages, confidence score."
    ),
)
def mcp_research_probe(
    query: str,
    topic: Optional[str] = None,
    subject: Optional[str] = None,
    year: Optional[int] = None,
    k: int = 10,
) -> Dict[str, Any]:
    """
    Search research papers and return structured response.
    """
    try:
        args = ResearchProbeArgs(
            query=query,
            topic=topic,
            subject=subject,
            year=year,
            k=k
        )
    except ValidationError as e:
        return {"error": "validation_error", "details": e.errors()}

    return _research_probe_fn(**args.model_dump(exclude_none=True))


# ============== Tool 2: Search arXiv ==============

@mcp.tool(
    name="search_arxiv",
    description=(
        "Search arXiv for academic papers. "
        "Returns a list of papers with title, abstract, authors, year, pdf_url, subject, and topic. "
        "Use subject and topic to organize results (e.g., subject='Artificial Intelligence', topic='Healthcare')."
    ),
)
def mcp_search_arxiv(
    query: str,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    max_results: int = 10,
) -> Dict[str, Any]:
    """
    Search arXiv for papers matching the query.
    """
    try:
        args = SearchArxivArgs(
            query=query,
            subject=subject,
            topic=topic,
            max_results=max_results
        )
    except ValidationError as e:
        return {"error": "validation_error", "details": e.errors()}

    return search_arxiv(**args.model_dump())


# ============== Tool 3: Download Paper ==============

@mcp.tool(
    name="download_paper",
    description=(
        "Download a PDF paper from arXiv and automatically add it to the RAG vector database. "
        "Use the pdf_url and title from search_arxiv results. "
        "Papers are saved to: Papers/subject/topic/title - year.pdf and indexed for future queries."
    ),
)
def mcp_download_paper(
    pdf_url: str,
    title: str,
    year: Optional[int] = None,
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    add_to_vectordb: bool = True,
) -> Dict[str, Any]:
    """
    Download a PDF paper and optionally index it in the vector database.
    """
    try:
        args = DownloadPdfArgs(
            pdf_url=pdf_url,
            title=title,
            year=year,
            subject=subject,
            topic=topic,
            add_to_vectordb=add_to_vectordb
        )
    except ValidationError as e:
        return {"error": "validation_error", "details": e.errors()}

    return download_pdf(**args.model_dump())


# ============== Server Entry Point ==============

if __name__ == "__main__":
    print("Starting Research Assistant MCP Server...")
    print("Available tools:")
    print("  1. research_paper_probe - Query the RAG knowledge base")
    print("  2. search_arxiv - Search arXiv for papers")
    print("  3. download_paper - Download and index papers")
    mcp.run(transport="sse")
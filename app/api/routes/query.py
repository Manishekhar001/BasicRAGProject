"""Query endpoints for RAG Q&A."""

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ErrorResponse,
    EvaluationScores,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from app.core.rag_chain import RAGChain
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


def get_rag_chain(request: Request) -> RAGChain:
    """Dependency: return the shared RAGChain from app state."""
    return request.app.state.rag_chain


@router.post(
    "",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Query processing error"},
    },
    summary="Ask a question",
    description="Submit a question and get an AI-generated answer based on the ingested documents.",
)
async def query(
    body: QueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
) -> QueryResponse:
    """Process a RAG query."""
    logger.info(
        f"Query received: {body.question[:100]}... "
        f"(sources={body.include_sources}, eval={body.enable_evaluation})"
    )
    start_time = time.time()

    try:
        # Determine which method to call based on request
        if body.enable_evaluation:
            # Evaluation requires sources, so we always include them
            result = await rag_chain.aquery_with_evaluation(
                question=body.question,
                include_sources=body.include_sources,
            )

            sources = (
                [
                    SourceDocument(
                        content=source["content"],
                        metadata=source["metadata"],
                    )
                    for source in result["sources"]
                ]
                if body.include_sources
                else None
            )

            answer = result["answer"]
            evaluation = EvaluationScores(**result["evaluation"])

        elif body.include_sources:
            result = await rag_chain.aquery_with_sources(body.question)
            sources = [
                SourceDocument(
                    content=source["content"],
                    metadata=source["metadata"],
                )
                for source in result["sources"]
            ]
            answer = result["answer"]
            evaluation = None
        else:
            answer = await rag_chain.aquery(body.question)
            sources = None
            evaluation = None

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Query processed in {processing_time:.2f}ms "
            f"(eval_included={body.enable_evaluation})"
        )

        return QueryResponse(
            question=body.question,
            answer=answer,
            sources=sources,
            processing_time_ms=round(processing_time, 2),
            evaluation=evaluation,
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


@router.post(
    "/stream",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Query processing error"},
    },
    summary="Ask a question (streaming)",
    description="Submit a question and get a streaming AI-generated answer.",
)
async def query_stream(
    body: QueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
) -> StreamingResponse:
    """Process a RAG query with streaming response."""
    logger.info(f"Streaming query received: {body.question[:100]}...")

    try:
        async def generate():
            """Generate streaming response."""
            try:
                async for chunk in rag_chain.chain.astream(body.question):
                    yield chunk
            except Exception as e:
                logger.error(f"Error in stream: {e}")
                yield f"\n\nError: {str(e)}"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
        )

    except Exception as e:
        logger.error(f"Error setting up stream: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


@router.post(
    "/search",
    responses={
        500: {"model": ErrorResponse, "description": "Search error"},
    },
    summary="Search documents",
    description="Search for relevant documents without generating an answer.",
)
async def search_documents(
    body: QueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
) -> dict:
    """Search for relevant documents."""
    logger.info(f"Search received: {body.question[:100]}...")

    try:
        results = rag_chain.vector_store.search_with_score(body.question)

        documents = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": round(score, 4),
            }
            for doc, score in results
        ]

        return {
            "query": body.question,
            "results": documents,
            "count": len(documents),
        }

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}",
        )

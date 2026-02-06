"""
Document processor for RAG pipeline.

Handles loading, parsing, and chunking of runbook documentation.
Supports markdown files with intelligent section-aware chunking.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""

    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Get the source file name."""
        return self.metadata.get("source", "unknown")

    @property
    def section(self) -> str:
        """Get the section name."""
        return self.metadata.get("section", "")

    def __hash__(self) -> int:
        return hash(self.chunk_id)


@dataclass
class Document:
    """Represents a full document."""

    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    chunks: list[DocumentChunk] = field(default_factory=list)


class DocumentProcessor:
    """
    Processes documents for RAG indexing.
    
    Features:
    - Markdown-aware parsing
    - Section-based chunking
    - Code block preservation
    - Metadata extraction
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self._chunk_size = chunk_size or settings.rag.chunk_size
        self._chunk_overlap = chunk_overlap or settings.rag.chunk_overlap
        self._logger = logger.bind(component="document_processor")

    def _generate_doc_id(self, filepath: Path) -> str:
        """Generate unique document ID."""
        return hashlib.md5(str(filepath).encode()).hexdigest()[:12]

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        return f"{doc_id}-{chunk_index:04d}"

    def _extract_title(self, content: str) -> str:
        """Extract title from markdown content."""
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        return match.group(1).strip() if match else ""

    def _extract_sections(self, content: str) -> list[tuple[str, str]]:
        """
        Extract sections from markdown content.
        
        Returns:
            List of (section_name, section_content) tuples
        """
        # Split by headers (## or ###)
        sections: list[tuple[str, str]] = []
        
        # Pattern to match markdown headers
        header_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
        
        matches = list(header_pattern.finditer(content))
        
        if not matches:
            # No sections found, return entire content
            return [("", content)]

        # Get content before first section
        if matches[0].start() > 0:
            intro = content[: matches[0].start()].strip()
            if intro:
                sections.append(("Introduction", intro))

        # Extract each section
        for i, match in enumerate(matches):
            section_name = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()
            
            if section_content:
                sections.append((section_name, section_content))

        return sections

    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        base_metadata: dict,
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            doc_id: Document ID for chunk ID generation
            base_metadata: Metadata to include in each chunk
            start_index: Starting index for chunk numbering
            
        Returns:
            List of DocumentChunks
        """
        chunks: list[DocumentChunk] = []
        
        if not text.strip():
            return chunks

        # Split by paragraphs first (preserve code blocks)
        paragraphs = self._split_preserving_code_blocks(text)
        
        current_chunk = ""
        chunk_index = start_index

        for para in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 1 > self._chunk_size:
                if current_chunk.strip():
                    chunks.append(
                        DocumentChunk(
                            chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                            content=current_chunk.strip(),
                            metadata=base_metadata.copy(),
                        )
                    )
                    chunk_index += 1
                    
                    # Keep overlap from end of current chunk
                    if self._chunk_overlap > 0:
                        overlap_text = current_chunk[-self._chunk_overlap :]
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add final chunk
        if current_chunk.strip():
            chunks.append(
                DocumentChunk(
                    chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                    content=current_chunk.strip(),
                    metadata=base_metadata.copy(),
                )
            )

        return chunks

    def _split_preserving_code_blocks(self, text: str) -> list[str]:
        """
        Split text into paragraphs while preserving code blocks.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraphs/code blocks
        """
        parts: list[str] = []
        
        # Pattern to match code blocks
        code_block_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        
        last_end = 0
        for match in code_block_pattern.finditer(text):
            # Add text before code block
            before = text[last_end : match.start()].strip()
            if before:
                # Split non-code text by paragraphs
                paragraphs = [p.strip() for p in before.split("\n\n") if p.strip()]
                parts.extend(paragraphs)
            
            # Add code block as single unit
            parts.append(match.group())
            last_end = match.end()

        # Add remaining text after last code block
        remaining = text[last_end:].strip()
        if remaining:
            paragraphs = [p.strip() for p in remaining.split("\n\n") if p.strip()]
            parts.extend(paragraphs)

        return parts

    def process_file(self, filepath: Path) -> Document:
        """
        Process a single file into a Document with chunks.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Document with chunks
        """
        self._logger.info("processing_file", filepath=str(filepath))

        content = filepath.read_text(encoding="utf-8")
        doc_id = self._generate_doc_id(filepath)
        title = self._extract_title(content)

        base_metadata = {
            "source": filepath.name,
            "filepath": str(filepath),
            "title": title,
            "doc_id": doc_id,
        }

        # Extract and chunk sections
        sections = self._extract_sections(content)
        all_chunks: list[DocumentChunk] = []
        chunk_index = 0

        for section_name, section_content in sections:
            section_metadata = {
                **base_metadata,
                "section": section_name,
            }
            
            chunks = self._chunk_text(
                section_content,
                doc_id,
                section_metadata,
                start_index=chunk_index,
            )
            all_chunks.extend(chunks)
            chunk_index += len(chunks)

        document = Document(
            doc_id=doc_id,
            content=content,
            metadata=base_metadata,
            chunks=all_chunks,
        )

        self._logger.info(
            "file_processed",
            filepath=str(filepath),
            title=title,
            chunk_count=len(all_chunks),
        )

        return document

    def process_directory(
        self,
        directory: Path,
        pattern: str = "*.md",
    ) -> list[Document]:
        """
        Process all matching files in a directory.
        
        Args:
            directory: Directory to process
            pattern: Glob pattern for files
            
        Returns:
            List of processed Documents
        """
        self._logger.info(
            "processing_directory",
            directory=str(directory),
            pattern=pattern,
        )

        documents: list[Document] = []
        
        for filepath in directory.glob(pattern):
            if filepath.is_file():
                try:
                    doc = self.process_file(filepath)
                    documents.append(doc)
                except Exception as e:
                    self._logger.error(
                        "file_processing_error",
                        filepath=str(filepath),
                        error=str(e),
                    )

        self._logger.info(
            "directory_processed",
            directory=str(directory),
            document_count=len(documents),
            total_chunks=sum(len(d.chunks) for d in documents),
        )

        return documents

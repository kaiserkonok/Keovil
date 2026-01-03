import os
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from langchain_core.documents import Document


class DocumentProcessor:
    """
    Service class to handle multi-format document conversion using Docling.
    Optimized for NVIDIA GPU acceleration and Markdown extraction.
    """

    def __init__(self, use_gpu: bool = True):
        # Configure hardware acceleration for your RTX 5060 Ti
        accel_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accel_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        # Initialize the converter service
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Formats Docling handles natively
        self.supported_extensions = {
            '.pdf', '.docx', '.pptx', '.md'
        }

    def convert_to_documents(self, file_paths: List[Path], chunker) -> List[Document]:
        docs = []
        complex_queue = []

        for fpath in file_paths:
            fpath = Path(fpath)
            if not fpath.exists():
                continue

            ext = fpath.suffix.lower()

            if ext == '.txt':
                self._process_text_file(fpath, chunker, docs)
            elif ext in self.supported_extensions:
                complex_queue.append(fpath)

        if complex_queue:
            docs.extend(self._process_complex_files(complex_queue, chunker))

        return docs

    def _process_text_file(self, fpath, chunker, docs_list):
        try:
            abs_path = str(fpath.absolute())

            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            if text.strip():
                chunks = chunker.chunk_document(text)
                docs_list.extend([
                    Document(
                        page_content=c.text,
                        metadata={
                            "source": abs_path,
                            "chunk_id": c.id,
                            **c.metadata # FIX: Changed from .meta to .metadata
                        }
                    ) for c in chunks
                ])
        except Exception as e:
            print(f"[DocumentProcessor] Failed .txt: {fpath.name} | Error: {e}")

    def _process_complex_files(self, paths, chunker) -> List[Document]:
        converted_docs = []
        results = self.converter.convert_all(paths, raises_on_error=False)

        for res in results:
            if res.status == ConversionStatus.SUCCESS:
                abs_path = str(Path(res.input.file).absolute())

                text = res.document.export_to_markdown()
                if text.strip():
                    chunks = chunker.chunk_document(text)
                    converted_docs.extend([
                        Document(
                            page_content=c.text,
                            metadata={
                                "source": abs_path,
                                "chunk_id": c.id,
                                **c.metadata # FIX: Changed from .meta to .metadata
                            }
                        ) for c in chunks
                    ])
            else:
                print(f"[DocumentProcessor] Failed: {res.input.file} | Status: {res.status}")

        return converted_docs
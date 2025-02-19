# Copyright 2023 The LangChain Authors. All rights reserved.
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

from pipelines.nodes.base import BaseComponent

logger = logging.getLogger(__name__)


class TextSplitter(BaseComponent):
    """Interface for splitting text into chunks."""

    outgoing_edges = 1

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        filters: list = [],
        separator: str = "",
    ):
        """Create a new TextSplitter."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size " f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._filter = filters
        self._separator = separator

    @abstractmethod
    def split_text(self, text: str, **kwargs) -> List[str]:
        """Split text into multiple components."""

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None, separator: Optional[str] = None, **kwargs
    ) -> List[dict]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text, separator, **kwargs):
                new_doc = {"content": chunk, "meta": copy.deepcopy(_metadatas[i])}
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: List[dict], **kwargs) -> List[dict]:
        """Split documents."""
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["meta"] for doc in documents]
        return self.create_documents(texts, metadatas, **kwargs)

    def _join_docs(self, docs: List[str], separator: str, **kwargs) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(
        self,
        splits: Iterable[str],
        separator: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        if chunk_size is None:
            chunk_size = self._chunk_size
        if chunk_overlap is None:
            chunk_overlap = self._chunk_overlap
        if separator is None:
            separator = self._separator
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
                if total > chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, " f"which is longer than the specified {chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0) > chunk_size and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def clean(self, documents: List[dict], filters: List[str]):
        for special_character in filters:
            for doc in documents:
                doc["content"] = doc["content"].replace(special_character, "")
        return documents

    def run(  # type: ignore
        self,
        documents: Union[dict, List[dict]],
        meta: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,  # type: ignore
        separator: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        filters: Optional[List[str]] = None,
    ):
        if separator is None:
            separator = self._separator
        if chunk_size is None:
            chunk_size = self._chunk_size
        if chunk_overlap is None:
            chunk_overlap = self._chunk_overlap
        if filters is None:
            filters = self._filter
        ret = []
        if type(documents) == dict:  # single document
            text_splits = self.split_text(
                documents["content"], separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            for i, txt in enumerate(text_splits):
                doc = copy.deepcopy(documents)
                doc["content"] = txt

                if "meta" not in doc.keys() or doc["meta"] is None:
                    doc["meta"] = {}

                doc["meta"]["_split_id"] = i
                ret.append(doc)

        elif type(documents) == list:  # list document
            for document in documents:
                text_splits = self.split_text(
                    document["content"], separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                for i, txt in enumerate(text_splits):
                    doc = copy.deepcopy(document)
                    doc["content"] = txt

                    if "meta" not in doc.keys() or doc["meta"] is None:
                        doc["meta"] = {}

                    doc["meta"]["_split_id"] = i
                    ret.append(doc)
        if filters is not None and len(filters) > 0:
            ret = self.clean(ret, filters)
        result = {"documents": ret}
        return result, "output_1"


class CharacterTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters."""

    def __init__(self, separator: str = "\n\n", filters: list = [], **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator
        self._filter = filters

    def split_text(self, text: str, separator: Optional[str] = None, **kwargs) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        if separator is None:
            separator = self._separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        return self._merge_splits(splits, separator, **kwargs)


class RecursiveCharacterTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters.
    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(self, separators: Optional[List[str]] = None, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str, **kwargs) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = self._separators[-1]
        for _s in self._separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        # Now that we have the separator, split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(
                        _good_splits,
                        separator,
                        chunk_size=kwargs.get("chunk_size", None),
                        chunk_overlap=kwargs.get("chunk_overlap", None),
                    )
                    final_chunks.extend(merged_text)
                    _good_splits = []
                other_info = self.split_text(s)
                final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(
                _good_splits,
                separator,
                chunk_size=kwargs.get("chunk_size", None),
                chunk_overlap=kwargs.get("chunk_overlap", None),
            )
            final_chunks.extend(merged_text)
        return final_chunks


class SpacyTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at sentences using Spacy."""

    def __init__(self, pipeline: str = "zh_core_web_sm", **kwargs: Any) -> None:
        """Initialize the spacy text splitter."""
        super().__init__(**kwargs)
        try:
            import spacy
        except ImportError:
            raise ImportError("Spacy is not installed, please install it with `pip install spacy`.")
        try:
            self._tokenizer = spacy.load(pipeline)
        except:
            spacy.cli.download(pipeline)
            self._tokenizer = spacy.load(pipeline)

    def split_text(self, text: str, separator: Optional[str] = None, **kwargs) -> List[str]:
        """Split incoming text and return chunks."""
        splits = (str(s) for s in self._tokenizer(text).sents)
        return self._merge_splits(splits, separator, **kwargs)


class HeaderType(TypedDict):
    """Header type as typed dict."""

    level: int
    name: str
    data: str


class LineType(TypedDict):
    """Line type as typed dict."""

    metadata: Dict[str, str]
    content: str


class MarkdownHeaderTextSplitter(BaseComponent):
    """Implementation of splitting markdown files based on specified headers."""

    outgoing_edges = 1

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]] = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ],
        return_each_line: bool = False,
        filters: list = [],
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        separator="\n",
    ):
        """Create a new MarkdownHeaderTextSplitter.

        Args:
            headers_to_split_on: Headers we want to track
            return_each_line: Return each line w/ associated headers
        """
        # Output line-by-line or aggregated into chunks w/ common headers
        self.return_each_line = return_each_line
        self._chunk_size = chunk_size
        # Given the headers we want to split on,
        # (e.g., "#, ##, etc") order by length
        self.headers_to_split_on = sorted(headers_to_split_on, key=lambda split: len(split[0]), reverse=True)
        self._filter = filters
        self._length_function = length_function
        self._separator = separator
        self._chunk_overlap = chunk_overlap

    def aggregate_lines_to_chunks(self, lines: List[LineType]) -> List[dict]:
        """Combine lines with common metadata into chunks
        Args:
            lines: Line of text / associated header metadata
        """
        aggregated_chunks: List[LineType] = []

        for line in lines:
            if aggregated_chunks and aggregated_chunks[-1]["metadata"] == line["metadata"]:
                # If the last line in the aggregated list
                # has the same metadata as the current line,
                # append the current content to the last lines's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
            else:
                # Otherwise, append the current line to the aggregated list
                aggregated_chunks.append(line)

        return [{"page_content": chunk["content"], "metadata": chunk["metadata"]} for chunk in aggregated_chunks]

    def split_text(
        self,
        text: str,
        separator: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[dict]:
        """Split markdown file
        Args:
            text: Markdown file"""
        if separator is None:
            separator = self._separator
        if chunk_size is None:
            chunk_size = self._chunk_size
        if chunk_overlap is None:
            chunk_overlap = self._chunk_overlap

        # Split the input text by newline character ("\n").
        lines = text.split(separator)
        # Final output
        lines_with_metadata: List[LineType] = []
        # Content and metadata of the chunk currently being processed
        current_content: List[str] = []
        current_metadata: Dict[str, str] = {}
        # Keep track of the nested header structure
        # header_stack: List[Dict[str, Union[int, str]]] = []
        header_stack: List[HeaderType] = []
        initial_metadata: Dict[str, str] = {}

        for line in lines:
            stripped_line = line.strip()
            # Check each line against each of the header types (e.g., #, ##)
            for sep, name in self.headers_to_split_on:
                # Check if line starts with a header that we intend to split on
                if stripped_line.startswith(sep) and (
                    # Header with no text OR header is followed by space
                    # Both are valid conditions that sep is being used a header
                    len(stripped_line) == len(sep)
                    or stripped_line[len(sep)] == " "
                ):
                    # Ensure we are tracking the header as metadata
                    if name is not None:
                        # Get the current header level
                        current_header_level = sep.count("#")

                        # Pop out headers of lower or same level from the stack
                        while header_stack and header_stack[-1]["level"] >= current_header_level:
                            # We have encountered a new header
                            # at the same or higher level
                            popped_header = header_stack.pop()
                            # Clear the metadata for the
                            # popped header in initial_metadata
                            if popped_header["name"] in initial_metadata:
                                initial_metadata.pop(popped_header["name"])

                        # Push the current header to the stack
                        header: HeaderType = {
                            "level": current_header_level,
                            "name": name,
                            "data": stripped_line[len(sep) :].strip(),
                        }
                        header_stack.append(header)
                        # Update initial_metadata with the current header
                        initial_metadata[name] = header["data"]

                    # Add the previous line to the lines_with_metadata
                    # only if current_content is not empty
                    if current_content:
                        lines_with_metadata.append(
                            {
                                "content": separator.join(current_content),
                                "metadata": current_metadata.copy(),
                            }
                        )
                        current_content.clear()

                    break
            else:
                if stripped_line:
                    current_content.append(stripped_line)
                elif current_content:
                    lines_with_metadata.append(
                        {
                            "content": separator.join(current_content),
                            "metadata": current_metadata.copy(),
                        }
                    )
                    current_content.clear()

            current_metadata = initial_metadata.copy()
        if current_content:
            lines_with_metadata.append({"content": separator.join(current_content), "metadata": current_metadata})
        # lines_with_metadata has each line with associated header metadata
        # aggregate these into chunks based on common metadata
        if not self.return_each_line:
            return self._merge_splits(
                self.aggregate_lines_to_chunks(lines_with_metadata), separator, chunk_size, chunk_overlap
            )
        else:
            return self._merge_splits(
                [{"page_content": chunk["content"], "metadata": chunk["metadata"]} for chunk in lines_with_metadata],
                separator,
                chunk_size,
                chunk_overlap,
            )

    def clean(self, documents: List[dict], filters: Optional[List[str]] = None):
        if filters is None:
            filters = self._filter
        for special_character in filters:
            for doc in documents:
                doc["content"] = doc["content"].replace(special_character, "")
        return documents

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(
        self,
        documents: List[dict],
        separator: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: [int] = None,
    ) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        if separator is None:
            separator = self._separator
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for doc in documents:
            if doc["metadata"] != {}:
                head = sorted(doc["metadata"].items(), key=lambda x: x[0], reverse=True)[0][1]
                d = head + separator + doc["page_content"]
            else:
                d = doc["page_content"]
            _len = self._length_function(d)
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
                if total > chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, " f"which is longer than the specified {chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0) > chunk_size and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def run(
        self,
        documents: Union[dict, List[dict]],
        meta: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
        filters: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separator: Optional[str] = None,
    ):
        if filters is None:
            filters = self._filter
        if chunk_size is None:
            chunk_size = self._chunk_size
        if chunk_overlap is None:
            chunk_overlap = self._chunk_overlap
        if separator is None:
            separator = self._separator
        ret = []
        if type(documents) == list:
            for document in documents:
                text_splits = self.split_text(document["content"], separator, chunk_size, chunk_overlap)
                for i, txt in enumerate(text_splits):
                    doc = {}
                    doc["content"] = txt

                    if "meta" not in doc.keys() or doc["meta"] is None:
                        doc["meta"] = {}

                    doc["meta"]["_split_id"] = i
                    ret.append(doc)
        elif type(documents) == dict:
            text_splits = self.split_text(documents["content"], separator, chunk_size, chunk_overlap)
            for i, txt in enumerate(text_splits):
                doc = {}
                doc["content"] = txt

                if "meta" not in doc.keys() or doc["meta"] is None:
                    doc["meta"] = {}

                doc["meta"]["_split_id"] = i
                ret.append(doc)
        if filters is None:
            filters = self._filter
        if filters is not None and len(filters) > 0:
            ret = self.clean(ret, filters)
        result = {"documents": ret}
        return result, "output_1"

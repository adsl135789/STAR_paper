#!/usr/bin/env python3
"""
Generate synthetic questions for tables using K-means clustering and Mtq model.

This script combines semantic clustering with question generation:
1. Cluster rows using BGE-M3 embeddings and K-means
2. Generate cluster-level questions
3. Create corpus with representative rows and synthetic questions

Input format: Same as generate_synthetic_questions.py (JSONL with tables)
Output format: Same as generate_synthetic_questions.py (JSONL with representations)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiment_module.cycle_training.generation.sglang_client import SGLangClient
from experiment_module.cycle_training.prompt import T2Q_PROMPT_KMEANS, IMPROVED_T2Q_PROMPT
from embedding_model import EmbeddingModel, init_embedding_model


class KMeansQuestionGenerator:
    """Generate synthetic questions using K-means clustering on table rows."""

    def __init__(
        self,
        sglang_url: str,
        k_clusters: int,
        include_diverse: bool = True,
        max_instances_per_cluster: int = 10,
        use_header_augmentation: bool = True,
        header_weight: float = 0.2,
        max_concurrent: int = 32,
        timeout: float = 60.0,
        use_centroid_mode: bool = False,
    ):
        """
        Initialize generator.

        Args:
            sglang_url: URL of SGLang server running Mtq model
            k_clusters: Number of K-means clusters
            include_diverse: Whether to include diverse (farthest) instance per cluster
                           If True: select 2 rows per cluster (closest + farthest)
                           If False: select 1 row per cluster (closest only)
            max_instances_per_cluster: Maximum number of instances per cluster to include in prompt
                                      (default: 10) to avoid exceeding model context window
            use_header_augmentation: Whether to augment rows with header info for embedding
                                    If True: Use weighted fusion E_fused = λ*E_header + (1-λ)*E_row
                                    If False: Use raw CSV rows only
            header_weight: Weight (λ) for header embedding in fusion (default: 0.2)
                          E_fused = λ*E_header + (1-λ)*E_row
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
            use_centroid_mode: Whether to use centroid mode for question generation
                             If True: Generate k questions from header + k centroid instances (one per instance)
                             If False: Generate 1 question per cluster (original mode)
        """
        self.client = SGLangClient(
            base_url=sglang_url,
            max_concurrent=max_concurrent,
            timeout=timeout,
        )
        self.k_clusters = k_clusters
        self.include_diverse = include_diverse
        self.max_instances_per_cluster = max_instances_per_cluster
        self.use_header_augmentation = use_header_augmentation
        self.header_weight = header_weight
        self.use_centroid_mode = use_centroid_mode
        self.embedding_model = None
        logger.info(f"Initialized SGLang client: {sglang_url}")
        logger.info(f"K-means clusters: {k_clusters}")
        logger.info(f"Question generation mode: {'centroid' if use_centroid_mode else 'cluster-based'}")
        if not use_centroid_mode:
            logger.info(f"Include diverse instances: {include_diverse}")
            logger.info(f"Max instances per cluster: {max_instances_per_cluster}")
        logger.info(f"Use header augmentation: {use_header_augmentation}")
        if use_header_augmentation:
            logger.info(f"Header weight (λ): {header_weight}")

    def load_embedding_model(self):
        """Load BGE-M3 embedding model."""
        if self.embedding_model is None:
            logger.info("Loading BGE-M3 model...")
            self.embedding_model = init_embedding_model(model_name="bge_m3_flag", batch_size=32)
            logger.info("BGE-M3 model loaded successfully")

    def augment_row_with_header(self, row: str, header: List[str]) -> str:
        """
        Augment a row with header information.

        Args:
            row: CSV-formatted row string
            header: List of header strings (supports multi-row headers)

        Returns:
            Header-augmented string
        """
        row_values = [v.strip() for v in row.split(',')]
        parsed_headers = []
        for header_row in header:
            parsed_headers.append([v.strip() for v in header_row.split(',')])

        pairs = []
        for i, value in enumerate(row_values):
            column_headers = []
            for header_row in parsed_headers:
                if i < len(header_row) and header_row[i]:
                    column_headers.append(header_row[i])

            if column_headers:
                combined_header = "/".join(column_headers)
                pairs.append(f"{combined_header}: {value}")
            else:
                pairs.append(value)

        return "; ".join(pairs)

    def get_header_embedding(self, header: List[str]) -> np.ndarray:
        """
        Generate embedding for the header.
        
        Creates a header-aware sentence like:
        "Country: ; Population: ; GDP: ; Region: "
        
        This captures the semantic meaning of column names.
        
        Args:
            header: List of header strings
            
        Returns:
            Header embedding vector
        """
        # Parse header to get column names
        if isinstance(header, list):
            if len(header) == 1:
                header_cols = [col.strip() for col in header[0].split(',')]
            else:
                # Multi-row header: combine all levels
                all_cols = []
                for header_row in header:
                    cols = [col.strip() for col in header_row.split(',')]
                    all_cols.append(cols)
                # Create hierarchical header
                header_cols = []
                for i in range(len(all_cols[0])):
                    col_parts = []
                    for row_cols in all_cols:
                        if i < len(row_cols) and row_cols[i]:
                            col_parts.append(row_cols[i])
                    header_cols.append("/".join(col_parts) if col_parts else "")
        else:
            header_cols = [col.strip() for col in str(header).split(',')]
        
        # Create header sentence (column names only)
        header_sentence = "; ".join([f"{col}" for col in header_cols if col])
        
        # Encode header
        result = self.embedding_model.encode([header_sentence])
        header_embedding = np.array(result.dense_vecs)[0]
        
        return header_embedding

    def embed_instances(self, instances: List[str], header: List[str]) -> np.ndarray:
        """
        Generate BGE-M3 embeddings for instances with header-aware fusion.
        
        Implements the weighted fusion approach:
        E_fused = λ * E_header + (1 - λ) * E_row
        
        Where:
        - E_header: Embedding of header (column names semantic)
        - E_row: Embedding of raw row values
        - λ (lambda): Header weight (default 0.2)
        
        This approach:
        1. Helps K-means understand column semantics
        2. Prevents all rows from being too similar due to same header
        3. Balances structural info (header) with content info (values)

        Args:
            instances: List of CSV-formatted instance strings
            header: List of header strings

        Returns:
            Numpy array of shape (n_instances, embedding_dim)
        """
        if not instances:
            return np.array([])

        if self.use_header_augmentation:
            # Header-Aware Fusion approach
            logger.debug(f"Using header-aware fusion with λ={self.header_weight}")
            
            # Step 1: Get header embedding (E_header)
            header_embedding = self.get_header_embedding(header)
            
            # Step 2: Get raw row embeddings (E_row)
            result = self.embedding_model.encode(instances)
            row_embeddings = np.array(result.dense_vecs)
            
            # Step 3: Fused embedding = λ * E_header + (1 - λ) * E_row
            lambda_weight = self.header_weight
            fused_embeddings = (
                lambda_weight * header_embedding[np.newaxis, :] +  # Broadcast header to all rows
                (1 - lambda_weight) * row_embeddings
            )
            
            logger.debug(f"Created {len(fused_embeddings)} fused embeddings (header weight={lambda_weight})")
            return fused_embeddings
        else:
            # Use raw CSV rows without header augmentation
            result = self.embedding_model.encode(instances)
            embeddings = np.array(result.dense_vecs)
            return embeddings

    def cluster_and_select_rows(
        self,
        instances: List[str],
        header: List[str]
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Cluster instances and select representative rows.

        Step 1: Semantic Clustering
        - Use BGE-M3 to embed each row (with or without header augmentation based on setting)
        - K-means clustering into k groups
        - Select rows per cluster based on include_diverse setting:
          * If include_diverse=True: 2 rows (closest + farthest) → 2k rows total
          * If include_diverse=False: 1 row (closest only) → k rows total

        Args:
            instances: List of CSV-formatted row strings
            header: Table header

        Returns:
            Tuple of (selected_rows, cluster_assignments)
            - selected_rows: List of selected representative rows (k or 2k rows)
            - cluster_assignments: List of lists containing row indices per cluster
        """
        n_instances = len(instances)

        # If fewer instances than clusters, return all instances
        if n_instances <= self.k_clusters:
            logger.debug(f"Table has {n_instances} instances <= k={self.k_clusters}, using all instances")
            return instances, [[i] for i in range(n_instances)]

        # Embed all instances
        embeddings = self.embed_instances(instances, header)

        # K-means clustering
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        # Select rows per cluster
        selected_rows = []
        cluster_assignments = [[] for _ in range(self.k_clusters)]

        for cluster_id in range(self.k_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_assignments[cluster_id] = cluster_indices.tolist()

            if len(cluster_indices) == 0:
                continue

            cluster_embeddings = embeddings[cluster_indices]
            centroid = centroids[cluster_id]

            # 1. Always select closest to centroid (most representative)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_rows.append(instances[closest_idx])

            # 2. Optionally select most diverse (farthest from centroid)
            if self.include_diverse:
                if len(cluster_indices) > 1:
                    farthest_idx = cluster_indices[np.argmax(distances)]
                    selected_rows.append(instances[farthest_idx])
                else:
                    # Only one instance in cluster, duplicate it if include_diverse=True
                    selected_rows.append(instances[closest_idx])

        num_per_cluster = 2 if self.include_diverse else 1
        logger.debug(
            f"Selected {len(selected_rows)} rows ({num_per_cluster} per cluster) "
            f"from {n_instances} instances across {self.k_clusters} clusters"
        )
        return selected_rows, cluster_assignments

    def select_centroid_instances(
        self,
        instances: List[str],
        header: List[str]
    ) -> List[str]:
        """
        Select k centroid instances (closest to each cluster center).

        This method is used in centroid mode to select representative instances
        for generating questions.

        Args:
            instances: List of CSV-formatted row strings
            header: Table header

        Returns:
            List of k centroid instances (one per cluster)
        """
        n_instances = len(instances)

        # If fewer instances than clusters, return all instances
        if n_instances <= self.k_clusters:
            logger.debug(f"Table has {n_instances} instances <= k={self.k_clusters}, using all instances")
            return instances

        # Embed all instances
        embeddings = self.embed_instances(instances, header)

        # K-means clustering
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        # Select centroid instance per cluster (closest to centroid)
        centroid_instances = []
        for cluster_id in range(self.k_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_embeddings = embeddings[cluster_indices]
            centroid = centroids[cluster_id]

            # Select closest to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            centroid_instances.append(instances[closest_idx])

        logger.debug(
            f"Selected {len(centroid_instances)} centroid instances "
            f"from {n_instances} instances across {self.k_clusters} clusters"
        )
        return centroid_instances

    def table_to_csv_text(self, header: List[str], instances: List[str]) -> str:
        """
        Convert table to CSV text representation.

        Args:
            header: Table header
            instances: Table instances

        Returns:
            CSV text string
        """
        lines = []

        # Add header
        if header:
            if isinstance(header, list):
                if len(header) == 1:
                    header_text = header[0]
                else:
                    header_text = ",".join(str(col) for col in header)
            else:
                header_text = str(header)
            lines.append(header_text)

        # Add instances
        for instance in instances:
            lines.append(str(instance))

        return "\n".join(lines)

    async def generate_cluster_questions(
        self,
        header: List[str],
        cluster_instances: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        lang: str = "en",
    ) -> List[str]:
        """
        Generate 1 question per cluster.

        Step 2: Cluster-level Question Generation
        - Input cluster rows + header to question generation model
        - Limit cluster instances to max_instances_per_cluster to avoid context overflow
        - Generate 1 synthetic question per cluster (total k questions)

        Args:
            header: Table header
            cluster_instances: All instances in the cluster (will be limited to max_instances_per_cluster)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            lang: Language code for question generation

        Returns:
            List containing 1 generated question
        """
        # Limit cluster instances to avoid exceeding context window
        if len(cluster_instances) > self.max_instances_per_cluster:
            logger.debug(
                f"Cluster has {len(cluster_instances)} instances, "
                f"limiting to {self.max_instances_per_cluster} for prompt"
            )
            cluster_instances = cluster_instances[:self.max_instances_per_cluster]

        # Create table text with header and cluster instances
        table_text = self.table_to_csv_text(header, cluster_instances)

        # Format prompt - use the same T2Q_PROMPT as original, but request 1 question
        prompt = T2Q_PROMPT_KMEANS.format(text=table_text, lang=lang)
        # Update the prompt to generate only 1 question (instead of 5)

        # Generate using SGLang client (same pattern as original)
        try:
            results = await self.client.generate_batch(
                prompts=[prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            result = results[0]

            if not result.success:
                logger.error(f"Failed to generate question for cluster: {result.error}")
                return []

            # Extract questions using the same method as original
            questions = self._extract_questions(result.text)
            return questions[:1]  # Return only 1 question

        except Exception as e:
            logger.error(f"Failed to generate question for cluster: {e}")
            return []

    async def generate_centroid_questions(
        self,
        header: List[str],
        centroid_instances: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        lang: str = "en",
        max_retries: int = 10,
    ) -> List[str]:
        """
        Generate k questions from header + k centroid instances (one question per instance).

        Centroid Mode Question Generation:
        - Input: header + k centroid instances (one per cluster)
        - Use IMPROVED_T2Q_PROMPT to generate k questions (one per instance)
        - Each question focuses on the specific content of its corresponding row
        - Cumulative generation: if fewer than k questions are generated, keep them
          and generate only the remaining needed questions in the next round

        Args:
            header: Table header
            centroid_instances: List of k centroid instances (one per cluster)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            lang: Language code for question generation
            max_retries: Maximum retry attempts if k questions are not generated

        Returns:
            List of exactly k generated questions (one per instance), or partial list if failed
        """
        num_questions = len(centroid_instances)
        accumulated_questions = []
        used_indices = set()  # Track which instances have been used

        # Retry until we get exactly k questions
        for attempt in range(max_retries):
            # Calculate how many more questions we need
            remaining_needed = num_questions - len(accumulated_questions)

            if remaining_needed <= 0:
                # We have enough questions
                logger.debug(f"Successfully accumulated {num_questions} questions")
                return accumulated_questions[:num_questions]

            # Get unused instances
            available_indices = [i for i in range(num_questions) if i not in used_indices]
            available_instances = [centroid_instances[i] for i in available_indices]

            if not available_instances:
                # All instances used but still not enough questions - this shouldn't happen
                logger.error(f"All instances used but only {len(accumulated_questions)}/{num_questions} questions generated")
                break

            # Create table text with header and available instances
            table_text = self.table_to_csv_text(header, available_instances)

            # Format prompt
            prompt = IMPROVED_T2Q_PROMPT.format(
                text=table_text,
                lang=lang,
                num_questions=len(available_instances)
            )

            try:
                results = await self.client.generate_batch(
                    prompts=[prompt],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                result = results[0]

                if not result.success:
                    logger.warning(f"Generation failed (attempt {attempt + 1}/{max_retries}): {result.error}")
                    continue

                # Extract questions
                new_questions = self._extract_questions(result.text)

                if not new_questions:
                    logger.warning(f"No questions extracted (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(0.5)
                    continue

                # Add new questions and mark indices as used
                # We assume questions are generated in order matching the instances
                questions_to_add = min(len(new_questions), len(available_instances))
                for i in range(questions_to_add):
                    accumulated_questions.append(new_questions[i])
                    used_indices.add(available_indices[i])

                logger.info(
                    f"Accumulated {len(accumulated_questions)}/{num_questions} questions "
                    f"(+{questions_to_add} this round, attempt {attempt + 1}/{max_retries})"
                )

                # Check if we're done
                if len(accumulated_questions) >= num_questions:
                    logger.debug(f"Successfully generated {num_questions} questions across {attempt + 1} attempts")
                    return accumulated_questions[:num_questions]

                # Add small delay before next retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error during generation (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                continue

        # Reached max retries - return what we have
        if accumulated_questions:
            logger.warning(
                f"Only generated {len(accumulated_questions)}/{num_questions} questions "
                f"after {max_retries} attempts, returning partial results"
            )
            return accumulated_questions
        else:
            logger.error(f"Failed to generate any questions after {max_retries} attempts")
            return []

    def _extract_questions(self, generated_text: str) -> List[str]:
        """
        Extract questions from generated text with robust parsing.

        This uses the same extraction logic as the original generate_synthetic_questions.py
        to maintain consistency.

        Expected format:
        ```json
        {
            "questions": ["question1", "question2", ...]
        }
        ```

        Args:
            generated_text: Generated text from model

        Returns:
            List of extracted questions (empty list on failure)
        """
        import re

        if not generated_text or not generated_text.strip():
            logger.warning("Empty generated text")
            return []

        try:
            # Strategy 1: Extract JSON block (```json ... ```)
            json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(json_block_pattern, generated_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    questions = data.get("questions", [])
                    if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                        return [q.strip() for q in questions if q.strip()]
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error in strategy 1: {e}")

            # Strategy 2: Extract first complete JSON object
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            match = re.search(json_pattern, generated_text, re.DOTALL)
            if match:
                try:
                    # Clean up JSON string (remove trailing comments)
                    json_str = match.group(0).split('#')[0].split('//')[0].strip()
                    data = json.loads(json_str)
                    questions = data.get("questions", [])
                    if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                        return [q.strip() for q in questions if q.strip()]
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error in strategy 2: {e}")

            # Strategy 3: Try to fix common JSON errors
            json_match = re.search(r'\{.*["\']questions["\'].*\[.*\].*\}', generated_text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    # Fix trailing commas
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    # Fix single quotes to double quotes
                    json_str = json_str.replace("'", '"')
                    data = json.loads(json_str)
                    questions = data.get("questions", [])
                    if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                        return [q.strip() for q in questions if q.strip()]
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error in strategy 3: {e}")

            # Strategy 4: Extract questions from array pattern directly
            array_pattern = r'"questions"\s*:\s*\[(.*?)\]'
            match = re.search(array_pattern, generated_text, re.DOTALL)
            if match:
                try:
                    array_content = match.group(1)
                    # Extract quoted strings
                    question_pattern = r'"([^"]+)"'
                    questions = re.findall(question_pattern, array_content)
                    if questions:
                        return [q.strip() for q in questions if q.strip()]
                except Exception as e:
                    logger.debug(f"Error in strategy 4: {e}")

            # If all strategies fail, log the full text
            logger.warning(f"Failed to extract questions. Generated text:\n{generated_text[:500]}...")
            return []

        except Exception as e:
            logger.error(f"Unexpected error parsing questions: {e}")
            logger.debug(f"Generated text:\n{generated_text[:500]}...")
            return []

    def create_representation(
        self,
        header: List[str],
        selected_rows: List[str],
        synthetic_questions: List[str],
        mode: str,
    ) -> str:
        """
        Create representation based on mode.

        Step 3: Corpus Construction
        - Combine partial table with k synthetic questions
        - Number of rows: k (if include_diverse=False) or 2k (if include_diverse=True)

        Args:
            header: Table header
            selected_rows: Selected representative rows (k or 2k rows)
            synthetic_questions: Generated questions (k questions)
            mode: 'header_only' or 'header_and_instance'

        Returns:
            Representation string
        """
        # Parse header
        if isinstance(header, list):
            if len(header) == 1:
                header_str = header[0]
            else:
                header_str = ",".join(str(col) for col in header)
        else:
            header_str = str(header)

        # Combine questions
        questions_text = "\n".join(f"Q: {q}" for q in synthetic_questions)

        if mode == "header_only":
            return f"{header_str}\n{questions_text}"
        else:  # header_and_instance
            table_text = self.table_to_csv_text(header, selected_rows)
            return f"{table_text}\n{questions_text}"

    async def process_single_table(
        self,
        table: Dict[str, Any],
        mode: str,
        max_new_tokens: int,
        temperature: float,
        max_retries: int,
        lang: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single table: cluster rows and generate questions.

        Two modes:
        1. Cluster mode (use_centroid_mode=False):
           - Generate 1 question per cluster (k questions total)
        2. Centroid mode (use_centroid_mode=True):
           - Generate k questions from header + k centroid instances
           - Each question focuses on one specific centroid instance

        Args:
            table: Table dictionary
            mode: Representation mode
            max_new_tokens: Max tokens per generation
            temperature: Sampling temperature
            max_retries: Max retry attempts
            lang: Language code

        Returns:
            Result dictionary or None if failed
        """
        header = table.get('header', [])
        instances = table.get('instances', [])
        original_id = table.get('id', '')
        file_name = table.get('file_name', '')
        sheet_name = table.get('sheet_name', '')

        if not instances:
            logger.warning(f"Table {original_id} has no instances, skipping")
            return None

        if self.use_centroid_mode:
            # Centroid Mode: Generate k questions from header + k centroid instances
            # Step 1: Select k centroid instances
            centroid_instances = self.select_centroid_instances(instances, header)

            # Step 2: Generate k questions from header + centroid instances
            # The generate_centroid_questions method has internal retry logic
            all_questions = await self.generate_centroid_questions(
                header=header,
                centroid_instances=centroid_instances,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                lang=lang,
                max_retries=max_retries
            )

            if not all_questions:
                logger.warning(f"No questions generated for table {original_id} in centroid mode")
                return None

            # Use centroid instances as selected rows
            selected_rows = centroid_instances

        else:
            # Cluster Mode: Generate 1 question per cluster
            # Step 1: Cluster and select representative rows
            selected_rows, cluster_assignments = self.cluster_and_select_rows(instances, header)

            # Step 2: Generate 1 question per cluster
            all_questions = []
            for cluster_id, cluster_indices in enumerate(cluster_assignments):
                if len(cluster_indices) == 0:
                    continue

                # Get all instances in this cluster
                cluster_instances = [instances[idx] for idx in cluster_indices]

                # Generate 1 question for this cluster
                for attempt in range(max_retries):
                    try:
                        questions = await self.generate_cluster_questions(
                            header=header,
                            cluster_instances=cluster_instances,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            lang=lang
                        )
                        if questions:
                            all_questions.extend(questions)
                            break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to generate question for cluster {cluster_id} after {max_retries} attempts: {e}")
                        else:
                            await asyncio.sleep(1)

            if not all_questions:
                logger.warning(f"No questions generated for table {original_id}")
                return None

        # Step 3: Create representation (partial table + questions)
        representation = self.create_representation(
            header=header,
            selected_rows=selected_rows,
            synthetic_questions=all_questions,
            mode=mode,
        )

        # Create table_contents with selected rows
        table_contents = self.table_to_csv_text(header, selected_rows)

        # Create output record
        record = {
            "id": original_id,
            "representation": representation,
            "table_contents": table_contents,
            "metadata": {
                "file_name": file_name,
                "sheet_name": sheet_name,
                "header": header,
                "instances": instances,
                "k_clusters": self.k_clusters,
                "use_centroid_mode": self.use_centroid_mode,
                "include_diverse": self.include_diverse if not self.use_centroid_mode else None,
                "max_instances_per_cluster": self.max_instances_per_cluster if not self.use_centroid_mode else None,
                "use_header_augmentation": self.use_header_augmentation,
                "header_weight": self.header_weight if self.use_header_augmentation else None,
                "num_selected_rows": len(selected_rows),
                "num_original_rows": len(instances),
            },
            "synthetic_questions": all_questions,
        }

        return record

    async def generate_for_dataset(
        self,
        input_file: str,
        output_dir: str,
        mode: str = "header_and_instance",
        batch_size: int = 64,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        max_retries: int = 10,
        lang="en"
    ):
        """
        Generate synthetic questions for entire dataset using K-means clustering.

        Args:
            input_file: Path to input JSONL file
            output_dir: Output directory for results
            mode: Representation mode
            batch_size: Processing batch size
            max_new_tokens: Max tokens per generation
            temperature: Sampling temperature
            max_retries: Max retry attempts
        """
        # Load embedding model
        self.load_embedding_model()

        # Load dataset
        logger.info(f"Loading dataset from {input_file}")
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(data)} tables from dataset")

        if not data:
            logger.warning("No data to process")
            return

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Determine output filename
        input_stem = Path(input_file).stem
        mode_suffix = "centroid" if self.use_centroid_mode else "cluster"
        output_file = Path(output_dir) / f"{input_stem}_kmeans_k{self.k_clusters}_{mode_suffix}.jsonl"

        logger.info(f"Processing {len(data)} tables...")
        logger.info(f"Mode: {mode}")
        logger.info(f"K-means clusters: {self.k_clusters}")
        logger.info(f"Question generation mode: {mode_suffix}")
        logger.info(f"Batch size: {batch_size}")

        results = []
        with tqdm(total=len(data), desc="Generating questions") as pbar:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]

                # Process batch concurrently
                tasks = [
                    self.process_single_table(
                        table=table,
                        mode=mode,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        max_retries=max_retries,
                        lang=lang
                    )
                    for table in batch
                ]

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter valid results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing error: {result}")
                    elif result is not None:
                        results.append(result)

                pbar.update(len(batch))

        # Save results
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # Summary
        total_questions = sum(len(r["synthetic_questions"]) for r in results)
        avg_questions = total_questions / len(results) if results else 0
        total_selected_rows = sum(r["metadata"]["num_selected_rows"] for r in results)
        total_original_rows = sum(r["metadata"]["num_original_rows"] for r in results)
        reduction_pct = (1 - total_selected_rows / total_original_rows) * 100 if total_original_rows > 0 else 0

        logger.info(f"Generated {total_questions} questions for {len(results)} tables")
        logger.info(f"Average: {avg_questions:.2f} questions per table")
        logger.info(f"Row reduction: {total_original_rows} → {total_selected_rows} ({reduction_pct:.1f}% reduction)")
        logger.info(f"Output saved to: {output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic questions using K-means clustering and Mtq model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input JSONL file containing tables"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for generated questions"
    )
    parser.add_argument(
        "--k_clusters",
        type=int,
        required=True,
        help="Number of K-means clusters (k)"
    )
    parser.add_argument(
        "--include_diverse",
        action="store_true",
        default=True,
        help="Include diverse (farthest) instance per cluster. "
             "If set, select 2 rows per cluster (closest + farthest). "
             "If not set, select 1 row per cluster (closest only)."
    )
    parser.add_argument(
        "--no_diverse",
        dest="include_diverse",
        action="store_false",
        help="Do not include diverse instances (select only closest to centroid)"
    )
    parser.add_argument(
        "--max_instances_per_cluster",
        type=int,
        default=10,
        help="Maximum number of instances per cluster to include in prompt "
             "(default: 10) to avoid exceeding model context window"
    )
    parser.add_argument(
        "--use_header_augmentation",
        action="store_true",
        default=True,
        help="Use header-aware fusion for embedding: E_fused = λ*E_header + (1-λ)*E_row"
    )
    parser.add_argument(
        "--no_header_augmentation",
        dest="use_header_augmentation",
        action="store_false",
        help="Do not augment rows with header (use raw CSV for embedding)"
    )
    parser.add_argument(
        "--header_weight",
        type=float,
        default=0.2,
        help="Weight (λ) for header embedding in fusion (default: 0.2). "
             "E_fused = λ*E_header + (1-λ)*E_row. Range: 0.0-1.0"
    )
    parser.add_argument(
        "--use_centroid_mode",
        action="store_true",
        default=False,
        help="Use centroid mode for question generation. "
             "If set, generate k questions from header + k centroid instances (one per instance). "
             "If not set (default), generate 1 question per cluster (cluster mode)."
    )
    parser.add_argument(
        "--mode",
        choices=["header_only", "header_and_instance"],
        default="header_and_instance",
        help="Representation mode"
    )
    parser.add_argument(
        "--sglang_url",
        default="http://localhost:8002",
        help="SGLang server URL"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=128,
        help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0 for greedy)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=500.0,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=15,
        help="Maximum retry attempts for failed generations"
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Validate k_clusters
    if args.k_clusters < 1:
        logger.error(f"k_clusters must be at least 1 (got {args.k_clusters})")
        sys.exit(1)

    # Create generator
    generator = KMeansQuestionGenerator(
        sglang_url=args.sglang_url,
        k_clusters=args.k_clusters,
        include_diverse=args.include_diverse,
        max_instances_per_cluster=args.max_instances_per_cluster,
        use_header_augmentation=args.use_header_augmentation,
        header_weight=args.header_weight,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        use_centroid_mode=args.use_centroid_mode,
    )
    dataset_name = Path(args.output_dir).stem
    logger.info(f"current dataset: {dataset_name}")
    if dataset_name == "mimo_ch":
        lang = "zh"
    else:
        lang = "en"
        
    # Generate questions
    await generator.generate_for_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        mode=args.mode,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        lang=lang
    )

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())

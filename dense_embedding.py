#!/usr/bin/env python3
"""Build dense embeddings for table corpora"""

import argparse
import json
import sys
from pathlib import Path
from pymilvus import MilvusClient
from loguru import logger
from embedding_model import init_embedding_model
import numpy as np
import torch
from experiment_module.QAF.attention_module import AttentionFusionModule
from experiment_module.QAF.config import QAFusionConfig

def load_jsonl(file_path):
    """Load JSONL file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    logger.info(f"Loaded {len(data)} documents from {file_path}")
    return data

def build_embeddings(jsonl_path, db_path, model_name="bge_m3_flag", model_path=None, gpu_id=None, batch_size=32, dimension=1024, force=False, mode="representation", table_weight=0.5, question_weight_min=0.1, question_weight_max=0.5, diversity_alpha=0.3, coverage_beta=2.0, attention_module_path=None, attention_config_path=None):
    """Build embeddings for single JSONL file

    Args:
        jsonl_path: Path to JSONL file
        db_path: Path to output database
        model_name: Name of embedding model
        model_path: Path to model weights
        gpu_id: GPU ID to use
        batch_size: Batch size for embedding
        dimension: Embedding dimension
        force: Force rebuild if database exists
        mode: Embedding mode - "representation", "fusion", "dynamic_fusion", "diversity_fusion", "dynamic_concat", "dynamic_concat_v2", "concat_fusion", or "attention_fusion"
            - "representation": Embed item["representation"] directly
            - "fusion": Embed item["table_contents"] and each item["synthetic_questions"], then weighted average
            - "dynamic_fusion": Cosine-based semantic attention - questions weighted by similarity to table
            - "diversity_fusion": Question diversity weighting - diverse questions get higher weights
            - "dynamic_concat": Concatenate questions, dynamic weight based on table-questions similarity
            - "dynamic_concat_v2": Concatenate questions, dynamic weight adjusted by intra-question coherence
            - "concat_fusion": Embed table separately, concatenate all questions then embed, weighted fusion
            - "attention_fusion": Use trained attention module to fuse table and question embeddings
        table_weight: Weight ratio for table_contents (default: 0.5, meaning 50% of total weight, used in fusion and concat_fusion modes, ignored in dynamic_fusion, diversity_fusion and attention_fusion modes)
        question_weight_min: Minimum weight for questions in dynamic_fusion, dynamic_concat, and dynamic_concat_v2 modes (default: 0.1)
        question_weight_max: Maximum weight for questions in dynamic_fusion, dynamic_concat, and dynamic_concat_v2 modes (default: 0.5)
        diversity_alpha: Diversity importance factor for dynamic_concat_v2 mode (default: 0.3)
        coverage_beta: Soft weighting temperature for semantic coverage in dynamic_concat_v2 mode (default: 2.0)
        attention_module_path: Path to trained attention module weights (required for attention_fusion mode)
        attention_config_path: Path to attention module config (required for attention_fusion mode)
    """
    if db_path.exists():
        if not force:
            logger.warning(f"Database {db_path} exists, skipping")
            return
        else:
            logger.warning(f"Database {db_path} exists, removing...")
            import shutil
            if db_path.is_dir():
                shutil.rmtree(db_path)
                logger.info(f"Removed old database directory: {db_path}")
            elif db_path.is_file():
                db_path.unlink()
                logger.info(f"Removed old database file: {db_path}")
    
    # Load data
    data = load_jsonl(jsonl_path)
    if not data:
        return
    
    # Initialize model
    embedding_model = init_embedding_model(model_name, model_path, gpu_id, batch_size)
    
    # Generate embeddings based on mode
    if mode == "representation":
        texts = [item["representation"] for item in data]
        logger.info(f"Generating embeddings for {len(texts)} documents (representation mode)")
        embeddings = embedding_model.encode(texts).dense_vecs
    
    elif mode == "fusion":
        question_weight = 1 - table_weight
        logger.info(f"Generating embeddings for {len(data)} documents (fusion mode)")
        logger.info(f"Weight distribution - table: {table_weight:.2f}, questions: {question_weight:.2f}")
        embeddings_list = []
        
        for item in data:
            # Collect all texts to embed: table_contents + synthetic_questions
            texts_to_embed = []
            weights = []
            
            # Add table_contents
            if "table_contents" in item:
                texts_to_embed.append(item["table_contents"])
                weights.append(table_weight)
            else:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'table_contents', skipping")
                continue
            
            # Add each synthetic_question
            num_questions = 0
            if "synthetic_questions" in item and isinstance(item["synthetic_questions"], list):
                num_questions = len(item["synthetic_questions"])
                if num_questions > 0:
                    # Distribute question_weight evenly among all questions
                    per_question_weight = question_weight / num_questions
                    for question in item["synthetic_questions"]:
                        texts_to_embed.append(question)
                        weights.append(per_question_weight)
                else:
                    logger.warning(f"Item {item.get('id', 'unknown')} has empty 'synthetic_questions', using only table_contents")
            else:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'synthetic_questions', using only table_contents")
            
            # Encode all texts
            if texts_to_embed:
                item_embeddings = embedding_model.encode(texts_to_embed).dense_vecs
                
                # Weighted average
                weights_array = np.array(weights).reshape(-1, 1)
                weighted_embeddings = item_embeddings * weights_array
                avg_embedding = np.sum(weighted_embeddings, axis=0) / np.sum(weights_array)
                
                embeddings_list.append(avg_embedding)
            else:
                logger.error(f"Item {item.get('id', 'unknown')} has no texts to embed")
                # Use zero vector as fallback
                embeddings_list.append(np.zeros(dimension))
        
        embeddings = np.array(embeddings_list)
    
    elif mode == "dynamic_fusion":
        logger.info(f"Generating embeddings for {len(data)} documents (dynamic_fusion mode)")
        logger.info("Using Cosine-based Semantic Attention with dynamic table-question fusion")
        logger.info(f"Temperature τ=0.07, question weight range: [{question_weight_min:.2f}, {question_weight_max:.2f}]")
        embeddings_list = []

        # Temperature parameter for softmax
        tau = 0.07

        # Weight range for questions
        weight_range = question_weight_max - question_weight_min

        for item in data:
            # Add table_contents
            if "table_contents" not in item:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'table_contents', skipping")
                embeddings_list.append(np.zeros(dimension))
                continue

            table_text = item["table_contents"]

            # Add synthetic_questions
            questions = []
            if "synthetic_questions" in item and isinstance(item["synthetic_questions"], list):
                questions = item["synthetic_questions"]
                if len(questions) == 0:
                    logger.warning(f"Item {item.get('id', 'unknown')} has empty 'synthetic_questions', using only table_contents")
            else:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'synthetic_questions', using only table_contents")

            # Encode table
            table_embedding = embedding_model.encode([table_text]).dense_vecs[0]

            # If no questions, use table embedding only
            if len(questions) == 0:
                embeddings_list.append(table_embedding)
                continue

            # Encode questions
            question_embeddings = embedding_model.encode(questions).dense_vecs

            # Calculate cosine similarity between table and each question
            # Normalize vectors
            table_norm = table_embedding / (np.linalg.norm(table_embedding) + 1e-10)
            question_norms = question_embeddings / (np.linalg.norm(question_embeddings, axis=1, keepdims=True) + 1e-10)

            # Compute cosine similarities: cos(E_t, q_i)
            similarities = np.dot(question_norms, table_norm)

            # Apply softmax with temperature: α_i = exp(cos(E_t, q_i)/τ) / Σ_j exp(cos(E_t, q_j)/τ)
            # Use numerically stable softmax by subtracting max value
            scaled_similarities = similarities / tau
            max_sim = np.max(scaled_similarities)
            exp_similarities = np.exp(scaled_similarities - max_sim)
            attention_weights = exp_similarities / np.sum(exp_similarities)

            # Compute weighted sum of question embeddings: E_q = Σ_i α_i * q_i
            weighted_question_embedding = np.sum(question_embeddings * attention_weights.reshape(-1, 1), axis=0)

            # Dynamic weight based on average similarity
            # Higher similarity → give more weight to questions
            avg_similarity = np.mean(similarities)
            # Map similarity [0, 1] to question_weight [question_weight_min, question_weight_max]
            # When similarity is high (close to 1), questions get more weight
            question_weight = question_weight_min + weight_range * avg_similarity
            table_weight = 1.0 - question_weight

            # Final embedding: weighted combination of table and questions
            final_embedding = table_weight * table_embedding + question_weight * weighted_question_embedding

            embeddings_list.append(final_embedding)

            # Log for first few items
            if len(embeddings_list) <= 3:
                logger.debug(f"Item {item.get('id', 'unknown')}: "
                           f"num_questions={len(questions)}, "
                           f"avg_similarity={avg_similarity:.3f}, "
                           f"table_weight={table_weight:.3f}, "
                           f"question_weight={question_weight:.3f}")

        embeddings = np.array(embeddings_list)

    elif mode == "diversity_fusion":
        logger.info(f"Generating embeddings for {len(data)} documents (diversity_fusion mode)")
        logger.info("Using Question Diversity Weighting with dynamic table-question fusion")
        embeddings_list = []

        for item in data:
            # Add table_contents
            if "table_contents" not in item:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'table_contents', skipping")
                embeddings_list.append(np.zeros(dimension))
                continue

            table_text = item["table_contents"]

            # Add synthetic_questions
            questions = []
            if "synthetic_questions" in item and isinstance(item["synthetic_questions"], list):
                questions = item["synthetic_questions"]
                if len(questions) == 0:
                    logger.warning(f"Item {item.get('id', 'unknown')} has empty 'synthetic_questions', using only table_contents")
            else:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'synthetic_questions', using only table_contents")

            # Encode table
            table_embedding = embedding_model.encode([table_text]).dense_vecs[0]

            # If no questions, use table embedding only
            if len(questions) == 0:
                embeddings_list.append(table_embedding)
                continue

            # If only one question, use equal weight (diversity not applicable)
            if len(questions) == 1:
                question_embedding = embedding_model.encode(questions).dense_vecs[0]
                # Use 50-50 weight for single question case
                final_embedding = 0.5 * table_embedding + 0.5 * question_embedding
                embeddings_list.append(final_embedding)
                continue

            # Encode questions
            question_embeddings = embedding_model.encode(questions).dense_vecs

            # Normalize question embeddings for cosine similarity calculation
            question_norms = question_embeddings / (np.linalg.norm(question_embeddings, axis=1, keepdims=True) + 1e-10)

            # Calculate diversity score for each question
            # d_i = (1/(N-1)) * Σ_{j≠i} (1 - cos(q_i, q_j))
            N = len(questions)
            diversity_scores = np.zeros(N)

            for i in range(N):
                # Compute cosine similarity between q_i and all other questions
                similarities = np.dot(question_norms, question_norms[i])
                # Remove self-similarity
                similarities[i] = 0
                # Calculate average distance (1 - similarity)
                distances = 1 - similarities
                diversity_scores[i] = np.sum(distances) / (N - 1)

            # Apply softmax to diversity scores (numerically stable)
            max_score = np.max(diversity_scores)
            exp_scores = np.exp(diversity_scores - max_score)
            diversity_weights = exp_scores / np.sum(exp_scores)

            # Compute weighted sum of question embeddings: E_q = Σ_i α_i * q_i
            weighted_question_embedding = np.sum(question_embeddings * diversity_weights.reshape(-1, 1), axis=0)

            # Dynamic weight based on table-question similarity
            table_norm = table_embedding / (np.linalg.norm(table_embedding) + 1e-10)
            question_similarities = np.dot(question_norms, table_norm)
            avg_similarity = np.mean(question_similarities)

            # Map similarity [0, 1] to question_weight [0.1, 0.5]
            question_weight = 0.1 + 0.4 * avg_similarity
            table_weight = 1.0 - question_weight

            # Final embedding: weighted combination of table and questions
            final_embedding = table_weight * table_embedding + question_weight * weighted_question_embedding

            embeddings_list.append(final_embedding)

            # Log for first few items
            if len(embeddings_list) <= 3:
                logger.debug(f"Item {item.get('id', 'unknown')}: "
                           f"num_questions={len(questions)}, "
                           f"diversity_scores={diversity_scores[:3]}, "
                           f"diversity_weights={diversity_weights[:3]}, "
                           f"avg_similarity={avg_similarity:.3f}, "
                           f"table_weight={table_weight:.3f}, "
                           f"question_weight={question_weight:.3f}")

        embeddings = np.array(embeddings_list)

    elif mode == "dynamic_concat":
        logger.info(f"Generating embeddings for {len(data)} documents (dynamic_concat mode)")
        logger.info("Table embedding: separate, Questions embedding: concatenated")
        logger.info(f"Dynamic weight based on table-questions similarity, range: [{question_weight_min:.2f}, {question_weight_max:.2f}]")
        embeddings_list = []

        # Weight range for questions
        weight_range = question_weight_max - question_weight_min

        for item in data:
            # Check table_contents
            if "table_contents" not in item:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'table_contents', skipping")
                embeddings_list.append(np.zeros(dimension))
                continue

            table_text = item["table_contents"]

            # Get synthetic_questions
            questions = []
            if "synthetic_questions" in item and isinstance(item["synthetic_questions"], list):
                questions = item["synthetic_questions"]
                if len(questions) == 0:
                    logger.warning(f"Item {item.get('id', 'unknown')} has empty 'synthetic_questions', using only table_contents")
            else:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'synthetic_questions', using only table_contents")

            # Encode table separately
            table_embedding = embedding_model.encode([table_text]).dense_vecs[0]

            # If no questions, use table embedding only
            if len(questions) == 0:
                embeddings_list.append(table_embedding)
                continue

            # Concatenate all questions with separator
            concatenated_questions = " ".join(questions)

            # Encode concatenated questions
            questions_embedding = embedding_model.encode([concatenated_questions]).dense_vecs[0]

            # Calculate cosine similarity between table and concatenated questions
            # Normalize vectors
            table_norm = table_embedding / (np.linalg.norm(table_embedding) + 1e-10)
            questions_norm = questions_embedding / (np.linalg.norm(questions_embedding) + 1e-10)

            # Compute cosine similarity
            similarity = np.dot(table_norm, questions_norm)

            # Dynamic weight based on similarity
            # Map similarity [0, 1] to question_weight [question_weight_min, question_weight_max]
            question_weight = question_weight_min + weight_range * similarity
            table_weight = 1.0 - question_weight

            # Weighted fusion
            final_embedding = table_weight * table_embedding + question_weight * questions_embedding

            embeddings_list.append(final_embedding)

            # Log for first few items
            if len(embeddings_list) <= 3:
                logger.debug(f"Item {item.get('id', 'unknown')}: "
                           f"num_questions={len(questions)}, "
                           f"concat_length={len(concatenated_questions)}, "
                           f"similarity={similarity:.3f}, "
                           f"table_weight={table_weight:.3f}, "
                           f"question_weight={question_weight:.3f}")

        embeddings = np.array(embeddings_list)

    elif mode == "dynamic_concat_v2":
        logger.info(f"Generating embeddings for {len(data)} documents (dynamic_concat_v2 mode)")
        logger.info("Table embedding: separate, Questions embedding: concatenated")
        logger.info("Using Diversity-based Dynamic Weighting with Semantic Coverage")
        logger.info(f"Question weight range: [{question_weight_min:.2f}, {question_weight_max:.2f}]")
        logger.info(f"Diversity alpha: {diversity_alpha:.2f}, Coverage beta: {coverage_beta:.2f}")
        embeddings_list = []

        for item in data:
            # Check table_contents
            if "table_contents" not in item:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'table_contents', skipping")
                embeddings_list.append(np.zeros(dimension))
                continue

            table_text = item["table_contents"]

            # Get synthetic_questions
            questions = []
            if "synthetic_questions" in item and isinstance(item["synthetic_questions"], list):
                questions = item["synthetic_questions"]
                if len(questions) == 0:
                    logger.warning(f"Item {item.get('id', 'unknown')} has empty 'synthetic_questions', using only table_contents")
            else:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'synthetic_questions', using only table_contents")

            # Encode table separately
            table_embedding = embedding_model.encode([table_text]).dense_vecs[0]

            # If no questions, use table embedding only
            if len(questions) == 0:
                embeddings_list.append(table_embedding)
                continue

            # If only one question, cannot compute coherence, fallback to simple weighted average
            if len(questions) == 1:
                concatenated_questions = questions[0]
                questions_embedding = embedding_model.encode([concatenated_questions]).dense_vecs[0]
                # Use 50-50 weight for single question case
                final_embedding = 0.5 * table_embedding + 0.5 * questions_embedding
                embeddings_list.append(final_embedding)
                continue

            # Concatenate all questions with separator
            concatenated_questions = " ".join(questions)

            # Encode concatenated questions
            questions_embedding = embedding_model.encode([concatenated_questions]).dense_vecs[0]

            # Encode individual questions for diversity and coverage calculation
            individual_question_embeddings = embedding_model.encode(questions).dense_vecs
            N = len(questions)

            # Normalize embeddings for cosine similarity
            table_norm = table_embedding / (np.linalg.norm(table_embedding) + 1e-10)
            question_norms = individual_question_embeddings / (np.linalg.norm(individual_question_embeddings, axis=1, keepdims=True) + 1e-10)

            # Step 1: Calculate coherence (c) - average pairwise similarity between questions
            # c = (2 / (N(N-1))) * Σ_{i<j} cos(E_qi, E_qj)
            coherence_sum = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    coherence_sum += np.dot(question_norms[i], question_norms[j])
            coherence = (2.0 / (N * (N - 1))) * coherence_sum

            # Step 2: Calculate diversity (v) - inverse of coherence
            # v = 1 - c
            # Higher diversity means questions are more semantically diverse
            diversity = 1.0 - coherence

            # Step 3: Calculate table-question similarities for each question
            # s_i = cos(E_t, E_qi)
            similarities = np.dot(question_norms, table_norm)

            # Step 4: Soft weighting - focus on questions with similarity near the mean
            # w_i = exp(β(1 - |s_i - s_mean|)) / Σ_j exp(β(1 - |s_j - s_mean|))
            s_mean = np.mean(similarities)
            deviations = np.abs(similarities - s_mean)
            soft_weights_unnorm = np.exp(coverage_beta * (1.0 - deviations))
            soft_weights = soft_weights_unnorm / np.sum(soft_weights_unnorm)

            # Calculate semantic coverage (r) - weighted average similarity
            # r = (1/N) * Σ_i s_i * w_i
            semantic_coverage = np.sum(similarities * soft_weights)

            # Step 5: Calculate final question weight with diversity boost
            # w_q = ((1 + r) / 2) * (1 + α * v)
            # First term: table-questions relevance
            # Second term: diversity contribution
            base_weight = (1.0 + semantic_coverage) / 2.0
            diversity_boost = 1.0 + diversity_alpha * diversity
            raw_question_weight = base_weight * diversity_boost

            # Map to [question_weight_min, question_weight_max]
            # Assuming raw_question_weight ∈ [0, ~1.5], map to target range
            # Normalize by expected max value (when r=1, v=1): (1+1)/2 * (1+α*1) = 1+α
            max_possible = 1.0 + diversity_alpha
            normalized = raw_question_weight / max_possible
            weight_range = question_weight_max - question_weight_min
            question_weight = question_weight_min + normalized * weight_range

            # Ensure within bounds
            question_weight = np.clip(question_weight, question_weight_min, question_weight_max)
            table_weight = 1.0 - question_weight

            # Weighted fusion
            final_embedding = table_weight * table_embedding + question_weight * questions_embedding

            embeddings_list.append(final_embedding)

            # Log for first few items
            if len(embeddings_list) <= 3:
                logger.debug(f"Item {item.get('id', 'unknown')}: "
                           f"num_questions={len(questions)}, "
                           f"coherence={coherence:.3f}, "
                           f"diversity={diversity:.3f}, "
                           f"s_mean={s_mean:.3f}, "
                           f"semantic_coverage={semantic_coverage:.3f}, "
                           f"base_weight={base_weight:.3f}, "
                           f"diversity_boost={diversity_boost:.3f}, "
                           f"question_weight={question_weight:.3f}")

        embeddings = np.array(embeddings_list)

    elif mode == "concat_fusion":
        question_weight = 1 - table_weight
        logger.info(f"Generating embeddings for {len(data)} documents (concat_fusion mode)")
        logger.info(f"Table embedding: separate, Questions embedding: concatenated")
        logger.info(f"Weight distribution - table: {table_weight:.2f}, questions: {question_weight:.2f}")
        embeddings_list = []

        for item in data:
            # Check table_contents
            if "table_contents" not in item:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'table_contents', skipping")
                embeddings_list.append(np.zeros(dimension))
                continue

            table_text = item["table_contents"]

            # Get synthetic_questions
            questions = []
            if "synthetic_questions" in item and isinstance(item["synthetic_questions"], list):
                questions = item["synthetic_questions"]
                if len(questions) == 0:
                    logger.warning(f"Item {item.get('id', 'unknown')} has empty 'synthetic_questions', using only table_contents")
            else:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'synthetic_questions', using only table_contents")

            # Encode table separately
            table_embedding = embedding_model.encode([table_text]).dense_vecs[0]

            # If no questions, use table embedding only
            if len(questions) == 0:
                embeddings_list.append(table_embedding)
                continue

            # Concatenate all questions with separator
            concatenated_questions = " ".join(questions)

            # Encode concatenated questions
            questions_embedding = embedding_model.encode([concatenated_questions]).dense_vecs[0]

            # Weighted fusion
            final_embedding = table_weight * table_embedding + question_weight * questions_embedding

            embeddings_list.append(final_embedding)

            # Log for first few items
            if len(embeddings_list) <= 3:
                logger.debug(f"Item {item.get('id', 'unknown')}: "
                           f"num_questions={len(questions)}, "
                           f"concat_length={len(concatenated_questions)}, "
                           f"table_weight={table_weight:.3f}, "
                           f"question_weight={question_weight:.3f}")

        embeddings = np.array(embeddings_list)

    elif mode == "attention_fusion":
        # Validate required arguments
        if attention_module_path is None or attention_config_path is None:
            raise ValueError("attention_fusion mode requires both --attention-module-path and --attention-config-path")

        logger.info(f"Generating embeddings for {len(data)} documents (attention_fusion mode)")
        logger.info(f"Loading attention module from {attention_module_path}")
        logger.info(f"Loading config from {attention_config_path}")

        # Load config
        config = QAFusionConfig.load(attention_config_path)

        # Initialize attention module with config settings
        attention_module = AttentionFusionModule(
            input_dim=1024,
            d_k=config.model.attention.d_k,
            d_v=config.model.attention.d_v,
            dropout=config.model.attention.dropout_rate,
            no_value_projection=config.model.attention.no_value_projection,
        )

        # Load trained weights with strict=False to handle checkpoint compatibility
        state_dict = torch.load(attention_module_path, map_location='cpu')
        attention_module.load_state_dict(state_dict, strict=False)

        # Log if there are missing or unexpected keys
        model_keys = set(attention_module.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint (randomly initialized): {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint (ignored): {unexpected_keys}")

        # Move to GPU if available
        device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')
        attention_module.to(device)
        attention_module.eval()

        logger.info(f"Attention module loaded on device: {device}")

        embeddings_list = []
        max_questions = config.model.max_synthetic_questions

        for item in data:
            # Collect texts to embed
            if "table_contents" not in item:
                logger.warning(f"Item {item.get('id', 'unknown')} missing 'table_contents', skipping")
                embeddings_list.append(np.zeros(dimension))
                continue

            table_text = item["table_contents"]

            # Get synthetic questions
            questions = []
            if "synthetic_questions" in item and isinstance(item["synthetic_questions"], list):
                questions = item["synthetic_questions"][:max_questions]

            if len(questions) == 0:
                logger.warning(f"Item {item.get('id', 'unknown')} has no synthetic_questions, using table_contents only")
                # Fallback to table embedding only
                table_emb = embedding_model.encode([table_text]).dense_vecs[0]
                embeddings_list.append(table_emb)
                continue

            # Encode table and questions
            table_emb = embedding_model.encode([table_text]).dense_vecs[0]
            question_embs = embedding_model.encode(questions).dense_vecs

            # Convert to torch tensors
            table_tensor = torch.from_numpy(table_emb).float().unsqueeze(0).to(device)  # (1, 1024)
            question_tensor = torch.from_numpy(question_embs).float().unsqueeze(0).to(device)  # (1, num_questions, 1024)

            # Apply attention fusion
            with torch.no_grad():
                fused_emb = attention_module(table_tensor, question_tensor)  # (1, 1024)

            # Convert back to numpy
            fused_emb_np = fused_emb.cpu().numpy()[0]
            embeddings_list.append(fused_emb_np)

        embeddings = np.array(embeddings_list)
        logger.info(f"Generated {len(embeddings)} fused embeddings using trained attention module")

    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'representation', 'fusion', 'dynamic_fusion', 'diversity_fusion', 'dynamic_concat', 'dynamic_concat_v2', 'concat_fusion', or 'attention_fusion'")

    # Create database
    client = MilvusClient(str(db_path))
    actual_dimension = embeddings.shape[1] if len(embeddings) > 0 else dimension
    client.create_collection(
        collection_name="corpus_dense",
        dimension=actual_dimension
    )
    
    # Prepare insert data
    insert_data = []
    for i, item in enumerate(data):
        # Handle different data formats (chunked vs non-chunked)

        item_id = item['id']
        metadata = item.get("metadata")

        insert_data.append({
            "id": item_id,
            "vector": embeddings[i].astype('float32').tolist(),
            "representation": item["representation"],
            "metadata": metadata
        })
    
    # Insert data
    logger.info("Inserting data to database")
    client.insert(collection_name="corpus_dense", data=insert_data)
    logger.success(f"Saved {len(insert_data)} records to {db_path}")

def process_directory(input_dir, output_dir, model_name, model_path, gpu_id, batch_size, dimension, force, mode, table_weight, question_weight_min, question_weight_max, diversity_alpha, coverage_beta, attention_module_path, attention_config_path):
    """Process all JSONL files in directory"""
    jsonl_files = list(Path(input_dir).rglob("*.jsonl"))

    if not jsonl_files:
        logger.warning("No JSONL files found")
        return 0

    successful = 0
    for i, file_path in enumerate(jsonl_files, 1):
        logger.info(f"Processing {i}/{len(jsonl_files)}: {file_path.name}")

        db_path = Path(output_dir) / f"{file_path.stem}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            build_embeddings(file_path, db_path, model_name, model_path, gpu_id, batch_size, dimension, force, mode, table_weight, question_weight_min, question_weight_max, diversity_alpha, coverage_beta, attention_module_path, attention_config_path)
            successful += 1
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    logger.info(f"Processed {successful}/{len(jsonl_files)} files")
    return successful

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='JSONL file or directory')
    parser.add_argument('--output', required=True, help='Output database path')
    parser.add_argument('--model', default="bge_m3_flag", help='Embedding model')
    parser.add_argument('--model-path', help='Model path (required for self_train model)')
    parser.add_argument('--gpu-id', type=int, help='GPU ID')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding model (default: 32)')
    parser.add_argument('--dimension', type=int, default=1024, help='Embedding dimension (default: 1024)')
    parser.add_argument('--force', action='store_true', help='Force rebuild (default: False)')  # 修改這裡
    parser.add_argument('--mode', default="representation", choices=["representation", "fusion", "dynamic_fusion", "diversity_fusion", "dynamic_concat", "dynamic_concat_v2", "concat_fusion", "attention_fusion"],
                        help='Embedding mode: representation (default), fusion, dynamic_fusion, diversity_fusion, dynamic_concat, dynamic_concat_v2, concat_fusion, or attention_fusion')
    parser.add_argument('--table-weight', type=float, default=0.5,
                        help='Weight ratio for table_contents in fusion and concat_fusion modes (default: 0.5, ignored in dynamic_fusion, diversity_fusion and attention_fusion)')
    parser.add_argument('--question-weight-min', type=float, default=0.1,
                        help='Minimum weight for questions in dynamic_fusion, dynamic_concat, and dynamic_concat_v2 modes (default: 0.1)')
    parser.add_argument('--question-weight-max', type=float, default=0.5,
                        help='Maximum weight for questions in dynamic_fusion, dynamic_concat, and dynamic_concat_v2 modes (default: 0.5)')
    parser.add_argument('--diversity-alpha', type=float, default=0.3,
                        help='Diversity importance factor for dynamic_concat_v2 mode (default: 0.3)')
    parser.add_argument('--coverage-beta', type=float, default=4.0,
                        help='Soft weighting temperature for semantic coverage in dynamic_concat_v2 mode (default: 2.0)')
    parser.add_argument('--attention-module-path', help='Path to trained attention module weights (required for attention_fusion mode)')
    parser.add_argument('--attention-config-path', help='Path to attention module config (required for attention_fusion mode)')

    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output path: {output_path}")
    try:
        if input_path.is_file():
            build_embeddings(input_path, output_path, args.model, args.model_path, args.gpu_id, args.batch_size, args.dimension, args.force, args.mode, args.table_weight, args.question_weight_min, args.question_weight_max, args.diversity_alpha, args.coverage_beta, args.attention_module_path, args.attention_config_path)
        elif input_path.is_dir():
            if process_directory(input_path, output_path, args.model, args.model_path, args.gpu_id, args.batch_size, args.dimension, args.force, args.mode, args.table_weight, args.question_weight_min, args.question_weight_max, args.diversity_alpha, args.coverage_beta, args.attention_module_path, args.attention_config_path) == 0:
                sys.exit(1)
        else:
            logger.error(f"Invalid input: {input_path}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
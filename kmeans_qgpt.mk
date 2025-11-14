# K-means + QGPT experiment targets (with semantic clustering)
.PHONY: kmeans_cluster_qgpt_data kmeans_cluster_qgpt_embedding kmeans_cluster_qgpt_evaluation kmeans_cluster_qgpt_experiment

# Configuration
K_CLUSTERS ?= 5
INCLUDE_DIVERSE ?= true
MAX_INSTANCES_PER_CLUSTER ?= 10
USE_HEADER_AUGMENTATION ?= true
KMEANS_QGPT_TEST_DATASETS := e2ewtq feta ottqa mimo_en mimo_ch
# KMEANS_QGPT_TEST_DATASETS := e2ewtq  
KMEANS_QGPT_GPU_ID := 2

# Determine corpus path based on include_diverse setting
ifeq ($(INCLUDE_DIVERSE),false)
    DIVERSE_SUFFIX := _no_diverse
    DIVERSE_FLAG := --no_diverse
else
    DIVERSE_SUFFIX :=
    DIVERSE_FLAG :=
endif

# Determine header augmentation flag
ifeq ($(USE_HEADER_AUGMENTATION),false)
    HEADER_AUG_FLAG := --no_header_augmentation
	HEADER_SUFFIX := _no_HA
else
    HEADER_AUG_FLAG :=
	HEADER_SUFFIX := 
endif

KMEANS_CORPUS_PATH := corpus/kmeans_cluster_qgpt_k$(K_CLUSTERS)$(DIVERSE_SUFFIX)$(HEADER_SUFFIX)

# Generate synthetic questions using K-means clustering (semantic clustering approach)
kmeans_cluster_qgpt_data:
	@echo "Generating K-means QGPT data with k=$(K_CLUSTERS) clusters (include_diverse=$(INCLUDE_DIVERSE), max_instances=$(MAX_INSTANCES_PER_CLUSTER), header_aug=$(USE_HEADER_AUGMENTATION))..."
	uv run -m experiment_module.cycle_training.scripts.generate_synthetic_questions_with_kmeans \
	    --input_file dataset/test/mimo_ch/table.jsonl \
	    --output_dir $(KMEANS_CORPUS_PATH)/test/mimo_ch \
	    --k_clusters $(K_CLUSTERS) \
	    $(DIVERSE_FLAG) \
	    $(HEADER_AUG_FLAG) \
	    $(HEADER_AUG_FLAG) \
	    --max_instances_per_cluster $(MAX_INSTANCES_PER_CLUSTER) \
	    --sglang_url http://localhost:8002 \
	    --temperature 0.4 \
	    --batch_size 64
	uv run -m experiment_module.cycle_training.scripts.generate_synthetic_questions_with_kmeans \
	    --input_file dataset/test/mimo_en/table.jsonl \
	    --output_dir $(KMEANS_CORPUS_PATH)/test/mimo_en \
	    --k_clusters $(K_CLUSTERS) \
	    $(DIVERSE_FLAG) \
	    $(HEADER_AUG_FLAG) \
	    --max_instances_per_cluster $(MAX_INSTANCES_PER_CLUSTER) \
	    --sglang_url http://localhost:8002 \
	    --temperature 0.4 \
	    --batch_size 100
	uv run -m experiment_module.cycle_training.scripts.generate_synthetic_questions_with_kmeans \
	    --input_file dataset/test/ottqa/table.jsonl \
	    --output_dir $(KMEANS_CORPUS_PATH)/test/ottqa \
	    --k_clusters $(K_CLUSTERS) \
	    $(DIVERSE_FLAG) \
	    $(HEADER_AUG_FLAG) \
	    --max_instances_per_cluster $(MAX_INSTANCES_PER_CLUSTER) \
	    --sglang_url http://localhost:8002 \
	    --temperature 0.4 \
	    --batch_size 100
	uv run -m experiment_module.cycle_training.scripts.generate_synthetic_questions_with_kmeans \
	    --input_file dataset/test/feta/table.jsonl \
	    --output_dir $(KMEANS_CORPUS_PATH)/test/feta \
	    --k_clusters $(K_CLUSTERS) \
	    $(DIVERSE_FLAG) \
	    $(HEADER_AUG_FLAG) \
	    --max_instances_per_cluster $(MAX_INSTANCES_PER_CLUSTER) \
	    --sglang_url http://localhost:8002 \
	    --temperature 0.4 \
	    --batch_size 100
# 	uv run -m experiment_module.cycle_training.scripts.generate_synthetic_questions_with_kmeans \
# 	    --input_file dataset/test/e2ewtq/table.jsonl \
# 	    --output_dir $(KMEANS_CORPUS_PATH)/test/e2ewtq \
# 	    --k_clusters $(K_CLUSTERS) \
# 	    $(DIVERSE_FLAG) \
# 	    $(HEADER_AUG_FLAG) \
# 	    --max_instances_per_cluster $(MAX_INSTANCES_PER_CLUSTER) \
# 	    --sglang_url http://localhost:8002 \
# 	    --temperature 0.4 \
# 	    --batch_size 100

# Generate synthetic questions for training data
kmeans_cluster_qgpt_data_train:
	@echo "Generating K-means cluster QGPT training data with k=$(K_CLUSTERS) clusters (include_diverse=$(INCLUDE_DIVERSE), max_instances=$(MAX_INSTANCES_PER_CLUSTER))..."
	uv run -m experiment_module.cycle_training.scripts.generate_synthetic_questions_with_kmeans \
	    --input_file dataset/train/ottqa/table.jsonl \
	    --output_dir $(KMEANS_CORPUS_PATH)/train/ottqa \
	    --k_clusters $(K_CLUSTERS) \
	    $(DIVERSE_FLAG) \
	    $(HEADER_AUG_FLAG) \
	    --max_instances_per_cluster $(MAX_INSTANCES_PER_CLUSTER) \
	    --sglang_url http://localhost:8002 \
	    --temperature 0.4 \
	    --batch_size 100
	uv run -m experiment_module.cycle_training.scripts.generate_synthetic_questions_with_kmeans \
	    --input_file dataset/train/feta/table.jsonl \
	    --output_dir $(KMEANS_CORPUS_PATH)/train/feta \
	    --k_clusters $(K_CLUSTERS) \
	    $(DIVERSE_FLAG) \
	    $(HEADER_AUG_FLAG) \
	    --max_instances_per_cluster $(MAX_INSTANCES_PER_CLUSTER) \
	    --sglang_url http://localhost:8002 \
	    --temperature 0.4 \
	    --batch_size 100

# Generate embeddings for K-means cluster QGPT corpus
kmeans_cluster_qgpt_embedding:
	@echo "Generating embeddings for K-means cluster QGPT corpus with k=$(K_CLUSTERS) (include_diverse=$(INCLUDE_DIVERSE))..."
	@for dataset in $(KMEANS_QGPT_TEST_DATASETS); do \
		echo "Processing dataset: $$dataset"; \
		uv run python dense_embedding.py \
			--input $(KMEANS_CORPUS_PATH)/test/$$dataset \
			--output db/kmeans_cluster_qgpt_k$(K_CLUSTERS)$(DIVERSE_SUFFIX)/$(EMBEDDING_MODE)/$$dataset \
			--mode $(EMBEDDING_MODE) \
			--table-weight 0.7 \
			--gpu-id $(KMEANS_QGPT_GPU_ID) \
			--batch-size 128 \
			--force; \
	done

# Run evaluation for K-means cluster QGPT
kmeans_cluster_qgpt_evaluation:
	@echo "Evaluating K-means cluster QGPT with k=$(K_CLUSTERS) (include_diverse=$(INCLUDE_DIVERSE))..."
	@for dataset in $(KMEANS_QGPT_TEST_DATASETS); do \
		echo "Evaluating dataset: $$dataset"; \
		uv run python dense_evaluator.py \
			--test-file dataset/test/$$dataset/query.jsonl \
			--db-folder db/kmeans_cluster_qgpt_k$(K_CLUSTERS)$(DIVERSE_SUFFIX)/$(EMBEDDING_MODE)/$$dataset/ \
			--output-folder experiment/kmeans_cluster_qgpt_k$(K_CLUSTERS)$(DIVERSE_SUFFIX)/$(EMBEDDING_MODE)/$$dataset \
			--batch-size 256 \
			--gpu-id $(KMEANS_QGPT_GPU_ID); \
	done

# Run complete K-means cluster QGPT experiment
kmeans_cluster_qgpt_experiment: kmeans_cluster_qgpt_data kmeans_cluster_qgpt_embedding kmeans_cluster_qgpt_evaluation
	@echo "K-means cluster QGPT experiment completed with k=$(K_CLUSTERS) (include_diverse=$(INCLUDE_DIVERSE))!"

kmeans_cluster_qgpt_embedding_evaluation: kmeans_cluster_qgpt_embedding kmeans_cluster_qgpt_evaluation
	@echo "K-means cluster QGPT experiment completed with k=$(K_CLUSTERS) (include_diverse=$(INCLUDE_DIVERSE))!"




# Convenience targets for different k values and diverse settings
.PHONY: kmeans_cluster_qgpt_k3 kmeans_cluster_qgpt_k5 kmeans_cluster_qgpt_k10
.PHONY: kmeans_cluster_qgpt_k5_no_diverse kmeans_cluster_qgpt_k10_no_diverse

kmeans_cluster_qgpt_k3:
	$(MAKE) kmeans_cluster_qgpt_experiment K_CLUSTERS=3 EMBEDDING_MODE=representation

kmeans_cluster_qgpt_k5:
	$(MAKE) kmeans_cluster_qgpt_experiment K_CLUSTERS=5 EMBEDDING_MODE=representation

kmeans_cluster_qgpt_k10:
	$(MAKE) kmeans_cluster_qgpt_experiment K_CLUSTERS=10 EMBEDDING_MODE=representation

# Targets without diverse instances (only closest to centroid)
kmeans_cluster_qgpt_k5_no_diverse:
	$(MAKE) kmeans_cluster_qgpt_experiment K_CLUSTERS=5 INCLUDE_DIVERSE=false EMBEDDING_MODE=representation

kmeans_cluster_qgpt_k10_no_diverse:
	$(MAKE) kmeans_cluster_qgpt_experiment K_CLUSTERS=10 INCLUDE_DIVERSE=false EMBEDDING_MODE=representation
	
kmeans_cluster_qgpt_k10_no_diverse_no_HA:
	$(MAKE) kmeans_cluster_qgpt_experiment K_CLUSTERS=10 INCLUDE_DIVERSE=false EMBEDDING_MODE=representation USE_HEADER_AUGMENTATION=false

kmeans_cluster_embedding_evaluation:
	$(MAKE) kmeans_cluster_qgpt_evaluation K_CLUSTERS=5 INCLUDE_DIVERSE=true EMBEDDING_MODE=representation

kmeans_cluster_qgpt_k10_no_diverse_embedding_evaluation:
	$(MAKE) kmeans_cluster_qgpt_embedding_evaluation K_CLUSTERS=10 INCLUDE_DIVERSE=false EMBEDDING_MODE=representation
	$(MAKE) kmeans_cluster_qgpt_embedding_evaluation K_CLUSTERS=10 INCLUDE_DIVERSE=false EMBEDDING_MODE=fusion
	$(MAKE) kmeans_cluster_qgpt_embedding_evaluation K_CLUSTERS=10 INCLUDE_DIVERSE=false EMBEDDING_MODE=dynamic_fusion
	$(MAKE) kmeans_cluster_qgpt_embedding_evaluation K_CLUSTERS=10 INCLUDE_DIVERSE=false EMBEDDING_MODE=diversity_fusion
	
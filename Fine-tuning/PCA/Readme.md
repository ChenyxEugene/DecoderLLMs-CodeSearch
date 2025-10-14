# PCA Visualization of Code & Query Embeddings
This tool visualizes the distribution of code and query embeddings to compare the effects of model fine-tuning. 

The process is completed in two main steps:

Generate Embeddings: Use a base model and a fine-tuned model to generate embedding vectors for code-query pairs.

Visualize with PCA: Read the generated embedding files and use PCA to create 2D comparison plots.

## ⚙️ Requirements
You need Python 3. Install all required libraries with the following command:

```
pip install numpy pandas seaborn matplotlib scikit-learn torch tqdm transformers peft more-itertools llm2vec
```
Note: For GPU acceleration, ensure you install a version of PyTorch that is compatible with your local CUDA environment.

## 📂 Directory & Data Structure
Before running the scripts, organize your files as shown below. The scripts will automatically create a cache/ subdirectory in your specified output directory to store the intermediate embedding files.
```
/path/to/your/project/
├── PCA_result/               <-- Your --output_dir
│   └── cache/                    <-- Auto-created to store embeddings
│       ├── embeds_original_queries.npy
│       ├── embeds_original_codes.npy
│       ├── embeds_finetuned_queries.npy
│       └── embeds_finetuned_codes.npy
│
└── data/
    └── random_samples.jsonl      <-- random samples file --samples_file
```
## 🚀 Workflow
Please follow these two steps in order.

### Step 1: Generate Embeddings
This step uses the generate_embeddings.py script to convert code and query text into vector representations.

Example Command:
```
python generate_embeddings.py \
    --base_model_path /path/to/your/base/codegemma-7b-it \
    --finetuned_model_path /path/to/your/lora/checkpoint-1000 \
    --samples_file /path/to/your/data/selected_samples.jsonl \
    --output_dir /path/to/your/project/final_analysis \
    --batch_size 16
```

### Step 2: PCA Visualization
After generating the embeddings, run the pca_visualization.py script to create the comparison plots.

Example Command:
python pca_visualization.py \
    --samples_file /path/to/your/data/random_samples.jsonl \
    --output_dir /path/to/your/project/PCA_result

**Arguments**:
--samples_file: (Required) The path to the .jsonl file containing the metadata for the samples to be plotted.
--output_dir: (Required) The path to the directory where the output plots will be saved. The script expects to find the embedding files in a cache/ subdirectory here.

**Output**:
The script generates two PDF files in the specified --output_dir:
figure_pca_before_finetuning_<strategy>.pdf: Visualization of the original embeddings.
figure_pca_after_finetuning_<strategy>.pdf: Visualization of the fine-tuned embeddings.
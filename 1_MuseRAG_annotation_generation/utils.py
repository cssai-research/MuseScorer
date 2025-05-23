import os
import pickle
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import ollama
import pandas as pd
from tqdm.notebook import tqdm
import re
import ast
import copy



class OllamaAgent:
    def __init__(self, model_name, object_name,comparison_k,prompt_method='CoT'):
        """
        Initialize the Ollama agent with a model and a fixed system prompt.

        Args:
            model_name (str): Name of the Ollama model (e.g., 'llama3', 'mistral')
            system_prompt (str): Fixed system prompt for the agent
        """
        self.model_name = model_name
        self.system_prompt = self.system_prompt_generator(object_name,comparison_k,prompt_method)
        self.client = ollama.Client()  # create a persistent client connection

    def system_prompt_generator(self, object_name, comparison_k,prompt_method='CoT'):
        if prompt_method=='CoT':
            sys_prompt = f"""You are an idea bucket annotator for ideas generated for the object {object_name} in Guilford's Alternative Uses Test. You will be given an input_idea to annotate against upto {comparison_k+1} comparison_ideas, given to you in a dictionary format with key-value pairs of comparison_idea_ID:comparison_idea_description. The keys are integers, and the values are strings. Your goal is to determine if the input_idea is a very obviously rephrased version of one of those comparison_idea_description, or if it is slightly different.
                    if input_idea is a very obviously rephrased version of a certain comparison_idea_description:
                        your_annotation_ID = comparison_idea_ID key of that comparison_idea_description value
                    elif input_idea is a slightly different one:
                        your_annotation_ID = -1
                    You will also provide a reason string containing a single sentence explaining why you gave the input_idea that specific your_annotation_ID.
                    Your response must be a text string containing exactly: <your_annotation_ID><SPACE><reason>
                    For example, if your_annotation_ID is 6 and the reason is "The input idea is a very obviously rephrased version of comparison_idea_ID 6", your response string should be "6 The input idea is a very obviously rephrased version of comparison_idea_ID 6". 
                    Another example: if your_annotation_ID is -1 and the reason is "The input idea is not an obvious rephrasing of any comparison_idea_ID", your response string should be "-1 The input idea is not an obvious rephrasing of any comparison_idea_ID". 
                    Absolutely do not provide any extra text."""
        elif prompt_method == 'baseline':
            sys_prompt = f"""You are an idea bucket annotator for ideas generated for the object {object_name} in Guilford's Alternative Uses Test. You will be given an input_idea to annotate against upto {comparison_k+1} comparison_ideas, given to you in a dictionary format with key-value pairs of comparison_idea_ID:comparison_idea_description. The keys are integers, and the values are strings. Your goal is to determine if the input_idea is a very obviously rephrased version of one of those comparison_idea_description, or if it is slightly different.
                    if input_idea is a very obviously rephrased version of a certain comparison_idea_description:
                        your_annotation_ID = comparison_idea_ID key of that comparison_idea_description value
                    elif input_idea is a slightly different one:
                        your_annotation_ID = -1
                    Your response must be a text string containing exactly: <your_annotation_ID>
                    For example, if your_annotation_ID is 6 since the input idea is a very obviously rephrased version of comparison_idea_ID 6, your response string should be "6". 
                    Another example: if your_annotation_ID is -1 because the input idea is not an obvious rephrasing of any comparison_idea_ID, your response string should be "-1". 
                    Absolutely do not provide any extra text."""
        return sys_prompt

    def query(self, user_input):
        """
        Query the Ollama model with the fixed system prompt and fresh user input.
        Stateless: no past conversation memory is kept.

        Args:
            user_input (str): User's query

        Returns:
            str: Model's response
        """
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response['message']['content']
    

def save_experiment_database(embeddings, codebook, annotated_ideas, exp_arg):
    # Get experiment name
    experiment_name = exp_arg["experiment_name"]
    
    # Create folder if not exist
    os.makedirs("databases", exist_ok=True)
    save_folder = os.path.join("databases", experiment_name)
    os.makedirs(save_folder, exist_ok=True)
    
    # Define file paths
    embeddings_path = os.path.join(save_folder, f"{experiment_name}_embeddings.npy")
    codebook_path = os.path.join(save_folder, f"{experiment_name}_codebook.pkl")
    annotated_ideas_path = os.path.join(save_folder, f"{experiment_name}_annotated_ideas.pkl")
    
    # Save embeddings
    np.save(embeddings_path, embeddings)
    
    # Save metadata (texts, IDs, etc.) of codebook
    with open(codebook_path, "wb") as f:
        pickle.dump(codebook, f)

    # Save metadata (texts, IDs, etc.) of annotations
    with open(annotated_ideas_path, "wb") as f:
        pickle.dump(annotated_ideas, f)
    
    # print(f"✅ Saved embeddings to: {embeddings_path}")
    # print(f"✅ Saved codebook to: {codebook_path}")
    # print(f"✅ Saved annotated_ideas to: {annotated_ideas_path}")

    
def export_annotations(experiment_name):
    """
    Loads metadata from an experiment and exports it to a CSV file.
    
    Args:
        exp_arg (dict): Dictionary with experiment_name key.
        output_folder (str): Where to save the exported CSVs.
        
    Returns:
        csv_path (str): Path to the exported CSV.
    """
    database_folder = os.path.join("databases", experiment_name)
    codebook_ids_descriptions_path = os.path.join(database_folder, f"{experiment_name}_codebook_ids_descriptions.pkl")
    annotated_ideas_path = os.path.join(database_folder, f"{experiment_name}_annotated_ideas.pkl")
    
    # Check if metadata exists
    if not os.path.exists(codebook_ids_descriptions_path):
        raise FileNotFoundError(f"No annotated_ideas file found at {codebook_ids_descriptions_path}")
    if not os.path.exists(annotated_ideas_path):
        raise FileNotFoundError(f"No annotated_ideas file found at {annotated_ideas_path}")
    
    # Load metadata
    with open(codebook_ids_descriptions_path, "rb") as f:
        codebook_ids, codebook_descriptions = pickle.load(f)
    with open(annotated_ideas_path, "rb") as f:
        idea_ids, idea_texts,idea_annotation_ids,idea_for_user_ids,idea_object_names,idea_reasons = pickle.load(f)
    
    # Create output folder if needed
    os.makedirs("exports", exist_ok=True)
    output_folder = f"exports/{experiment_name}/"
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert metadata to DataFrame
    df_codebook = pd.DataFrame({
        "codebook_ids": codebook_ids, 
        "codebook_descriptions": codebook_descriptions
    })
    df_annotated_ideas = pd.DataFrame({
        "idea_ids": idea_ids, 
        "idea_texts": idea_texts, 
        "idea_annotation_ids": idea_annotation_ids,
        "idea_for_user_ids": idea_for_user_ids,
        "idea_object_names" : idea_object_names,
        "idea_reasons": idea_reasons
    })
    
    # Define output CSV path
    csv_path_codebook = os.path.join(output_folder, f"{experiment_name}_codebook.csv")
    csv_path_annotated_ideas = os.path.join(output_folder, f"{experiment_name}_annotated_ideas.csv")
    
    # Save DataFrame to CSV
    df_codebook.to_csv(csv_path_codebook, index=False)
    df_annotated_ideas.to_csv(csv_path_annotated_ideas, index=False)

    print(f"📦 Exported annotations and codebook for '{experiment_name}' to 'exports/' folder.\n\n==================\n\n")
    
    
    
def load_checkpoint(experiment_name):
    checkpoint_file = f"checkpoints/{experiment_name}/checkpoint.txt"
    if not os.path.exists(checkpoint_file):
        return []
    with open(checkpoint_file, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]
    
    
def save_checkpoint(experiment_name, idea_id):
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_dir = f"checkpoints/{experiment_name}/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.txt")
    # Append mode ("a") instead of overwrite mode ("w")
    with open(checkpoint_file, "a") as f:
        f.write(str(idea_id) + "\n")
        
        
        
def load_experiment_csv(filepath, successful_ids, seed=None):
    """
    Load CSV and filter out rows that have not been successfully processed yet.
    Shuffle the filtered rows randomly but reproducibly using the given seed.

    Args:
        filepath (str): Path to input CSV file.
        successful_ids (list of int): List of already processed idea IDs.
        seed (int, optional): Seed for reproducible shuffling.

    Returns:
        List of dicts (rows that still need processing, shuffled).
    """
    df = pd.read_csv(filepath)

    # Ensure 'id' column exists
    if "id" not in df.columns:
        raise ValueError("Input CSV must have an 'id' column.")
    
    # Filter out successful IDs
    df_filtered = df[~df["id"].isin(successful_ids)]

    # Shuffle rows if seed is provided
    if seed is not None:
        df_filtered = df_filtered.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_filtered.to_dict(orient="records")



def create_user_query(idea_text,comparison_ideas,prompt_method):
    if prompt_method=='CoT':
        user_query= f"""input_idea: {idea_text}
        comparison_ideas = {repr(comparison_ideas)}
        """
    elif prompt_method=='baseline':
        user_query= f"""input_idea: {idea_text}
        comparison_ideas = {repr(comparison_ideas)}
        """
    return user_query




def get_forbidden_idea(object_name):
    """
    Given an object_name, return its forbidden idea.
    Looks up from 'input_files/forbidden_ideas.csv'.

    Args:
        object_name (str): Name of the object.

    Returns:
        str: Forbidden idea for the object, or None if not found or file missing.
    """
    csv_path = "input_files/forbidden_ideas.csv"
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        return None
    
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Basic validation
    if 'object_name' not in df.columns or 'forbidden_idea' not in df.columns:
        raise ValueError("CSV must have 'object_name' and 'forbidden_idea' columns.")
    
    # Drop duplicate object_names, keeping the first forbidden idea
    df_unique = df.drop_duplicates(subset='object_name', keep='first')

    # Create dictionary: object_name -> forbidden_idea
    forbidden_dict = dict(zip(df_unique['object_name'], df_unique['forbidden_idea']))
    
    # Return the forbidden idea for the given object_name, or None if not found
    return forbidden_dict.get(object_name)


def create_comparison_ideas(forbidden_idea, similar_ideas_dict):
    """
    Create a dictionary of comparison ideas.
    
    If forbidden_idea is provided, it is included with key 0.
    Otherwise, only similar_ideas_dict is used.

    Args:
        forbidden_idea (str or None): The forbidden idea to optionally include.
        similar_ideas_dict (dict): Other ideas to include.

    Returns:
        dict: Combined dictionary of ideas.
    """
    if forbidden_idea is not None:
        common_use_dict = {0: forbidden_idea}
        return {**common_use_dict, **similar_ideas_dict}
    else:
        return similar_ideas_dict

def extract_annotation_from_noisy_text(text, prompt_method='CoT'):
    """
    Clean LLM output and extract (annotation_ID as int, reason as str).
    
    - Removes any <think>...</think> block first.
    - If prompt_method == 'CoT', expects <integer> <reason>.
    - If prompt_method == 'baseline', expects only <integer>.
    """
    text = text.strip()
    
    # Step 1: Remove <think>...</think> block if it exists
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Step 2: Extract integer and reason
    match = re.search(r'(-?\d+)\s*(.*)', text, re.DOTALL)
    
    if not match:
        raise ValueError("No valid <integer> pattern found in text.")
    
    raw_id, raw_reason = match.groups()
    
    try:
        annotation_id = int(raw_id)
    except ValueError:
        raise ValueError(f"First matched token '{raw_id}' is not a valid integer.")
    
    if prompt_method == 'baseline':
        reason = ""  # Ignore reason
    elif prompt_method == 'CoT':
        reason = raw_reason.strip()
    else:
        raise ValueError(f"Unknown prompt_method: {prompt_method}")
    
    return annotation_id, reason


def embed_texts(texts,embedder):
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    return embeddings.astype('float32')



def load_experiment_database(exp_arg,embedder):
    experiment_name = exp_arg['experiment_name']
    database_folder = os.path.join("databases", experiment_name)
    os.makedirs(database_folder, exist_ok=True)  # Make sure folder exists
    
    codebook_embeddings_path = os.path.join(database_folder, f"{experiment_name}_codebook_embeddings.npy")
    codebook_ids_descriptions_path = os.path.join(database_folder, f"{experiment_name}_codebook_ids_descriptions.pkl")
    annotated_ideas_path = os.path.join(database_folder, f"{experiment_name}_annotated_ideas.pkl")

    # Load existing embeddings and metadata if available
    if os.path.exists(codebook_embeddings_path) and os.path.exists(codebook_ids_descriptions_path) and os.path.exists(annotated_ideas_path):
        # print("Loading existing codebook and annotations data...")
        codebook_embeddings = np.load(codebook_embeddings_path)
        with open(codebook_ids_descriptions_path, "rb") as f:
            codebook_ids, codebook_descriptions = pickle.load(f)
        with open(annotated_ideas_path, "rb") as f:
            idea_ids, idea_texts,idea_annotation_ids,idea_for_user_ids,idea_object_names,idea_reasons = pickle.load(f)
        # Fit NearestNeighbors
        if len(codebook_ids)>0:
            knn = NearestNeighbors(n_neighbors=exp_arg['comparison_k'], metric='cosine')
            knn.fit(codebook_embeddings)
        else:
            knn= None
    else:
        # print("Creating new empty annotation store...")
        embedding_dim = embedder.get_sentence_embedding_dimension()
        
        # Create empty structures
        codebook_embeddings = np.empty((0, embedding_dim), dtype='float32')
        codebook_ids = []
        codebook_descriptions = []
        idea_ids = []
        idea_texts = []
        idea_annotation_ids = []
        idea_for_user_ids = []
        idea_object_names = []
        idea_reasons = []
        knn = None  # No index until enough data
        
        # Save empty files immediately
        np.save(codebook_embeddings_path, codebook_embeddings)
        with open(codebook_ids_descriptions_path, "wb") as f:
            pickle.dump((codebook_ids, codebook_descriptions), f)
        with open(annotated_ideas_path, "wb") as f:
            pickle.dump((idea_ids, idea_texts, idea_annotation_ids, idea_for_user_ids,idea_object_names ,idea_reasons), f)

    database_dict = {
        "codebook_embeddings":codebook_embeddings, 
        "codebook_ids": codebook_ids, 
        "codebook_descriptions": codebook_descriptions, 
        "idea_ids": idea_ids, 
        "idea_texts": idea_texts, 
        "idea_annotation_ids": idea_annotation_ids,
        "idea_for_user_ids": idea_for_user_ids,
        "idea_object_names" : idea_object_names,
        "idea_reasons": idea_reasons,
        "knn": knn
    }
    return database_dict


def annotate_idea(agent,embedder,idea_arg,exp_arg,forbidden_idea):
    # print("Loading database")
    database_dict = load_experiment_database(exp_arg,embedder)
    
    # print("Generating embedding of the new idea")
    new_embedding = embed_texts([idea_arg['idea_content']],embedder)

    # print("Searching for similar annotations")
    if len(database_dict['codebook_ids']) > 0:
        if len(database_dict['codebook_ids']) <= exp_arg['comparison_k']:
            # Too few examples, return all
            similar_ideas_dict = {
                database_dict["codebook_ids"][idx]: database_dict["codebook_descriptions"][idx]
                for idx in range(len(database_dict["codebook_ids"]))
            }
        else:
            # Enough examples, run KNN
            D, I = database_dict['knn'].kneighbors(new_embedding, n_neighbors=exp_arg['comparison_k'])
            similar_ideas_dict = {
                database_dict["codebook_ids"][idx]: database_dict["codebook_descriptions"][idx]
                for idx in I[0]
            }
    else:
        similar_ideas_dict = {}

    # print("Creating query for LLM annotator")
    comparison_ideas = create_comparison_ideas(forbidden_idea,similar_ideas_dict)
    user_query = create_user_query(idea_arg["idea_content"],comparison_ideas,exp_arg['prompt_method'])

    # print("Generating LLM annotation")
    response = agent.query(user_query)
    try:
        current_annotation_id, current_reason = extract_annotation_from_noisy_text(response,exp_arg['prompt_method'])
        # print(f"Idea ID: {idea_arg['id']}, Idea: {idea_arg['idea_content']}, annotation id: {current_annotation_id}, reason: {current_reason}\n")
    except Exception as e:
        print(f"Failed response: {response}")
        return False

    database_folder = os.path.join("databases", exp_arg["experiment_name"])
    
    if current_annotation_id == -1:
        # print("Preparing codebook update since a new annotation was created")
        updated_codebook_embeddings = np.vstack([database_dict['codebook_embeddings'], new_embedding]) # saving the new bucket's embedding
        
        updated_codebook_descriptions = database_dict['codebook_descriptions'].copy()
        updated_codebook_descriptions.append(idea_arg["idea_content"]) # this idea's content becomes the description for this new bucket
        
        updated_codebook_ids = database_dict['codebook_ids'].copy()
        current_annotation_id = len(database_dict['codebook_embeddings'])+1 # increment ID by one. +1 makes sure the 0 ID is preserved for common use
        updated_codebook_ids.append(current_annotation_id) 

        # print("Saving updated codebook")
        codebook_embeddings_path = os.path.join(database_folder, f"{exp_arg['experiment_name']}_codebook_embeddings.npy")
        codebook_ids_descriptions_path = os.path.join(database_folder, f"{exp_arg['experiment_name']}_codebook_ids_descriptions.pkl")
        
        np.save(codebook_embeddings_path, updated_codebook_embeddings)
        with open(codebook_ids_descriptions_path, "wb") as f:
            pickle.dump((updated_codebook_ids, updated_codebook_descriptions), f)

    # print("Preparing annotated data point for saving")
    # idea IDs
    updated_idea_ids = database_dict['idea_ids'].copy()
    updated_idea_ids.append(idea_arg["id"])

    # idea texts
    updated_idea_texts = database_dict['idea_texts'].copy()
    updated_idea_texts.append(idea_arg["idea_content"])

    # annotation IDs
    updated_annotation_ids = database_dict['idea_annotation_ids'].copy()
    updated_annotation_ids.append(current_annotation_id) 

    # for user IDs
    updated_for_user_ids = database_dict['idea_for_user_ids'].copy()
    updated_for_user_ids.append(idea_arg["for_user_id"])

    # object_name
    updated_object_names = database_dict['idea_object_names'].copy()
    updated_object_names.append(exp_arg["object_name"])

    # reasons
    updated_reasons = database_dict['idea_reasons'].copy()
    updated_reasons.append(current_reason) 

    # print("Saving annotated data point")
    annotated_ideas_path = os.path.join(database_folder, f"{exp_arg['experiment_name']}_annotated_ideas.pkl")
    with open(annotated_ideas_path, "wb") as f:
            pickle.dump((updated_idea_ids, updated_idea_texts,updated_annotation_ids,updated_for_user_ids,updated_object_names,updated_reasons), f)
            
            
def run_experiment(exp_arg):
    failed_ideas = [] 
    for itr_ in range(exp_arg['max_attempts']):
        # print("Checking Progress")
        successful_ids = load_checkpoint(exp_arg['experiment_name'])
        ideas = load_experiment_csv(exp_arg['input_csv_path'], successful_ids, exp_arg['replication_id'])
        total_ideas = len(ideas)
        
        if total_ideas == 0:
            print(f"All ideas have been annotated for the experiment {exp_arg['experiment_name']}; Exiting.")
            break
        else:
            print(f"{total_ideas} ideas remaining for the experiment {exp_arg['experiment_name']}.")
            print(f"Attempt {itr_+1}.")
            
        # print("Loading any forbidden idea for the object")
        forbidden_idea = get_forbidden_idea(exp_arg['object_name'])
    
        # print("Initiating LLM Agent and Embedder Models")
        agent = OllamaAgent(
            model_name=exp_arg['llm_model'], 
            object_name=exp_arg['object_name'],
            comparison_k=exp_arg['comparison_k'],
            prompt_method=exp_arg['prompt_method']
        )
        
        failed_ideas = []  # To collect failed ideas
        
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else: 
            device = "cpu"
            
        embedder = SentenceTransformer(exp_arg['embedding_model'], device=device)
    
        # print("Models Initialized! Let's loop through ideas")
        for idx, idea in enumerate(tqdm(ideas, desc="🔍 Annotating Ideas", dynamic_ncols=True)):
            try:
                annotate_idea(
                    agent=agent,
                    embedder=embedder,
                    idea_arg = idea,
                    exp_arg=exp_arg,
                    forbidden_idea=forbidden_idea
                )
                save_checkpoint(exp_arg['experiment_name'], idea["id"])
            except Exception as e:
                print(f"❌ Failed to annotate idea {idx + 1}: {str(e)}")
                failed_ideas.append(idea["id"])
    
        if failed_ideas:
            failed_dir = f"checkpoints/{exp_arg['experiment_name']}/"
            os.makedirs(failed_dir, exist_ok=True)
            failed_path = os.path.join(failed_dir, "failed_ids.txt")
            with open(failed_path, "w") as f:
                for idea_id in failed_ideas:
                    f.write(f"{idea_id}")
            print(f"❌ Saved {len(failed_ideas)} failed ideas to {failed_path}")

    if exp_arg['save_csvs']:
        if failed_ideas:
            print("You have unfinished annotations.")
        else:
            export_annotations(exp_arg['experiment_name'])

def process_and_merge_sorted_csvs(
    study_prefix,
    llm_index,
    prompt_index,
    embedding_index,
    comparison_k,
    replication_id,
    object_names,
    base_path='exports'
):
    """
    Sorts and merges annotated idea CSVs from experiment folders based on naming conventions.

    Parameters:
        study_prefix (str): Prefix for the experiment name (e.g., 'simpl_prmpt').
        llm_model (str): Model identifier (e.g., 'phi4', 'llama3.3').
        prompt_method (str): Either 'CoT' or 'baseline'.
        embedding_code (int): Embedding method code.
        comparison_k (int): Number of comparison ideas.
        replication_id (int): Replication ID or seed.
        base_path (str): Root folder containing experiment folders.
    """
     
    # Generate experiment names
    experiment_names = [
        f"{study_prefix}_m{llm_index}_p{prompt_index}_e{embedding_index}_k{comparison_k}_r{replication_id}_{object_name}"
        for object_name in object_names
    ]

    all_dfs = []

    for exp_name in experiment_names:
        folder_path = os.path.join(base_path, exp_name)
        original_csv = f"{exp_name}_annotated_ideas.csv"
        original_path = os.path.join(folder_path, original_csv)

        # Load and sort
        df = pd.read_csv(original_path)
        df_sorted = df.sort_values(by=['idea_for_user_ids','idea_ids'], ascending=[True, True])

        # Save sorted version
        sorted_csv = f"{exp_name}_annotated_ideas_sorted.csv"
        sorted_path = os.path.join(folder_path, sorted_csv)
        df_sorted.to_csv(sorted_path, index=False)

        all_dfs.append(df_sorted)

    # Merge and save
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Use common prefix (removing object_name)
    merged_prefix = f"{study_prefix}_m{llm_index}_p{prompt_index}_e{embedding_index}_k{comparison_k}_r{replication_id}"
    final_merged_path = os.path.join(base_path, f"{merged_prefix}_all.csv")
    final_df.to_csv(final_merged_path, index=False)
    print(f"📦 Exported merged CSV for '{merged_prefix}' to 'exports/' folder.\n\n==================\n\n")
    

def muserag(study_prefix,llm_index,prompt_index,embedding_index,comparison_k,replication_id,object_names):
    llm_choice_dict = {
        1: "llama3.3", 
        2: "phi4", 
        3: "qwen3"
    }
    
    prompt_choice_dict = {
        1: "baseline", 
        2: "CoT", 
    }
    
    embedding_choice_dict = {
        1: "mixedbread-ai/mxbai-embed-large-v1", 
        2: "intfloat/e5-large-v2", 
        3: "all-mpnet-base-v2", 
        4: "BAAI/bge-large-en-v1.5"
    }
    
    llm_model = llm_choice_dict[llm_index]
    prompt_method = prompt_choice_dict[prompt_index]
    embedding_code = embedding_choice_dict[embedding_index]
    
    # the rest will take care of itself
    for object_name in object_names: 
        exp_arg = {
            'experiment_name': f"{study_prefix}_m{llm_index}_p{prompt_index}_e{embedding_index}_k{comparison_k}_r{replication_id}_{object_name}",
            'object_name': object_name, 
            'input_csv_path': f"input_files/ideas_{object_name}.csv",
            'llm_model': llm_model, 
            'prompt_method': prompt_method,
            'replication_id': replication_id,
            'comparison_k': comparison_k,
            'save_csvs': True,
            'max_attempts':3,
            'embedding_model': embedding_code
        }
        run_experiment(exp_arg)
        
    process_and_merge_sorted_csvs(study_prefix,llm_index,prompt_index,embedding_index,comparison_k,replication_id,object_names,base_path='exports')
    

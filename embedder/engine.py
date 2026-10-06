from sentence_transformers import SentenceTransformer

def get_model(model_name="bkai-foundation-models/vietnamese-bi-encoder"):
    """
    Load and return a SentenceTransformer model based on the provided model name.

    Args:
        model_name (str): The name of the model to load.

    """
    return SentenceTransformer(model_name)
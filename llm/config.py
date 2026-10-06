from lmdeploy import GeneratorConfig, pipeline

def get_pipeline():
    """
    Load and return a Qwen2.5-7B-Instruct model pipeline.

    Returns:
        pipeline: A pipeline object for the Qwen2.5-7B-Instruct model.
    """
    config = pipeline(model="Qwen/Qwen2.5-7B-Instruct", tp=1)
    return pipeline(config)

def get_generator_config():
    """
    Load and return a GeneratorConfig for the Qwen2.5-7B-Instruct model.

    Returns:
        GeneratorConfig: A configuration object for the Qwen2.5-7B-Instruct model.
    """
    config = GeneratorConfig(temperature=0.7, top_p=0.8, max_new_tokens=512)
    return config
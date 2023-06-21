from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs="hearts",
                outputs="preprocessed_hearts",
                name="preprocess_hearts_node",
            ),
        ]
    )

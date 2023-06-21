from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, prepare_data_for_modeling, hyperparameters_tuning


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_data_for_modeling,
                inputs="preprocessed_hearts",
                outputs="model_input_hearts_processed",
                name="prepare_data_for_modeling_node",
            ),
            node(
                func=split_data,
                inputs=["model_input_hearts_processed", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=hyperparameters_tuning,
                inputs=None,
                outputs=["n_estimators", "max_depth"],
                name="hyperparameters_tuning_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "n_estimators", "max_depth"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs=["accuracy", "roc_auc"],
                name="evaluate_model_node",
            ),
        ]
    )

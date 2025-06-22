"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs="preprocessed_student_depression",
            outputs="depression_model",
            name="train_model_node"
        )
    ])

from .nodes import train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs="preprocessed_student_depression",
            outputs="depression_model",
            name="train_model_node"
        ),
        node(
            func=evaluate_model,
            inputs="preprocessed_student_depression",
            outputs="model_evaluation_result",
            name="evaluate_model_node"
        )
    ])

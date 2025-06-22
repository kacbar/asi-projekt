"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs="student_depression",
                outputs="preprocessed_student_depression",
                name="preprocess_node",
            ),
        ]
    )

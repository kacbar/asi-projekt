from student_depression.pipelines import modeling, data_processing as dp

def register_pipelines():
    return {
        "dp": dp.create_pipeline(),
        "modeling": modeling.create_pipeline(),
        "__default__": dp.create_pipeline() + modeling.create_pipeline()
    }


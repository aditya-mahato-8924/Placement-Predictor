from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.schema.validate_user_input import PlacementPredictorInput
from src.schema.validate_output import ResponseModel

# create FastAPI app
app = FastAPI(
    title='Placement Prediction API',
    description='API for predicting student placement based on various features.',
    version='1.0.0'
)

# create a home page route
@app.get('/')
def home():
    """Home page for the Placement Prediction API."""
    welcome_msg = {"message": "Welcome to the Placement Prediction API"}

    return JSONResponse(content=welcome_msg, status_code=200)

# create a health check route
@app.get('/health')
def health_check():
    """Health check endpoint for the Placement Prediction API."""

    model_info =  {
        "status": "OK",
        "model_name": "Placement Predictor Model",
        "model_version": "3",
    }

    return JSONResponse(content=model_info, status_code=200)


# create a prediction route
@app.post('/predict', response_model=ResponseModel)
def get_prediction(data: PlacementPredictorInput):
    """Endpoint to get placement prediction based on input features."""
    try:
        # convert Pydantic model to dictionary
        input_data = data.model_dump()

        # convert the input data to a dataframe
        custom_data = CustomData(**input_data)
        data = custom_data.get_data_as_dataframe()

        # create an instance of the prediction pipeline
        predict_pipeline = PredictPipeline()

        # get the prediction
        result = predict_pipeline.get_prediction(data)

        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        # return the error response
        return JSONResponse(status_code=500, content=str(e))
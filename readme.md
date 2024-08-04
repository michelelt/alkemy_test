# install venv
oss: this is valid only for unix
`python3 -m venv venv`
`source venv\bin\activate`
`pip install -r requirements.txt`

# Run the to create the model
`python main.py`
- you need to create an sql server
- the validation step predicts flows on a data chunk that was not included into training
- the directory resutls contains the results. for final performances check `validation`

# deployment
the deployment is a demo. It relies on a server flask. The server receves in input (post) a json and return the predictions.  
To run the demo:
- activate the venv `source venv\bin\activate`
- train a model `python main.py`
- start the flask server `python app.py`
- run the requests `request_example.py`

# improvements
- it is possible crater a system that monitors the perdiction degradation.
    - once the predicted MAE > threshold the systme must be retrained
    - it will changes the tree splits and recompute the `magic number`

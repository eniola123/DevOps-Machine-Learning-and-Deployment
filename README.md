# Interview Task

The goal of this task is to create a simple machine learning application.

1) Use the `training_data.csv` to build and train a simple classification model. The model
needs to predict the `target` using the features `f1`, `f2`, and `f3`.

2) Build a Flask application with a REST API that will use the trained model to classify new data samples.

3) Use a MySQL or SQLite database to log and monitor the model performance.

The Flask application shall have two endpoints: `/classify` and `/stats`.

The first endpoint `/classify` accepts `GET` requests
with three parameters `f1`, `f2`, and `f3` (the features used in the training data). For example:

```bash
https://localhost:5000/classify?f3=A&f1=0.23&f2=0.4
```

The application returns the following `JSON` response:

```json
{"predicted_class": 0, "status": "OK"}
```

Each `/classify` request and response shall be logged in a database table. Below is an example
of an SQLite schema. However, you can design your own schema.

```SQL
CREATE TABLE classification_requests
(
    id_request INTEGER PRIMARY KEY,
    request_timestamp TEXT NOT NULL,
    predicted_class INTEGER,
    response_status TEXT NOT NULL,
    error_message TEXT
);

-- to store individual parameters for each request
CREATE TABLE classification_request_params
(
    id_request_param INTEGER PRIMARY KEY,
    id_request INTEGER,
    param_name TEXT NOT NULL,
    param_value TEXT NOT NULL,
    FOREIGN KEY(id_request) REFERENCES classification_requests(id_request)
)
```

If the `predicted_class` is the same as the prediction in __two__ most recently logged requests,
the response `status` shall be: `WARNING`.

```json
{"predicted_class": 0, "status": "WARNING"}
```

If the request is not valid (e.g. wrong parameters), the response shall contain `error_message` and its
`status` shall be `ERROR`.

```json
{"status": "ERROR", "error_message": "f1 is not a float."}
```

The second enpoint `/stats` accepts `GET` requests with no parameters:

```bash
https://localhost:5000/stats
```

The response contains statistics for the logged classification requests with `status=OK`:
mean values for `f1`, `f2`, and the most frequent value for `f3`.

```json
{"mean_f1": 0.25, "mean_f2": 0.33, "most_frequent_f3": "A"}
```

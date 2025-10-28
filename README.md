# LAIA - Taxi Trip Prediction

**Repository for the Advanced Artificial Intelligence Laboratory course.**  
This project focuses on designing, implementing, and managing a complete **Machine Learning Operations (MLOps)** pipeline.  
The goal is to build and deploy a model capable of performing **NYC Taxi Trip Duration Prediction**.

---

To run the full system locally, follow the steps below:

### 1. Build and start the services
Run the following command in the root directory:
```bash
docker-compose up -d --build
```

### 2. Train the model

Once all services are up, execute:
```bash
python train.py
```
### 3. Access MLFlow UI
After training completes, open MLflow in your browser:
```
URL
http://localhost:5000/
```
### 4. Promote the trained model

In the MLflow UI:

- Go to Models → iris

- Promote the latest model version to Production
(Use the deprecated Model Registry UI if needed)

### 5. Use the API endpoints

Once the model is in production, you can start sending requests to the Flask API (for example, via `curl` or Postman).

Endpoints:
```bash
curl -X POST http://localhost:8080/predict ^
 -H "Content-Type: application/json" ^
 -d "{\"columns\": [\"f1\", \"f2\", \"f3\", \"f4\"], \"data\": [[5.1, 3.5, 1.4, 0.2]]}"

 curl -X POST http://localhost:8080/reload
```
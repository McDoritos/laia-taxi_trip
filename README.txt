Para o Flask API funcionar teve de ser ir ao "http://localhost:5000/" e promover o modelo para
production manualmente pois a API está à espera de ver um modelo em produção. Foram também mantidas
as correções feitas pelo professor no lab anterior que ainda nao tinham sido feitas neste.

curl -X POST http://localhost:8080/predict ^
 -H "Content-Type: application/json" ^
 -d "{\"columns\": [\"f1\", \"f2\", \"f3\", \"f4\"], \"data\": [[5.1, 3.5, 1.4, 0.2]]}"

curl -X POST http://localhost:8080/reload
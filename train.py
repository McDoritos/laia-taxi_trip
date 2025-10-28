import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configurar o MLflow para usar o servidor remoto com upload de artefatos
mlflow.set_tracking_uri("http://localhost:5000")

# Nome do experimento e URI remoto para artefatos
experiment_name = "iris_classification"
artifact_uri = "mlflow-artifacts:/"

# Verificar se o experimento já existe e tem o local de artefatos correto
existing_experiment = mlflow.get_experiment_by_name(experiment_name)

if existing_experiment:
    # Verifica se o experimento existente tem um caminho problemático de artefatos (ex: Docker local)
    if existing_experiment.artifact_location.startswith("/mlflow") or \
       existing_experiment.artifact_location.startswith("file:///mlflow"):
        print(f"Existing experiment has Docker path artifact location: {existing_experiment.artifact_location}")
        print("Creating new experiment with remote artifact storage...")

        # Usa um novo nome de experimento
        experiment_name = "iris_classification_remote"
        try:
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
            print(f"Created new experiment '{experiment_name}'")
        except Exception:
            pass
    else:
        print(f"Using existing experiment '{experiment_name}'")
else:
    # Criar novo experimento com local de artefatos remoto
    try:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
        print(f"Created new experiment '{experiment_name}' with remote artifact storage")
    except Exception as e:
        print(f"Note: {e}")

mlflow.set_experiment(experiment_name)

# Carregar dados do Iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

best_acc = 0
best_run_id = None
best_C = None

# Treinar modelos com diferentes valores de C
for C in [0.1, 1.0, 10.0]:
    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=200, C=C)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log de parâmetros e métricas
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)

        # Criar gráfico e logar como artefato
        fig, ax = plt.subplots()
        ax.scatter(y_test, preds)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"C={C}")
        mlflow.log_figure(fig, f"results_C{C}.png")
        plt.close(fig)

        # Inferir assinatura e logar modelo
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, input_example=X_train[:5])

        # Guardar o melhor modelo
        if acc > best_acc:
            best_acc = acc
            best_run_id = run.info.run_id
            best_C = C

# Registrar o melhor modelo no MLflow Model Registry
print(f"\nBest model: C={best_C}, accuracy={best_acc:.4f}")
model_uri = f"runs:/{best_run_id}/model"
registered_model = mlflow.register_model(model_uri, "iris")
print(f"Registered model version: {registered_model.version}")

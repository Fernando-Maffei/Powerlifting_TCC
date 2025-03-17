import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Adicionado para criar o diretório

def evaluate_model(model_path, data_path, output_path="results/evaluation_results.txt"):
    """
    Avalia o modelo treinado usando um conjunto de teste.
    """
    modelo = joblib.load(model_path)
    print("Modelo carregado com sucesso.")

    df_test = pd.read_parquet(data_path)
    print("Distribuição das classes no conjunto de teste:")
    print(df_test["correto"].value_counts())

    X_test = df_test.drop(columns=["exercicio", "correto"])
    y_test = df_test["correto"]

    # Verificar e tratar valores NaN
    if X_test.isnull().any().any():
        print("Valores NaN encontrados no conjunto de teste. Preenchendo com zeros...")
        X_test = X_test.fillna(0)  # Preenche valores NaN com zeros

    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo no conjunto de teste: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    report = classification_report(y_test, y_pred, zero_division=0)  # Evita warnings de divisão por zero
    print(report)

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("\nMatriz de Confusão:")
    print(cm)

    # Visualização da Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Incorreto', 'Correto'], yticklabels=['Incorreto', 'Correto'])
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')

    # Cria o diretório "results" se ele não existir
    os.makedirs("results", exist_ok=True)

    # Salva a matriz de confusão como uma imagem
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    # Salvar resultados da avaliação
    with open(output_path, 'w') as f:
        f.write(f"Acurácia do modelo no conjunto de teste: {accuracy:.4f}\n")
        f.write("\nRelatório de Classificação:\n")
        f.write(report)
        f.write("\nMatriz de Confusão:\n")
        f.write(str(cm))

if __name__ == "__main__":
    evaluate_model("models/classificador_movimentos.pkl",
                   "dataset/processed/keypoints_data.parquet")
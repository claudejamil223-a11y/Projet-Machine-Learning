import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from ucimlrepo import fetch_ucirepo

# On s'assure que les dossiers existent
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports/tables", exist_ok=True)

SEEDS = [42, 100, 2023, 777, 1234]

def load_and_preprocess_data():
    print("Téléchargement du dataset Online Shoppers...")
    dataset = fetch_ucirepo(id=468)
    X = dataset.data.features
    y = dataset.data.targets

    print("Prétraitement des données (MODE CORRIGÉ)...")
    y = y['Revenue'].astype(int)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # CORRECTION : On supprime la variable "PageValues" qui servait de raccourci/triche au modèle
    if 'PageValues' in X_encoded.columns:
        X_encoded = X_encoded.drop(columns=['PageValues'])
        print("-> Variable 'PageValues' supprimée avec succès pour forcer un vrai apprentissage.")
        
    return X_encoded, y

def run_experiment():
    X, y = load_and_preprocess_data()
    results = []

    print(f"\n--- Lancement du modèle CORRIGÉ sur {len(SEEDS)} seeds ---")
    
    for seed in SEEDS:
        print(f"\nExécution pour la seed : {seed}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        model = RandomForestClassifier(random_state=seed)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results.append({
            'Seed': seed,
            'Accuracy_Globale': round(report['accuracy'], 4),
            'Recall_Acheteurs': round(report['1']['recall'], 4),
            'Precision_Acheteurs': round(report['1']['precision'], 4)
        })

        # Génération de la figure pour la première seed
        if seed == SEEDS[0]:
            print("Génération du graphique de la NOUVELLE importance des variables...")
            importances = model.feature_importances_
            feature_imp_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
            feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(15)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_imp_df, hue='Feature', palette='magma', legend=False)
            plt.title('Top 15 des Variables (Modèle Corrigé - Sans PageValues)')
            plt.tight_layout()
            
            plt.savefig('reports/figures/feature_importance_corrected.png', dpi=300)
            plt.close()

    # Sauvegarde des résultats
    df_results = pd.DataFrame(results)
    csv_path = 'reports/tables/metrics_corrected_model.csv'
    df_results.to_csv(csv_path, index=False)
    
    print("\n=================================================")
    print("RÉSUMÉ DU MODÈLE CORRIGÉ (MOYENNE SUR 5 SEEDS)")
    print("=================================================")
    print(df_results.mean(numeric_only=True))

if __name__ == "__main__":
    run_experiment()
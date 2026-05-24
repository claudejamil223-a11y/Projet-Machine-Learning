import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from ucimlrepo import fetch_ucirepo

# --- Configuration et Création des dossiers ---
# On s'assure que les dossiers pour les livrables existent
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports/tables", exist_ok=True)

# Définition des 5 seeds exigées par le barème
SEEDS = [42, 100, 2023, 777, 1234]

def load_and_preprocess_data():
    print("Téléchargement du dataset Online Shoppers...")
    online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468)
    X = online_shoppers_purchasing_intention_dataset.data.features
    y = online_shoppers_purchasing_intention_dataset.data.targets

    print("Prétraitement des données...")
    # Conversion de la cible en binaire
    y = y['Revenue'].astype(int)
    
    # Encodage One-Hot des variables catégorielles
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y

def run_experiment():
    X, y = load_and_preprocess_data()
    
    # Identifier les colonnes liées au TrafficType
    traffic_cols = [col for col in X.columns if 'TrafficType' in col]
    
    results = []

    print(f"\n--- Lancement de l'expérience sur {len(SEEDS)} seeds ---")
    
    for seed in SEEDS:
        print(f"\nExécution pour la seed : {seed}")
        
        # 1. Split Train/Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # 2. Entraînement du modèle "Broken" (avec le biais)
        model = RandomForestClassifier(random_state=seed)
        model.fit(X_train, y_train)
        
        # 3. Évaluation sur données normales (Baseline)
        y_pred_base = model.predict(X_test)
        report_base = classification_report(y_test, y_pred_base, output_dict=True)
        recall_base = report_base['1']['recall']
        
        # 4. Perturbation : Forcer TrafficType à 0 pour les acheteurs
        X_test_perturbed = X_test.copy()
        mask_buyers = (y_test == 1)
        X_test_perturbed.loc[mask_buyers, traffic_cols] = 0
        
        # 5. Évaluation sur données perturbées
        y_pred_perturbed = model.predict(X_test_perturbed)
        report_perturbed = classification_report(y_test, y_pred_perturbed, output_dict=True)
        recall_perturbed = report_perturbed['1']['recall']
        
        # Sauvegarde des métriques pour cette seed
        results.append({
            'Seed': seed,
            'Recall_Acheteurs_Normal': round(recall_base, 4),
            'Recall_Acheteurs_Perturbe': round(recall_perturbed, 4),
            'Chute_Recall': round(recall_base - recall_perturbed, 4)
        })

        # --- Génération et sauvegarde de la figure (uniquement pour la première seed) ---
        if seed == SEEDS[0]:
            print("Génération du graphique de l'importance des variables...")
            importances = model.feature_importances_
            feature_names = X_train.columns
            feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(15)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
            plt.title('Top 15 des Variables les Plus Importantes (Modèle Biaisé)')
            plt.tight_layout()
            
            # Sauvegarde de l'image
            plt.savefig('reports/figures/feature_importance_broken.png', dpi=300)
            plt.close()
            print("Graphique sauvegardé dans 'reports/figures/feature_importance_broken.png'")

    # --- Sauvegarde des résultats sous forme de tableau CSV ---
    df_results = pd.DataFrame(results)
    csv_path = 'reports/tables/metrics_broken_model.csv'
    df_results.to_csv(csv_path, index=False)
    
    print("\n=================================================")
    print("RÉSUMÉ DE L'EXPÉRIENCE (MOYENNE SUR LES 5 SEEDS)")
    print("=================================================")
    print(df_results.mean(numeric_only=True))
    print(f"\nTous les résultats ont été sauvegardés dans : {csv_path}")

if __name__ == "__main__":
    run_experiment()
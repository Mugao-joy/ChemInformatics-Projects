#!/usr/bin/env python3
"""
Molecule Property Predictor - Cheminformatics Project
Predicts aqueous solubility of molecules using RDKit descriptors and scikit-learn

This project demonstrates:
- Molecular descriptor calculation with RDKit
- Machine learning for molecular property prediction
- Data visualization and model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, GraphDescriptors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MoleculePropertyPredictor:
    """
    A class for predicting molecular properties using RDKit descriptors and ML models
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def smiles_to_mol(self, smiles):
        """Convert SMILES string to RDKit molecule object"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
        return mol
    
    def calculate_descriptors(self, smiles_list):
        """
        Calculate molecular descriptors for a list of SMILES
        Returns a DataFrame with calculated descriptors
        """
        descriptors_data = []
        valid_smiles = []
        
        print(f"Calculating descriptors for {len(smiles_list)} molecules...")
        
        for i, smiles in enumerate(smiles_list):
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                # Calculate various molecular descriptors
                desc_dict = {
                    'MolWt': Descriptors.MolWt(mol),
                    'LogP': Crippen.MolLogP(mol),
                    'NumHDonors': rdMolDescriptors.CalcNumHBD(mol),
                    'NumHAcceptors': rdMolDescriptors.CalcNumHBA(mol),
                    'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                    'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                    'TPSA': rdMolDescriptors.CalcTPSA(mol),
                    'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
                    'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
                    'RingCount': rdMolDescriptors.CalcNumRings(mol),
                    'FractionCsp3': rdMolDescriptors.CalcFractionCSP3(mol),
                    'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
                    'BertzCT': GraphDescriptors.BertzCT(mol)
                }
                
                descriptors_data.append(desc_dict)
                valid_smiles.append(smiles)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} molecules...")
        
        print(f"Successfully processed {len(descriptors_data)} out of {len(smiles_list)} molecules")
        
        df_descriptors = pd.DataFrame(descriptors_data)
        df_descriptors['SMILES'] = valid_smiles
        
        return df_descriptors
    
    def create_sample_data(self, n_samples=1000):
        """
        Create sample molecular data for demonstration
        In practice, you would load real experimental data
        """
        print("Creating sample molecular dataset...")
        
        # Sample SMILES strings representing various drug-like molecules
        sample_smiles = [
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CC1=CC=C(C=C1)C(=O)O',  # p-Toluic acid
            'O=C(O)C1=CC=CC=C1',  # Benzoic acid
            'CCO',  # Ethanol
            'C1=CC=CC=C1',  # Benzene
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O',  # Salbutamol
            'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',  # Tryptophan
            'C1=CC=C(C=C1)CCN',  # Phenethylamine
        ] * 100  # Repeat to get more samples
        
        # Add some random variations and simple molecules
        simple_molecules = [
            'C', 'CC', 'CCC', 'CCCC', 'CCCCC',  # Alkanes
            'CCO', 'CCCO', 'CCCCO',  # Alcohols
            'CC(=O)O', 'CCC(=O)O',  # Carboxylic acids
            'c1ccccc1', 'c1ccc(cc1)C', 'c1ccc(cc1)O',  # Aromatics
        ] * 100
        
        all_smiles = (sample_smiles + simple_molecules)[:n_samples]
        
        # Calculate descriptors
        df_descriptors = self.calculate_descriptors(all_smiles)
        
        # Generate synthetic solubility data based on known relationships
        # This mimics real solubility trends but isn't actual experimental data
        np.random.seed(42)
        
        # Solubility tends to decrease with molecular weight and LogP
        # and increase with polar surface area and H-bond donors/acceptors
        synthetic_solubility = (
            -0.02 * df_descriptors['MolWt'] +
            -1.5 * df_descriptors['LogP'] +
            0.01 * df_descriptors['TPSA'] +
            0.3 * df_descriptors['NumHDonors'] +
            0.2 * df_descriptors['NumHAcceptors'] +
            np.random.normal(0, 1, len(df_descriptors))  # Add noise
        )
        
        df_descriptors['Solubility_LogS'] = synthetic_solubility
        
        return df_descriptors
    
    def prepare_features(self, df):
        """Prepare feature matrix for machine learning"""
        # Select numerical descriptor columns (exclude SMILES and target)
        feature_columns = [col for col in df.columns 
                          if col not in ['SMILES', 'Solubility_LogS']]
        
        X = df[feature_columns].values
        self.feature_names = feature_columns
        
        return X
    
    def train_model(self, df, test_size=0.2, model_type='rf'):
        """Train the machine learning model"""
        print("Training machine learning model...")
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df['Solubility_LogS'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose model
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\nModel Performance:")
        print(f"Training R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        print(f"Training RMSE: {train_rmse:.3f}")
        print(f"Test RMSE: {test_rmse:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                   cv=5, scoring='r2')
        print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return {
            'X_train': X_train_scaled, 'X_test': X_test_scaled,
            'y_train': y_train, 'y_test': y_test,
            'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse
        }
    
    def plot_results(self, results):
        """Create visualization plots"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training set predictions vs actual
        axes[0, 0].scatter(results['y_train'], results['y_pred_train'], 
                          alpha=0.6, color='blue')
        axes[0, 0].plot([results['y_train'].min(), results['y_train'].max()], 
                       [results['y_train'].min(), results['y_train'].max()], 
                       'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Solubility (LogS)')
        axes[0, 0].set_ylabel('Predicted Solubility (LogS)')
        axes[0, 0].set_title(f'Training Set (R² = {results["train_r2"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Test set predictions vs actual
        axes[0, 1].scatter(results['y_test'], results['y_pred_test'], 
                          alpha=0.6, color='green')
        axes[0, 1].plot([results['y_test'].min(), results['y_test'].max()], 
                       [results['y_test'].min(), results['y_test'].max()], 
                       'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Solubility (LogS)')
        axes[0, 1].set_ylabel('Predicted Solubility (LogS)')
        axes[0, 1].set_title(f'Test Set (R² = {results["test_r2"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feature importance (for Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            axes[1, 0].bar(range(len(indices)), importances[indices])
            axes[1, 0].set_xlabel('Feature Index')
            axes[1, 0].set_ylabel('Importance')
            axes[1, 0].set_title('Top 10 Feature Importances')
            axes[1, 0].set_xticks(range(len(indices)))
            axes[1, 0].set_xticklabels([self.feature_names[i] for i in indices], 
                                      rotation=45, ha='right')
        
        # Plot 4: Residuals
        residuals = results['y_test'] - results['y_pred_test']
        axes[1, 1].scatter(results['y_pred_test'], residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Solubility (LogS)')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_molecule(self, smiles):
        """Predict solubility for a single molecule"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Calculate descriptors for the molecule
        df_single = self.calculate_descriptors([smiles])
        
        if len(df_single) == 0:
            return None, "Invalid SMILES"
        
        # Prepare features
        X_single = self.prepare_features(df_single)
        X_single_scaled = self.scaler.transform(X_single)
        
        # Make prediction
        prediction = self.model.predict(X_single_scaled)[0]
        
        return prediction, "Success"
    
    def analyze_descriptors(self, df):
        """Analyze and visualize molecular descriptors"""
        print("\nAnalyzing molecular descriptors...")
        
        # Basic statistics
        desc_stats = df.describe()
        print("\nDescriptor Statistics:")
        print(desc_stats)
        
        # Correlation with solubility (only numeric columns)
        df_numeric = df.select_dtypes(include=[np.number])
        correlations = df_numeric.corr()['Solubility_LogS'].abs().sort_values(ascending=False)
        print("\nTop correlations with solubility:")
        print(correlations.head(10))
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
        plt.title('Molecular Descriptor Correlation Matrix')
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the complete pipeline"""
    print("🧪 Molecule Property Predictor - Cheminformatics Project")
    print("=" * 60)
    
    # Initialize predictor
    predictor = MoleculePropertyPredictor()
    
    # Step 1: Create/Load dataset
    print("\n1. Creating sample dataset...")
    df = predictor.create_sample_data(n_samples=500)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Step 2: Analyze descriptors
    print("\n2. Analyzing molecular descriptors...")
    predictor.analyze_descriptors(df)
    
    # Step 3: Train model
    print("\n3. Training machine learning model...")
    results = predictor.train_model(df, model_type='rf')
    
    # Step 4: Visualize results
    print("\n4. Creating visualizations...")
    predictor.plot_results(results)
    
    # Step 5: Test predictions on new molecules
    print("\n5. Testing predictions on new molecules...")
    test_molecules = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'CCO',  # Ethanol
        'C1=CC=CC=C1',  # Benzene
    ]
    
    molecule_names = ['Aspirin', 'Ibuprofen', 'Ethanol', 'Benzene']
    
    for name, smiles in zip(molecule_names, test_molecules):
        pred, status = predictor.predict_molecule(smiles)
        if status == "Success":
            print(f"{name}: Predicted LogS = {pred:.2f}")
        else:
            print(f"{name}: {status}")
    
    print("\n🎉 Project completed successfully!")
    print("\nNext steps to explore:")
    print("- Try different ML algorithms (SVM, Neural Networks)")
    print("- Use real experimental datasets (ChEMBL, EPA, etc.)")
    print("- Add more sophisticated descriptors (Morgan fingerprints)")
    print("- Implement uncertainty quantification")
    print("- Build a web interface for predictions")


if __name__ == "__main__":
    main()
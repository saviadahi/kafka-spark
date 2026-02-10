
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_history():
    """Charge l'historique d'agrégation"""
    history_path = "models/aggregation_history.json"
    
    if not os.path.exists(history_path):
        print(f"❌ Fichier d'historique introuvable: {history_path}")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


def plot_convergence(history):
    """Affiche le graphique de convergence"""
    
    if not history or len(history) == 0:
        print("⚠️  Aucune donnée d'historique disponible")
        return
    
    # Extraire les données
    rounds = [entry['round'] for entry in history]
    losses = [entry['avg_loss'] for entry in history]
    timestamps = [datetime.fromtimestamp(entry['timestamp']) for entry in history]
    num_nodes = [entry['num_nodes'] for entry in history]
    total_samples = [entry['total_samples'] for entry in history]
    
    # Créer la figure avec 2 sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Graphique 1: Convergence de la loss
    ax1.plot(rounds, losses, 'b-o', linewidth=2, markersize=8, label='Loss moyenne')
    ax1.set_xlabel('Round d\'agrégation', fontsize=12)
    ax1.set_ylabel('Loss (Binary Cross-Entropy)', fontsize=12)
    ax1.set_title('Convergence du Modèle Global - Federated Learning', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Ajouter les valeurs sur les points
    for i, (r, l) in enumerate(zip(rounds, losses)):
        ax1.annotate(f'{l:.4f}', 
                    xy=(r, l), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Graphique 2: Nombre de samples
    ax2.bar(rounds, total_samples, color='green', alpha=0.6, label='Total samples')
    ax2.set_xlabel('Round d\'agrégation', fontsize=12)
    ax2.set_ylabel('Nombre de samples', fontsize=12)
    ax2.set_title('Évolution du nombre de samples', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    
    # Ajouter les valeurs sur les barres
    for i, (r, s) in enumerate(zip(rounds, total_samples)):
        ax2.text(r, s, f'{s}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = "models/convergence_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {output_path}")
    
    # Afficher
    plt.show()


def print_summary(history):
    """Affiche un résumé textuel"""
    
    if not history or len(history) == 0:
        print("⚠️  Aucune donnée disponible")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 RÉSUMÉ DE L'AGRÉGATION FÉDÉRÉE")
    print(f"{'='*70}\n")
    
    print(f"Nombre total de rounds: {len(history)}")
    
    first = history[0]
    last = history[-1]
    
    print(f"\n📈 Progression:")
    print(f"   Loss initiale: {first['avg_loss']:.6f}")
    print(f"   Loss finale: {last['avg_loss']:.6f}")
    
    improvement = (first['avg_loss'] - last['avg_loss']) / first['avg_loss'] * 100
    print(f"   Amélioration: {improvement:.2f}%")
    
    print(f"\n📊 Détails par round:")
    print(f"{'Round':>7} | {'Timestamp':^19} | {'Loss':>10} | {'Nodes':>6} | {'Samples':>8}")
    print(f"{'-'*70}")
    
    for entry in history:
        timestamp = datetime.fromtimestamp(entry['timestamp'])
        print(f"{entry['round']:>7} | "
              f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
              f"{entry['avg_loss']:>10.6f} | "
              f"{entry['num_nodes']:>6} | "
              f"{entry['total_samples']:>8}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("\n🔍 Chargement de l'historique d'agrégation...\n")
    
    history = load_history()
    
    if history:
        print_summary(history)
        plot_convergence(history)
    else:
        print("\n⚠️  Lancez d'abord l'agrégateur pour générer des données !")
        print("   python cloud_aggregator/cloud_aggregator.py\n")
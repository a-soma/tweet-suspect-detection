"""
Étape 3 - Évaluation et rapport de performance
Auteur : Alassane SOMA | UV-BF | 2026
"""
import json
import sys

def afficher_resultats(results_path: str = 'models/results.json') -> None:
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    best = results.get('_best', 'N/A')
    print("\n" + "="*70)
    print("  RAPPORT D'EVALUATION — Detection de Tweets Suspects")
    print("  Auteur : Alassane SOMA | UV-BF | 2026")
    print("="*70)
    print(f"{'Modele':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'CV-F1':>12}")
    print("-"*70)
    for nom, r in results.items():
        if nom.startswith('_'): continue
        mark = " MEILLEUR" if nom == best else ""
        print(f"{nom:<28} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
              f"{r['recall']:>6.3f} {r['f1']:>6.3f} "
              f"{r['cv_f1_mean']:>6.3f}±{r['cv_f1_std']:.3f}{mark}")
    print("="*70)
    print(f"  Meilleur modele : {best}")
    print("="*70 + "\n")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'models/results.json'
    afficher_resultats(path)

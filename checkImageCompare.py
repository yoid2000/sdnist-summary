import json
import pandas as pd

resultsPath = "image_compare.json"
with open(resultsPath, 'r') as f:
    results = json.load(f)

'''
Root mean square error (RMSE)
    perfect is 0
    good
Peak signal-to-noise ratio (PSNR)
    perfect is infinity
    good
Structural Similarity Index (SSIM)
    perfect is 1
    good
Feature-based similarity index (FSIM)
    perfect is 1
    good
Information theoretic-based Statistic Similarity Measure (ISSM)
    perfect is 1
Signal to reconstruction error ratio (SRE)
    perfect is 1
Spectral angle mapper (SAM)
    perfect is 1
Universal image quality index (UIQ)
    perfect is 1
'''

visualCompare = [
    ["SynDiffix", 8],
    ["Aindo", 10],
    ["Anonos", 10],
    ["MostlyAI", 10],
    ["YData", 4],
    ["CTGAN", 5],
    ["mwem-pgm", 6],
    ["AIM", 7],
    ["Pategan", 1],
    ["Genetic", 6],
    ["Sarus", 4],
    ["CART", 10],
    ["K6-Anon", 7],
    ["PRAM", 4],
    ["SMOTE", 10],
    ["Sample40", 10],
]

best = results['same']

# ordered by smallest (worst) first
visualCompare.sort(key=lambda row: row[1])
print(visualCompare)
visual = [x[1] for x in visualCompare]
minV = min(visual)
maxV = max(visual)

def getScoreByMethod(method):
    scores = []
    for alg, val in visualCompare:
        scores.append(results[alg][method][method])
    return scores


for method in best.keys():
    print(method)
    # get method scores indexed in same order as visualCompare
    scores = getScoreByMethod(method)
    visual_series = pd.Series(visual)
    scores_series = pd.Series(scores)

    correlation = visual_series.corr(scores_series)
    print(f"correlation: {correlation:.4f}")


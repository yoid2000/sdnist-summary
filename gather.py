import os
import json
import statistics
import pytextable
import pandas as pd
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np

import pprint
pp = pprint.PrettyPrinter(indent=4)

RES_PATH = "https://htmlpreview.github.io/?https://github.com/yoid2000/sdnist-summary/blob/main/results/"
# If 'repo' is None, then product is considered proprietary
methodInfo = {
    "aim": {"name":"AIM",
            "repo":"https://github.com/opendp/smartnoise-sdk",
            "cite":"mckenna2022aim",
            "web":"https://opendp.org/",
            "org":"OpenDP",
            "tech":"Workload adaptive + DP",
            },
    "aindo-synth": {"name":"Aindo",
            "repo":None,
            "cite":None,
            "web":"https://www.aindo.com/",
            "org":"Aindo",
            "tech":"Generative model",
            },
    "Anonos Data Embassy SDK": {"name":"Anonos",
            "repo":None,
            "cite":None,
            "web":"https://www.anonos.com/",
            "org":"Anonos",
            "tech":"Generative model",
            },
    "cart": {"name":"CART",
            "repo":"https://CRAN.R-project.org/package=synthpop",
            "cite":"nowok2016synthpop",
            "web":"https://synthpop.org.uk/",
            "org":"Synthpop",
            "tech":"Decision trees",
            },
    "Genetic SD": {"name":"Genetic",
            "repo":"https://github.com/giusevtr/private_gsd",
            "cite":"liu2023generating",
            "web":"https://github.com/giusevtr/private_gsd",
            "org":"See pub",
            "tech":"Approximate DP",
            },
    "kanonymity": {"name":"K6-Anon",
            "repo":"https://github.com/sdcTools/sdcMicro",
            "cite":"templ2015statistical",
            "web":"https://github.com/sdcTools/sdcMicro",
            "org":"sdcMicro",
            "tech":"K-anonymity",
            },
    "pram": {"name":"PRAM",
            "repo":"https://github.com/sdcTools/sdcMicro",
            "cite":"meindl2019feedback",
            "web":"https://github.com/sdcTools/sdcMicro",
            "org":"sdcMicro",
            "tech":"Random value changes",
            },
    "MostlyAI SD": {"name":"MostlyAI",
            "repo":None,
            "cite":None,
            "web":"https://mostly.ai/",
            "org":"MostlyAI",
            "tech":"Generative model",
            },
    "MWEM+PGM": {"name":"mwem-pgm",
            "repo":None,
            "cite":"mckenna2019graphical",
            "web":"https://dream.cs.umass.edu/",
            "org":"See pub",
            "tech":"Graphical models + DP",
            },
    "pategan": {"name":"Pategan",
            "repo":"https://github.com/PerceptionLab-DurhamUniversity/pategan",
            "cite":"jordon2018pate",
            "web":"https://github.com/PerceptionLab-DurhamUniversity/pategan",
            "org":"See pub",
            "tech":"Generative model + DP",
            },
    "Sarus SDG": {"name":"Sarus",
            "repo":None,
            "cite":"canale2022generative",
            "web":"https://www.sarus.tech/",
            "org":"Sarus",
            "tech":"Generative model + DP",
            },
    "ctgan": {"name":"CTGAN",
            "repo":"https://github.com/sdv-dev/SDV",
            "cite":"xu2019modeling",
            "web":"https://sdv.dev/",
            "org":"SDV",
            "tech":"Generative model",
            },
    "smote": {"name":"SMOTE",
            "repo":"https://github.com/ut-dallas-dspl-lab/AI-Fairness",
            "cite":"zhou2023improving",
            "web":"https://github.com/ut-dallas-dspl-lab/AI-Fairness",
            "org":"See pub",
            "tech":"Minority oversampling",
            },
    "subsample_40pcnt": {"name":"Sample40",
            "repo":None,
            "cite":"acsBasics2021",
            "web":"https://www.census.gov/content/dam/Census/library/publications/2021/acs/acs_pums_handbook_2021_ch01.pdf",
            "org":"US Census",
            "tech":"Simple sampling",
            },
    "SynDiffix": {"name":"SynDiffix",
            "repo":"https://github.com/diffix/syndiffix",
            "cite":"francis2023syndiffix",
            "web":"https://www.open-diffix.org/",
            "org":"Open Diffix",
            "tech":"Anonymous decision trees",
            },
    "YData Fabric Synthesizers": {"name":"YData",
            "repo":"https://github.com/ydataai/ydata-synthetic",
            "cite":None,
            "web":"https://ydata.ai/",
            "org":"YData",
            "tech":"Generative model",
            },
}

keyOrder = [
    "SynDiffix",
    "Aindo",
    "Anonos",
    "MostlyAI",
    "YData",
    "CTGAN",
    "mwem-pgm",
    "AIM",
    "Pategan",
    "Genetic",
    "Sarus",
    "CART",
    "K6-Anon",
    "PRAM",
    "SMOTE",
    "Sample40",
]

'''
Light blue: #A6CEE3
Dark blue: #1F78B4
Light green: #B2DF8A
Dark green: #33A02C
Light red: #FB9A99
Dark red: #E31A1C
Light orange: #FDBF6F
Dark orange: #FF7F00
Light purple: #CAB2D6
Dark purple: #6A3D9A
Light brown: #FFFF99
Dark brown: #B15928
'''

pltColors = {
    "SynDiffix": "blue",
    "Aindo": "red",
    "Anonos": "red",
    "MostlyAI": "red",
    "YData": "red",
    "CTGAN": "red",
    "mwem-pgm": "ForestGreen",
    "AIM": "ForestGreen",
    "Pategan": "ForestGreen",
    "Genetic": "YellowGreen",
    "Sarus": "YellowGreen",
    "CART": "SkyBlue",
    "K6-Anon": "pink",
    "PRAM": "Tan",
    "SMOTE": "indigo",
    "Sample40": "indigo",
}

dashes = {
    "SynDiffix": (0,()),
    "Aindo": (0,()),
    "Anonos": (0,(3,3)),
    "MostlyAI": (0,(1,1)),
    "YData": (0,(1,1,3,1)),
    "CTGAN": (0,(1,1,1,1,3,1)),
    "mwem-pgm": (0,()),
    "AIM": (0,(3,3)),
    "Pategan": (0,(1,1)),
    "Genetic": (0,(1,1,3,1)),
    "Sarus": (0,(1,1,1,1,3,1)),
    "CART": (0,()),
    "K6-Anon": (0,()),
    "PRAM": (0,()),
    "SMOTE": (0,()),
    "Sample40": (0,(3,3)),
}

latexColors = {
    "SynDiffix": "blue",
    "Aindo": "red",
    "Anonos": "red",
    "MostlyAI": "red",
    "YData": "red",
    "CTGAN": "red",
    "mwem-pgm": "ForestGreen",
    "AIM": "ForestGreen",
    "Pategan": "ForestGreen",
    "Genetic": "YellowGreen",
    "Sarus": "YellowGreen",
    "CART": "SkyBlue",
    "K6-Anon": "pink",
    "PRAM": "Tan",
    "SMOTE": "Sepia",
    "Sample40": "Sepia",
}


markers = {
    "SynDiffix": "o",
    "Aindo": ",",
    "Anonos": "v",
    "MostlyAI": "^",
    "YData": "<",
    "CTGAN": ">",
    "mwem-pgm": "1",
    "AIM": "2",
    "Pategan": "D",
    "Genetic": "d",
    "Sarus": "p",
    "CART": "h",
    "K6-Anon": "H",
    "PRAM": "8",
    "SMOTE": "+",
    "Sample40": "x",
}

class ReportJsonReader:
    def __init__(self, resultsDir, dir):
        self.dirPath = os.path.join(resultsDir, dir)
        resPath = os.path.join(self.dirPath, 'report.json')
        with open(resPath, 'r') as f:
            self.algData = json.load(f)
        self.algName = self.algData['data_description']['deid']['labels']['algorithm name']
        self.team = self.algData['data_description']['deid']['labels']['team']
        self.info = methodInfo[self.algName]
        self.name = self.info['name']
        self.repo = self.info['repo']
        self.cite = self.info['cite']
        self.web = self.info['web']
        self.tech = self.info['tech']
        self.org = self.info['org']
        self.dir_name = dir
        self.features = self.algData['data_description']['deid']['labels']['features list'].split(',')
        self.ncol = len(self.features)
        self.nrows = self.algData['data_description']['target']['records']
        self.dp = True if self.algData['data_description']['deid']['labels']['privacy category'] == 'dp' else False
        self.epsilon = self.algData['data_description']['deid']['labels']['epsilon'] if self.dp is True else None
        self.recordsMatched = self.algData['unique_exact_matches']['records matched in target data']
        self.recordsMatchedPercent = self.algData['unique_exact_matches']['percent records matched in target data']
        self._read_apparent_match()
        self._get_uni_stats()
        self._get_corr_stats()
        self.sampling_equiv = self.algData['k_marginal']['k_marginal_synopsys']['sub_sampling_equivalent']
        self.triple_score = self.algData['k_marginal']['k_marginal_synopsys']['k_marginal_score']
        self._get_regression_stats()
        self.pmse = self.algData['propensity mean square error']['pmse_score']
    
    def _get_regression_stats(self):
        reg_info = {'total_population':{'label':'all', 'count':23006},
                    'white_men':{'label':'WM', 'count':6463},
                    'white_women':{'label':'WW', 'count':6505},
                    'black_men':{'label':'BM', 'count':2720},
                    'black_women':{'label':'BW', 'count':3366},
                    'asian_men':{'label':'AM', 'count':914},
                    'asian_women':{'label':'AW', 'count':982},
                    'aiannh_men':{'label':'NM', 'count':376},
                    'aiannh_women':{'label':'NW', 'count':395},
                    }
        lr = self.algData['linear_regression']
        count_all = reg_info['total_population']['count']
        stats = []
        for group, data in lr.items():
            slope_orig = data['target_regression_slope_and_intercept'][0]
            slope_syn = data['deidentified_regression_slope_and_intercept'][0]
            percent = f"{round((reg_info[group]['count'] * 100) / count_all)}%"
            label = f"{reg_info[group]['label']} {percent}"
            error = abs(slope_orig - slope_syn)
            error = error if error != 0 else 0.005   # deal with log scale plot
            stats.append({'group': group,
                          'slope_orig': slope_orig,
                          'slope_syn': slope_syn,
                          'count': reg_info[group]['count'],
                          'group_label': reg_info[group]['label'],
                          'label': label,
                          'error': error, 
                          'percent': percent,
                          })
        self.df_linear_regression = pd.DataFrame(stats)
        self.df_linear_regression = self.df_linear_regression.sort_values(by='count', ascending=False)

    def _get_corr_stats(self):
        self.corr_diffs = []
        csvPath = self.algData['Correlations']['kendall correlation difference']['correlation_difference']
        df = self._get_csv(csvPath)
        cols = list(df.columns)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                if pd.isna(df.iloc[i,j]) is False:
                    self.corr_diffs.append(abs(df.iloc[i,j]))
        self.corr_num_measures = len(self.corr_diffs)
        self.corr_median = statistics.median(self.corr_diffs)
        self.corr_mean = statistics.mean(self.corr_diffs)

    def _get_uni_stats(self):
        self.uni_comp_error = []
        self.uni_abs_error = []
        for info in self.algData['Univariate']['counts'].values():
            if 'counts' in info:
                df = self._get_csv(info['counts'])
                self._update_uni(df)
            else:
                for subinfo in info.values():
                    df = self._get_csv(subinfo['counts'])
                    self._update_uni(df)
        self.uni_num_measures = len(self.uni_abs_error)
        self.uni_median_comp_error = statistics.median(self.uni_comp_error)
        self.uni_mean_comp_error = statistics.mean(self.uni_comp_error)

    def _update_uni(self, df):
        fact = 2.5 if self.name == 'Sample40' else 1.0
        df['abs'] = abs(df['count_target'] - (df['count_deidentified'] * fact))
        self.uni_abs_error += df['abs'].tolist()
        df['rel'] = (df['abs'] / df['count_target']) * 100
        df['comp'] = df[['abs','rel']].min(axis=1)
        self.uni_comp_error += df['comp'].tolist()

    def _get_csv(self, csvPath):
        path = os.path.join(self.dirPath, csvPath)
        return(pd.read_csv(path))

    def _read_apparent_match(self):
        ''' df contains one row per record whose quasi-identifiers are
            a unique match in both the original and synthetic data
        '''
        self.df_app_match = self._get_csv(self.algData['apparent_match_distribution']['unique_matched_percents'])
        self.qiRecordsMatched = len(self.df_app_match)
        if self.qiRecordsMatched == 0:
            self.qiAverageInference = 0.0
        else:
            self.qiAverageInference = (self.df_app_match['percent_match'].mean())/100
        pass

def _fi(this, syndiffix):
    # returns a positive improvement factor if syndiffix better than (<) this,
    # otherwise returns a negative improvement factor
    if syndiffix <= this:
        return this / syndiffix
    else:
        return -1 * (syndiffix / this)

class GatherResults:
    def __init__(self, resultsDir, outDir):
        self.resultsDir = resultsDir
        self.outDir = outDir
        self.algsBase = {}
        self.algDirs = [d for d in os.listdir(self.resultsDir) if os.path.isdir(os.path.join(self.resultsDir, d))]
        self.res = {}
        for dir in self.algDirs:
            rjr = ReportJsonReader(self.resultsDir, dir)
            self.res[rjr.name] = rjr

    def makePairTriplePlot(self):
        labels = []
        corr_improve_labels = []
        triple_improve_labels = []
        corr_values = []
        triple_values = []
        triple_equivs = []
        triple_score_labels = []
        colors = []
        syndiffix_corr_gap = self.res['SynDiffix'].corr_mean - 0
        syndiffix_triple_gap = 100 - self.res['SynDiffix'].sampling_equiv
        for alg in keyOrder:
            rjr = self.res[alg]
            labels.append(alg)
            corr_values.append(rjr.corr_diffs)
            this_corr_gap = rjr.corr_mean - 0
            corr_improve = _fi(this_corr_gap, syndiffix_corr_gap)
            corr_improve_labels.append(f"{corr_improve:.1f}x")
            triple_values.append(rjr.triple_score)
            triple_equivs.append(rjr.sampling_equiv)
            this_triple_gap = 100 - rjr.sampling_equiv
            triple_improve = _fi(this_triple_gap, syndiffix_triple_gap)
            triple_improve_labels.append(f"{triple_improve:.1f}x")
            triple_score_labels.append(f"({rjr.triple_score})")
            colors.append(pltColors[alg])

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # Create DataFrame for seaborn
        df_corr = pd.DataFrame({'Label': [label for label, sublist in zip(labels, corr_values) for _ in sublist],
                    'Values': [val for sublist in corr_values for val in sublist]})
        df = pd.DataFrame({
            'labels': labels,
            'triple_equiv': triple_equivs,
            'triple_score_labels': triple_score_labels,
            'triple_improv': triple_improve_labels,
            'colors': colors
        })

        # Plot corr_values
        sns.boxplot(ax=axes[0], y='Label', x='Values', hue='Label', data=df_corr, orient='h', palette=pltColors, legend = False)
        axes[0].set_xlabel('Pairwise Correlation Difference', fontsize=14)
        axes[0].set_xscale('log')
        axes[0].set_ylabel('')
        axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

        ax2 = axes[0].twinx()
        ax2.set_ylim(axes[0].get_ylim())
        ax2.set_yticks(range(len(corr_improve_labels)))
        ax2.set_yticklabels(corr_improve_labels)

        # Plot triple_values
        barplot = sns.barplot(x='triple_equiv', y='labels', data=df, palette=pltColors, orient='h', ax=axes[1])
        for i in range(df.shape[0]):
            barplot.text(df.triple_equiv.iloc[i] + 6, i, df.triple_score_labels.iloc[i], color='black', ha="right", va="center")
        barplot.set_yticklabels(df.labels)
        axes[1].set_xlim([0,70])
        axes[1].set_xlabel('Sampling Equivalent % (3-Marginal Score)', fontsize=14)
        axes[1].set_ylabel('')

        if False:
            ax3 = axes[1].twinx()
            ax3.set_ylim(axes[1].get_ylim())
            ax3.set_yticks(range(len(triple_improve_labels)))
            ax3.set_yticklabels(triple_improve_labels)

        plt.tight_layout()
        outPath = os.path.join(self.outDir, 'pairTripleStats.png')
        plt.savefig(outPath)
        plt.close()

    def makeUniPlot(self):
        labels = []
        improve_labels = []
        abs_values = []
        comp_values = []
        syndiffix_gap = self.res['SynDiffix'].uni_mean_comp_error - 0
        for alg in keyOrder:
            rjr = self.res[alg]
            labels.append(alg)
            abs_values.append(rjr.uni_abs_error)
            comp_values.append(rjr.uni_comp_error)
            this_gap = rjr.uni_mean_comp_error - 0
            improve = _fi(this_gap, syndiffix_gap)
            improve_labels.append(f"{improve:.1f}x")

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # Create DataFrame for seaborn
        df1 = pd.DataFrame({'Label': [label for label, sublist in zip(labels, abs_values) for _ in sublist],
                    'Values': [val for sublist in abs_values for val in sublist]})
        df2 = pd.DataFrame({'Label': [label for label, sublist in zip(labels, comp_values) for _ in sublist],
                    'Values': [val for sublist in comp_values for val in sublist]})

        # Plot abs_values
        sns.boxplot(ax=axes[0], y='Label', x='Values', hue='Label', data=df1, orient='h', palette=pltColors, legend = False)
        axes[0].set_xlabel('Univariate Absolute Error', fontsize=14)
        axes[0].set_xscale('log')
        axes[0].set_ylabel('')
        axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

        # Plot comp_values
        sns.boxplot(ax=axes[1], y='Label', x='Values', hue='Label', data=df2, orient='h', palette=pltColors, legend = False)
        axes[1].set_xlabel('Univariate Composite Error', fontsize=14)
        axes[1].set_xscale('log')
        axes[1].set_yticklabels([])
        axes[1].set_yticks([])
        ax2 = axes[1].twinx()
        ax2.set_ylim(axes[1].get_ylim())
        ax2.set_yticks(range(len(improve_labels)))
        ax2.set_yticklabels(improve_labels)
        axes[1].set_ylabel('')
        axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

        plt.tight_layout()
        outPath = os.path.join(self.outDir, 'uniStats.png')
        plt.savefig(outPath)
        plt.close()

    def makePmseTable(self):
        header = [" "," ",
                  "\\multicolumn{2}{c}{PMSE (\\S\\ref{sec:pmse})}",
                  ]
        addHeader = " & & pmse & imp \\\\"
        alignReplace = ['lll', 'llr@{\hskip 6pt}r']
        label="tab:pmse"
        caption="Summary table for Propensity MSE."
        body = []
        for alg in keyOrder:
            rjr = self.res[alg]
            row = []
            row.append("\\cellcolor{" + latexColors[alg] + "}")
            row.append("\\href{" + RES_PATH + rjr.dir_name +"/report.html}{" + rjr.name + "}")

            row.append(f"{rjr.pmse:.4f}")
            syndiffix_gap = self.res['SynDiffix'].pmse - 0
            this_gap = rjr.pmse - 0
            improve = _fi(this_gap, syndiffix_gap)
            row.append(f"{improve:.1f}x")

            body.append(row)
        outPath = os.path.join(self.outDir, 'pmseTable.tex')
        with open(outPath, 'w') as f:
            table = pytextable.tostring(body, header=header, label=label, caption=caption,alignment='l')
            table = table.replace(alignReplace[0], alignReplace[1])
            table = table.replace("\midrule", '\n' + addHeader + '\n' + "\midrule")
            f.write(table)

    def makeAccuracyTable(self):
        header = [" "," ",
                  "\\multicolumn{3}{c}{Univariate (\\S\\ref{sec:univariate})}",
                  "\\multicolumn{3}{c}{Correlation (\\S\\ref{sec:pairs})}",
                  "\\multicolumn{3}{c}{3-marginals (\\S\\ref{sec:triples})}",
                  ]
        addHeader = " & & N & mean & imp & N & mean & imp & score & samp & imp \\\\"
        alignReplace = ['lllllllllll', 'llrlr@{\hskip 10pt}r@{\hskip 6pt}l@{\hskip 6pt}r@{\hskip 10pt}r@{\hskip 6pt}r@{\hskip 6pt}r']
        label="tab:accuracy"
        caption="Summary table for low-dimensional accuracy measures."
        body = []
        for alg in keyOrder:
            rjr = self.res[alg]
            row = []
            row.append("\\cellcolor{" + latexColors[alg] + "}")
            row.append("\\href{" + RES_PATH + rjr.dir_name +"/report.html}{" + rjr.name + "}")

            row.append(str(rjr.uni_num_measures))
            row.append(f"{rjr.uni_mean_comp_error:.1f}")
            syndiffix_gap = self.res['SynDiffix'].uni_mean_comp_error - 0
            this_gap = rjr.uni_mean_comp_error - 0
            improve = _fi(this_gap, syndiffix_gap)
            row.append(f"{improve:.1f}x")

            row.append(str(rjr.corr_num_measures))
            row.append(f"{rjr.corr_mean:.4f}")
            syndiffix_corr_gap = self.res['SynDiffix'].corr_mean - 0
            this_corr_gap = rjr.corr_mean - 0
            corr_improve = _fi(this_corr_gap,  syndiffix_corr_gap)
            row.append(f"{corr_improve:.1f}x")

            row.append(rjr.triple_score)
            row.append(f"{rjr.sampling_equiv}\%")
            syndiffix_samp_gap = 100 - self.res['SynDiffix'].sampling_equiv
            this_samp_gap = 100 - rjr.sampling_equiv
            samp_improve = _fi(this_samp_gap,  syndiffix_samp_gap)
            row.append(f"{samp_improve:0.1f}x")

            body.append(row)
        outPath = os.path.join(self.outDir, 'accTable.tex')
        with open(outPath, 'w') as f:
            table = pytextable.tostring(body, header=header, label=label, caption=caption,alignment='l')
            table = table.replace(alignReplace[0], alignReplace[1])
            table = table.replace("\midrule", '\n' + addHeader + '\n' + "\midrule")
            f.write(table)

    def makePrivacyTable(self):
        header = [" "," ", "\\multicolumn{2}{c}{Full Match (\\S\\ref{sec:privacy})}",  "\\multicolumn{3}{c}{QI Match (\\S\\ref{sec:privacy})}",]
        addHeader = " & & count & \\quad \% & prec & cov & count \\\\"
        alignReplace = ['lllllll', 'llrl@{\hskip 10pt}r@{\hskip 6pt}l@{\hskip 6pt}l']
        label="tab:privacy"
        caption="Summary table for privacy measures."
        body = []
        for alg in keyOrder:
            rjr = self.res[alg]
            row = []
            row.append("\\cellcolor{" + latexColors[alg] + "}")
            row.append("\\href{" + RES_PATH + rjr.dir_name +"/report.html}{" + rjr.name + "}")
            row.append(str(rjr.recordsMatched))
            percent = f"{((rjr.recordsMatched / rjr.nrows) * 100):.2f}"
            row.append("\\quad" + percent)
            # precision
            row.append(f"{rjr.qiAverageInference:.2f}")
            # recall
            row.append(f"{(rjr.qiRecordsMatched / rjr.nrows):.3f}")
            # count
            row.append(f"{rjr.qiRecordsMatched}")
            body.append(row)
        outPath = os.path.join(self.outDir, 'privTable.tex')
        with open(outPath, 'w') as f:
            table = pytextable.tostring(body, header=header, label=label, caption=caption,alignment='l')
            table = table.replace(alignReplace[0], alignReplace[1])
            table = table.replace("\midrule", '\n' + addHeader + '\n' + "\midrule")
            f.write(table)

    def makeInfoTable(self):
        header = [" ", "Algorithm", "Tech", "Org", "Cols", "\\thinspace$\\epsilon$\\qquad\\qquad", "Cite", "Repo"]
        label="tab:infotable"
        caption="Set of compared algorithms, showing the number of columns synthesized (out of 24). Labels link to the SDNIST report. Algorithms without an epsilon do not use differential privacy. Algorithms without both a repo and citation are proprietary."
        body = []
        for alg in keyOrder:
            rjr = self.res[alg]
            row = []
            row.append("\\cellcolor{" + latexColors[alg] + "}")
            row.append("\\href{" + RES_PATH + rjr.dir_name +"/report.html}{" + rjr.name + "}")
            row.append(rjr.tech)
            row.append("\\href{" + rjr.web +"}{" + rjr.org + "}")
            row.append(rjr.ncol)
            row.append(rjr.epsilon if rjr.epsilon is not None else '')
            row.append("\\cite{" + rjr.cite +"}" if rjr.cite is not None else '')
            row.append("\\href{" + rjr.repo +"}{link}" if rjr.repo is not None else '')
            body.append(row)
        outPath = os.path.join(self.outDir, 'infoTable.tex')
        with open(outPath, 'w') as f:
            f.write(pytextable.tostring(body, header=header, label=label, caption=caption,alignment='l'))

    def makeRegressionPlot(self):
        dataframes = {}
        sd_mean_gap = self.res['SynDiffix'].df_linear_regression['error'].mean() - 0
        for alg in keyOrder:
            rjr = self.res[alg]
            this_mean_gap = rjr.df_linear_regression['error'].mean() - 0
            improve = _fi(this_mean_gap, sd_mean_gap)
            tag = '   ' if improve >= 0 else '* '
            new_label = f"{tag}{alg} ({improve:.1f}x)"
            dataframes[alg] = {'df':rjr.df_linear_regression, 'label':new_label}
        plt.figure(figsize=(8, 2.7))
        for alg, stuff in dataframes.items():
            df = stuff['df']
            label = stuff['label']
            sns.lineplot(x=df['label'], y=df['error'], color=pltColors[alg], marker=markers[alg], linestyle=dashes[alg], label=label)
        plt.yscale('log')
        plt.xlabel('Race/Sex Group, Percentage of Population')
        plt.ylabel('Error')
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=8)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0, fontsize=7, ncol=2)
        plt.tight_layout()
        outPath = os.path.join(self.outDir, 'regression.png')
        plt.savefig(outPath)
        plt.close()

    def makeAttackScatter(self):
        data = []
        for alg in keyOrder:
            rjr = self.res[alg]
            data.append([rjr.name,
                         rjr.qiAverageInference,
                         (rjr.qiRecordsMatched / rjr.nrows)])
        df = pd.DataFrame(data, columns=['name','Precision','Coverage'])

        #plt.figure(figsize=(8, 6))  # Optional: Set the figure size
        fig, ax = plt.subplots()
        texts = []
        for i, row in df.iterrows():
            plt.scatter(row['Coverage'], row['Precision'], label=row['name'], s=100, c=pltColors[row['name']], marker=markers[row['name']])
            if row['Precision'] < 0.65 and row['Precision'] > 0.53 and row['Coverage'] > 0.03 and row['Coverage'] < 0.065:
                label = ''
            else:
                label = row['name']
            texts.append(plt.text(row['Coverage'],row['Precision'],label))
        adjust_text(texts)
        #for i, row in df.iterrows():
            #plt.text(row['Coverage'], row['Precision'], row['name'], ha='left', va='bottom')

        plt.xlabel('Coverage', fontsize=16)
        plt.ylabel('Precision', fontsize=16)
        plt.legend(loc='lower right', ncol=2, fontsize='small')
        outPath = os.path.join(self.outDir, 'attackPrecCov.png')
        plt.savefig(outPath)
        plt.close()

if __name__ == "__main__":
    gr = GatherResults('results', 'outputs')
    gr.makeRegressionPlot()
    gr.makeInfoTable()
    gr.makePrivacyTable()
    gr.makeAccuracyTable()
    gr.makePmseTable()
    gr.makeAttackScatter()
    gr.makeUniPlot()
    gr.makePairTriplePlot()
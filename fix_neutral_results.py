import pandas as pd
results_files = [
    # "eval_results_sc1_binary_20240922_195817.xlsx",
    "eval_results_sc1_binary_20240923_025316.xlsx",
    "eval_results_sc1_binary_20240923_221728.xlsx",
    "eval_results_sc1_binary_20240925_002013.xlsx",
    # "eval_results_sc1_neutral_20240922_195817.xlsx",
    "eval_results_sc1_neutral_20240923_025316.xlsx",
    "eval_results_sc2_binary_20240923_032713.xlsx",
    "eval_results_sc2_binary_20240923_225509.xlsx",
    "eval_results_sc2_binary_20240925_003659.xlsx",
    "eval_results_sc2_neutral_20240923_032713.xlsx",
    "eval_results_sc3_binary_20240923_032652.xlsx",
    "eval_results_sc3_binary_20240923_225638.xlsx",
    "eval_results_sc3_binary_20240925_004518.xlsx",
    "eval_results_sc3_neutral_20240923_032652.xlsx",
    "eval_results_sc4_binary_20240923_033502.xlsx",
    "eval_results_sc4_binary_20240923_225045.xlsx",
    "eval_results_sc4_binary_20240925_005645.xlsx",
    "eval_results_sc4_neutral_20240923_033502.xlsx",
    "eval_results_sc5_binary_20240923_032552.xlsx",
    "eval_results_sc5_binary_20240923_223755.xlsx",
    "eval_results_sc5_binary_20240925_004204.xlsx",
    "eval_results_sc5_neutral_20240923_032552.xlsx",
    # "eval_results_ablation_binary_20240923_050130.xlsx",
    "eval_results_ablation_binary_20240924_035111.xlsx",
    "eval_results_ablation_binary_20240925_034658.xlsx",
    "eval_results_ablation_binary_20240925_035448.xlsx",
    "eval_results_ablation_neutral_20240923_050130.xlsx",
    "eval_results_sc1_binary_20241015_032359.xlsx",
    "eval_results_sc1_neutral_20241015_032359.xlsx",
    "eval_results_sc5_binary_20241015_041210.xlsx",
    "eval_results_sc5_neutral_20241015_041210.xlsx",
    "eval_results_sc3_binary_20241015_050043.xlsx",
    "eval_results_sc3_neutral_20241015_050043.xlsx",
    "eval_results_sc2_binary_20241015_052152.xlsx",
    "eval_results_sc2_neutral_20241015_052152.xlsx",
    "eval_results_sc4_binary_20241015_054259.xlsx",
    "eval_results_sc4_neutral_20241015_054259.xlsx",
    "eval_results_sc1_binary_20241018_001146.xlsx",
    "eval_results_sc1_neutral_20241018_001146.xlsx",
    "eval_results_sc2_binary_20241018_004109.xlsx",
    "eval_results_sc2_neutral_20241018_004109.xlsx",
    "eval_results_sc3_binary_20241018_004827.xlsx",
    "eval_results_sc3_neutral_20241018_004827.xlsx",
    "eval_results_sc5_binary_20241018_004943.xlsx",
    "eval_results_sc5_neutral_20241018_004943.xlsx",
    "eval_results_sc4_binary_20241018_005119.xlsx",
    "eval_results_sc4_neutral_20241018_005119.xlsx",
    "eval_results_sc1_binary_20241021_000514.xlsx",
    "eval_results_sc1_neutral_20241021_000514.xlsx",
    "eval_results_sc3_binary_20241021_002410.xlsx",
    "eval_results_sc3_neutral_20241021_002410.xlsx",
    "eval_results_sc5_binary_20241021_002545.xlsx",
    "eval_results_sc5_neutral_20241021_002545.xlsx",
    "eval_results_sc2_binary_20241021_003630.xlsx",
    "eval_results_sc2_neutral_20241021_003630.xlsx",
    "eval_results_sc4_binary_20241021_004658.xlsx",
    "eval_results_sc4_neutral_20241021_004658.xlsx",
]


def load_results(file_path):
    return pd.read_excel(file_path)


def load_all_results(file_paths):
    return [
        (file_path, load_results(file_path))
        for file_path in file_paths
        if "sc3_neutral" in file_path
    ]

results = load_all_results(results_files)

for file_path, result in results:
    result["Scenario"] = "Scenario 3"
    result.to_excel(f"new_results/{file_path}", index=False)
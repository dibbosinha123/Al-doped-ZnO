# ==========================================================
# Al-Doped ZnO Conductivity Prediction
# Gradient Boosting + GridSearchCV + SHAP
# ==========================================================

!pip install shap openpyxl -q

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys
import sklearn


from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    LeaveOneOut,
    cross_val_score,
    cross_val_predict
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.ensemble import GradientBoostingRegressor

# ==========================================================
# SOFTWARE VERSIONS
# ==========================================================

print("Python:", sys.version)
print("scikit-learn:", sklearn.__version__)
print("SHAP:", shap.__version__)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    "AZO_dataset.csv",
    encoding="utf-8-sig"
)

print("Original Shape:", df.shape)

# ==========================================================
# CLEAN DATA
# ==========================================================

df = df.replace("NAN", np.nan)
df = df.replace("nan", np.nan)

numeric_cols = [
    'Al_at%',
    'DepTemp',
    'AnnealTemp',
    'Thickness_nm',
    'CarrierConc_cm-3',
    'Bandgap_eV',
    'Temp_(K)',
    'Mobility_cm2v-1s-1',
    'Crystallite_size'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ==========================================================
# Improved Physics-Informed Conductivity Model
# ==========================================================

def calculate_n_type_conductivity_ZnO(
    bandgap,
    doping_percent,
    mobility_dataset,
    mobility_median,
    carrier_conc=None,
    temperature=300
):

    q = 1.602e-19
    k_B = 8.617333262e-5

    # --------------------------------------------------
    # Handle missing values
    # --------------------------------------------------

    C = 0 if pd.isna(doping_percent) else doping_percent
    T = 300 if pd.isna(temperature) else temperature


    # --------------------------------------------------
    # Carrier concentration
    # --------------------------------------------------

    # --------------------------------------------------
    # Carrier concentration
    # --------------------------------------------------

    # Zn atomic density (cm^-3)
    NZn = 4.2e22

    # Activation efficiency
    eta = 0.047

    # Nominal Al donor concentration
    NAl = (C / 100.0) * NZn

    # Model 1 donor concentration
    n_model1 = eta * NAl

    # Use measured carrier concentration if available
    if pd.notna(carrier_conc):

        n = carrier_conc

    else:

        n = n_model1

    # --------------------------------------------------
    # Mobility
    # --------------------------------------------------

    if pd.notna(mobility_dataset):

        # Use experimentally measured Hall mobility directly
        mu = mobility_dataset

    else:

        # Estimate missing mobility from dataset median
        mu = mobility_median

        # Mild structural corrections for estimated mobility only

        temp_factor = (300 / T) ** 0.5
        mu *= temp_factor
        # -----------------------------------
        # Ionized impurity scattering
        # -----------------------------------
        # Characteristic donor concentration (cm^-3)
        Nref = 1e20

        # Exponent
        m = 1.0

        mu /= (
            1 +
            (n_model1 / Nref)**m
        )

    # --------------------------------------------------
    # Physical limits
    # --------------------------------------------------

    mu = np.clip(mu, 1, 120)

    n = np.clip(n, 1e15, 1e22)

    # --------------------------------------------------
    # Conductivity
    # --------------------------------------------------

    sigma = q * n * mu * 100

    return sigma


# ==========================================================
# Fill missing mobility values
# ==========================================================

mobility_median = df["Mobility_cm2v-1s-1"].median()


# ==========================================================
# Calculate conductivity
# ==========================================================

df["Conductivity_Sm"] = df.apply(
    lambda row:

    calculate_n_type_conductivity_ZnO(

        bandgap=row["Bandgap_eV"],

        doping_percent=row["Al_at%"],

        mobility_dataset=row["Mobility_cm2v-1s-1"],

        mobility_median=mobility_median,

        carrier_conc=row["CarrierConc_cm-3"],

        temperature=row["Temp_(K)"]

    )

    if (
        pd.notna(row["Bandgap_eV"])
        and pd.notna(row["Al_at%"])
    )

    else np.nan,

    axis=1
)

df["Cond_Model1"] = df.apply(
    lambda row:
    calculate_n_type_conductivity_ZnO(
        bandgap=row["Bandgap_eV"],
        doping_percent=row["Al_at%"],
        mobility_dataset=row["Mobility_cm2v-1s-1"],
        mobility_median=mobility_median,
        carrier_conc=None,
        temperature=row["Temp_(K)"]
    ),
    axis=1
)

# ==========================================================
# CORRELATION ANALYSIS
# ==========================================================

corr = df[
    [
        "DepTemp",
        "AnnealTemp",
        "Mobility_cm2v-1s-1",
        "CarrierConc_cm-3",
        "Conductivity_Sm"
    ]
].corr()

print("\nCorrelation Matrix")
print(corr.round(3))


# ==========================================================
# REMOVE INVALID TARGETS
# ==========================================================

df = df.dropna(
    subset=[
        "Bandgap_eV",
        "Al_at%"
    ]
)

print("Remaining Samples:", len(df))

# ==========================================================
# FEATURES
# ==========================================================

features = [
    'Al_at%',
    'DepTemp',
    'AnnealTemp',
    'Thickness_nm',

    'Bandgap_eV',
    'Temp_(K)',
    'Crystallite_size'
]

features = [f for f in features if f in df.columns]

X = df[features]


# log conductivity

y = np.log10(df["Conductivity_Sm"])

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

imputer = SimpleImputer(strategy='median')

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=features
)
# Reset indices

X = X.reset_index(drop=True)

y = y.reset_index(drop=True)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Train Samples:", len(X_train))
print("Test Samples :", len(X_test))

# ==========================================================
# HYPERPARAMETER SEARCH
# ==========================================================

param_grid={

'n_estimators':[300,500,700],

'learning_rate':[0.01,0.03,0.05],

'max_depth':[2,3,4],

'subsample':[0.75,0.85,1.0],

'min_samples_split':[2,4,6],

'min_samples_leaf':[1,2,4],

'max_features':['sqrt',None]

}

gbr = GradientBoostingRegressor(
    random_state=42
)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    scoring='r2',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

# ==========================================================
# BEST MODEL
# ==========================================================

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest CV R²:")
print(round(grid.best_score_,4))

# ==========================================================
# CROSS VALIDATION PERFORMANCE
# ==========================================================

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=cv,
    scoring='r2'
)

print("\n5-Fold Cross Validation")

print("Fold Scores:")
print(cv_scores)

print("\nMean CV R²:")
print(round(cv_scores.mean(),4))

print("Std CV R²:")
print(round(cv_scores.std(),4))
# ==========================================================
# LEAVE-ONE-OUT CROSS VALIDATION (LOOCV)
# ==========================================================

print("\n")
print("="*60)
print("LEAVE-ONE-OUT CROSS VALIDATION (LOOCV)")
print("="*60)

loo = LeaveOneOut()

# Predict every sample using LOOCV
loo_pred = cross_val_predict(
    best_model,
    X,
    y,
    cv=loo,
    n_jobs=-1
)

# Metrics
loo_r2 = r2_score(y, loo_pred)

loo_mae = mean_absolute_error(
    y,
    loo_pred
)

loo_rmse = np.sqrt(
    mean_squared_error(
        y,
        loo_pred
    )
)

print(f"LOOCV R²   : {loo_r2:.4f}")
print(f"LOOCV MAE  : {loo_mae:.4f}")
print(f"LOOCV RMSE : {loo_rmse:.4f}")
# ----------------------------------------------------------
# LOOCV Actual vs Predicted
# ----------------------------------------------------------

plt.figure(figsize=(6,6))

plt.scatter(
    y,
    loo_pred,
    alpha=0.8
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    'r--'
)

plt.xlabel(
    "Actual log10 Conductivity",
    fontsize=16
)

plt.ylabel(
    "LOOCV Predicted log10 Conductivity",
    fontsize=16
)


plt.grid(True)

plt.show()

# ==========================================================
# TEST SET PERFORMANCE
# ==========================================================

best_model.fit(X_train,y_train)
# ==========================================================
# AL DOPING EFFECT ON CONDUCTIVITY
# (AVERAGED OVER ENTIRE DATASET)
# ==========================================================

al_values = [0, 1, 2, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]

mean_cond = []

for al in al_values:

    temp = X.copy()

    temp["Al_at%"] = al

    pred_log = best_model.predict(temp)

    pred_cond = 10**pred_log

    mean_cond.append(pred_cond.mean())

al_prediction = pd.DataFrame({
    "Al_at%": al_values,
    "Predicted_Conductivity_Sm": mean_cond
})

print("\nPredicted Conductivity vs Al Doping")
al_prediction["Predicted_Conductivity_Sm"] = (
    al_prediction["Predicted_Conductivity_Sm"]
    .apply(lambda x: f"{x:.2E}")
)

print(al_prediction)

# Save results
al_prediction.to_excel(
    "Al_Doping_Conductivity_Trend.xlsx",
    index=False
)


# ==========================================================
# CONDUCTIVITY TREND AT FIXED DEPOSITION TEMPERATURES
# ==========================================================

print("\n")
print("="*75)
print("CONDUCTIVITY TREND AT FIXED DEPOSITION TEMPERATURES")
print("="*75)

# Al concentrations to investigate

al_values = [
    0,1,2,3,4,5,
    6,7,8,9,10,
    11,12,13,14,15,
    16,17,18,20
]

# Deposition temperatures chosen from dataset

dep_temps = [27,300,400]

# ----------------------------------------------------------
# Use median values of remaining features
# ----------------------------------------------------------

base = pd.DataFrame({

    "AnnealTemp":[X["AnnealTemp"].median()],

    "Thickness_nm":[X["Thickness_nm"].median()],

    "Bandgap_eV":[X["Bandgap_eV"].median()],

    "Temp_(K)":[300],

    "Crystallite_size":[X["Crystallite_size"].median()]

})

# ----------------------------------------------------------
# Predict conductivity
# ----------------------------------------------------------

prediction_table = []

plt.figure(figsize=(8.5,6))

for dep in dep_temps:

    conductivity = []

    for al in al_values:

        sample = base.copy()

        sample["Al_at%"] = al

        sample["DepTemp"] = dep

        sample = sample[features]

        pred_log = best_model.predict(sample)[0]

        pred_sigma = 10**pred_log

        conductivity.append(pred_sigma)

        prediction_table.append([

            dep,

            al,

            pred_sigma

        ])

    plt.plot(

        al_values,

        conductivity,

        marker="o",

        markersize=7,

        linewidth=2.5,

        label=f"Deposition Temperature = {dep} °C"

    )

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

plt.xlabel("Al Concentration (at.%)")

#plt.yscale("log")
plt.ylabel("Predicted Conductivity (S/m)")

plt.title("Predicted Conductivity at Different Deposition Temperatures")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Conductivity_vs_Al_at_Different_DepTemps.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

# ----------------------------------------------------------
# Save table
# ----------------------------------------------------------

prediction_table = pd.DataFrame(

    prediction_table,

    columns=[

        "DepTemp",

        "Al_at%",

        "Predicted_Conductivity_Sm"

    ]

)

prediction_table.to_excel(

    "Conductivity_vs_Al_at_Fixed_DepTemps.xlsx",

    index=False

)

print(prediction_table)


y_pred = best_model.predict(X_test)

r2 = r2_score(y_test,y_pred)

mae = mean_absolute_error(y_test,y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test,y_pred)
)

print("\n========================")
print("TEST PERFORMANCE")
print("========================")

print("R²   :",round(r2,4))
print("MAE  :",round(mae,4))
print("RMSE :",round(rmse,4))

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

plt.figure(figsize=(6,6))

plt.scatter(y_test,y_pred)

plt.plot(
    [y_test.min(),y_test.max()],
    [y_test.min(),y_test.max()],
    'r--'
)

plt.xlabel(
    "Actual log10 Conductivity",
    fontsize=16
)
plt.ylabel(
    "Predicted log10 Conductivity",
    fontsize=16
)
plt.grid(True)

plt.show()

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    'Feature':features,
    'Importance':best_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(8,5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel(
    "Value",
    fontsize=16
)

plt.ylabel(
    "Feature",
    fontsize=16
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

# ==========================================================
# SHAP ANALYSIS
# ==========================================================

print("\nRunning SHAP Analysis...")

explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X)
interaction_values = explainer.shap_interaction_values(X)
# ==========================================================
# SHAP INTERACTION VALUES
# ==========================================================

print("\nCalculating SHAP Interaction Values...")

interaction_matrix = np.abs(interaction_values).mean(axis=0)

interaction_df = pd.DataFrame(
    interaction_matrix,
    index=X.columns,
    columns=X.columns
)

print(interaction_df.round(4))
plt.figure(figsize=(10,8))

im = plt.imshow(
    interaction_matrix,
    cmap="viridis",
    aspect="auto"
)

plt.xticks(
    range(len(X.columns)),
    X.columns,
    rotation=45,
    ha="right",
    fontsize=15
)

plt.yticks(
    range(len(X.columns)),
    X.columns,
    fontsize=15
)

# -------------------------------------------------
# ADD VALUES INSIDE EACH CELL
# -------------------------------------------------

for i in range(len(X.columns)):
    for j in range(len(X.columns)):

        value = interaction_matrix[i, j]

        plt.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

cbar = plt.colorbar(im)

cbar.set_label(
    "Mean |Interaction SHAP|",
    fontsize=15
)

cbar.ax.tick_params(
    labelsize=13
)

plt.title("SHAP Feature Interaction Heatmap")

plt.tight_layout()

plt.savefig(
    "SHAP_Interaction_Heatmap.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()


# ==========================================================
# SHAP SUMMARY
# ==========================================================

shap.summary_plot(
    shap_values,
    X,
    show=False
)

ax = plt.gca()

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

plt.tight_layout()

plt.savefig(
    "SHAP_Summary.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# SHAP BAR
# ==========================================================

shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)

ax = plt.gca()

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

plt.tight_layout()

plt.savefig(
    "SHAP_Bar.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
# ==========================================================
# UNIQUE SHAP INTERACTION SUMMARY (NO DUPLICATES)
# ==========================================================

interaction_features = []
interaction_shap = []

n_features = len(X.columns)

for i in range(n_features):

    for j in range(i + 1, n_features):

        interaction_features.append(
            f"{X.columns[i]} * {X.columns[j]}"
        )

        interaction_shap.append(
            interaction_values[:, i, j]
        )

interaction_shap = np.array(interaction_shap).T

interaction_df = pd.DataFrame(
    interaction_shap,
    columns=interaction_features
)

# ----------------------------------------------------------
# TOP 5 SHAP INTERACTIONS
# ----------------------------------------------------------

interaction_strength = np.mean(
    np.abs(interaction_shap),
    axis=0
)

top5_idx = np.argsort(interaction_strength)[-5:]

top5_idx = top5_idx[::-1]

top5_shap = interaction_shap[:, top5_idx]

top5_df = interaction_df.iloc[:, top5_idx]


shap.summary_plot(
    top5_shap,
    top5_df,
    plot_type="dot",
    max_display=5,
    show=False
)

fig = plt.gcf()
ax = plt.gca()

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

ax.set_frame_on(True)

ax.tick_params(
    direction="out",
    length=5,
    width=1.2
)

plt.title("Top Five SHAP Feature Interactions")

plt.tight_layout()

plt.savefig(
    "Top5_SHAP_Interactions.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
# ==========================================================
# STRONGEST SHAP INTERACTIONS
# ==========================================================

interaction_strength = np.abs(interaction_values).mean(axis=0)

interaction_df = pd.DataFrame(
    interaction_strength,
    index=X.columns,
    columns=X.columns
)

print("\nAverage Interaction Matrix")
print(interaction_df.round(4))

interaction_df.to_excel(
    "SHAP_Interaction_Matrix.xlsx"
)
plt.figure(figsize=(10,8))

im = plt.imshow(
    interaction_df.values,
    cmap="viridis",
    aspect="auto"
)

plt.xticks(
    range(len(X.columns)),
    X.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(X.columns)),
    X.columns
)

for i in range(len(X.columns)):
    for j in range(len(X.columns)):

        value = interaction_df.values[i, j]

        plt.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

plt.colorbar(im, label="Mean |Interaction SHAP|")

plt.title("Feature Interaction Strength")

plt.tight_layout()

plt.savefig(
    "SHAP_Interaction_Heatmap.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

# ==========================================================
# SHAP DEPENDENCE
# ==========================================================

if "Al_at%" in X.columns:

    shap.dependence_plot(
        "Al_at%",
        shap_values,
        X
    )

# ==========================================================
# TOP FEATURE INTERACTION PLOTS
# ==========================================================

print("\nTop Feature Interaction Plots")

feature_pairs = [

    ("DepTemp", "Crystallite_size"),
    ("Al_at%", "Bandgap_eV"),
    ("DepTemp", "Bandgap_eV"),
    ("Thickness_nm", "Bandgap_eV"),
    ("Al_at%", "DepTemp")


]

for f1, f2 in feature_pairs:

    i = X.columns.get_loc(f1)
    j = X.columns.get_loc(f2)

    plt.figure(figsize=(8,6))

    plt.scatter(
        X[f1],
        interaction_values[:, i, j],
        c=X[f2],
        cmap="viridis",
        s=60
    )

    if f1 == "DepTemp":
        plt.xlabel("Deposition Temperature (°C)")
    elif f1 == "AnnealTemp":
        plt.xlabel("Annealing Temperature (°C)")
    else:
        plt.xlabel(f1)
    plt.ylabel(f"Interaction SHAP ({f1} × {f2})")

    plt.title(f"{f1} × {f2} Interaction")

    cbar = plt.colorbar()
    cbar.set_label(f2)

    plt.grid(True)

    plt.show()
# ==========================================================
# RANK FEATURE PAIRS
# ==========================================================

pairs = []

for i in range(len(X.columns)):
    for j in range(i+1, len(X.columns)):

        score = np.mean(
            np.abs(interaction_values[:, i, j])
        )

        pairs.append([
            f"{X.columns[i]} * {X.columns[j]}",
            score
        ])

pair_df = pd.DataFrame(
    pairs,
    columns=[
        "Interaction",
        "Interaction Strength"
    ]
)

pair_df = pair_df.sort_values(
    "Interaction Strength",
    ascending=False
)

print(pair_df)

pair_df.to_excel(
    "Top_SHAP_Interactions.xlsx",
    index=False
)

# ==========================================================
# DOPING REGION DEFINITIONS
# ==========================================================

region = []

for val in X["Al_at%"]:

    if val <= 5:
        region.append("Low (0-5%)")

    elif val <= 15:
        region.append("Moderate (5-15%)")

    else:
        region.append("High (15-20%)")

region = np.array(region)

# ==========================================================
# TOP 3 FEATURES
# ==========================================================

top_features = [
    "DepTemp",
    "Al_at%",
    "AnnealTemp"
]

# ==========================================================
# FEATURE-WISE SHAP ANALYSIS
# ==========================================================

for feature in top_features:

    idx = X.columns.get_loc(feature)

    feature_values = X[feature]

    shap_feature = shap_values[:, idx]

    conductivity = 10**y.values

    # ------------------------------------------------------
    # NUMERICAL SUMMARY
    # ------------------------------------------------------

    print("\n")
    print("="*70)
    print(feature)
    print("="*70)

    summary = pd.DataFrame({
        "Feature_Value": np.array(feature_values),
        "Conductivity_Sm": np.array(conductivity),
        "SHAP_Value": np.array(shap_feature),
        "Region": np.array(region)
    })

    numerical = summary.groupby("Region").agg({
        "SHAP_Value":["mean","std","min","max"],
        "Conductivity_Sm":["mean","std"]
    })

    print(numerical)

    numerical.to_excel(
        f"{feature}_Region_SHAP_Statistics.xlsx"
    )

    # ------------------------------------------------------
    # GRAPH 1
    # FEATURE VS CONDUCTIVITY
    # ------------------------------------------------------

    plot_df = pd.DataFrame({
        feature: feature_values,
        "Conductivity": conductivity
    })

    plot_df = (
        plot_df
        .groupby(feature)
        .mean()
        .reset_index()
        .sort_values(feature)
    )

    plt.figure(figsize=(10,5))

    # Al doping -> BAR GRAPH
    if feature == "Al_at%":

        plt.bar(
            plot_df[feature],
            plot_df["Conductivity"]
        )

    # Thickness and DepTemp -> LINE GRAPH
    else:

        plt.plot(
            plot_df[feature],
            plot_df["Conductivity"],
            marker="o"
        )

    plt.xlabel(feature)

    plt.ylabel("Mean Conductivity (S/m)")

    plt.title(
        f"{feature} vs Conductivity"
    )

    plt.grid(True)

    plt.show()

    # ------------------------------------------------------
    # GRAPH 2
    # FEATURE VS SHAP IMPACT
    # ------------------------------------------------------

    shap_df = pd.DataFrame({
        feature: feature_values,
        "SHAP": shap_feature
    })

    shap_df = (
        shap_df
        .groupby(feature)
        .mean()
        .reset_index()
        .sort_values(feature)
    )

    plt.figure(figsize=(10,5))

    # Al doping -> BAR GRAPH
    if feature == "Al_at%":

        plt.bar(
            shap_df[feature],
            shap_df["SHAP"]
        )

    # Thickness and DepTemp -> LINE GRAPH
    else:

        plt.plot(
            shap_df[feature],
            shap_df["SHAP"],
            marker="o"
        )

    plt.xlabel(feature)

    plt.ylabel("Mean SHAP Value")

    plt.title(
        f"{feature} vs SHAP Impact"
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.grid(True)

    plt.show()

    # ------------------------------------------------------
    # GRAPH 3
    # SHAP DEPENDENCE
    # ------------------------------------------------------

    shap.dependence_plot(
        feature,
        shap_values,
        X,
        show=True
    )
# ==========================================================
# WATERFALL
# ==========================================================

sample_id = 0

exp = shap.Explanation(
    values=shap_values[sample_id],
    base_values=explainer.expected_value,
    data=X.iloc[sample_id],
    feature_names=X.columns
)

shap.plots.waterfall(exp)

# ==========================================================
# SAVE RESULTS
# ==========================================================

results = X_test.copy()

results["Actual_log10_Conductivity"] = y_test.values
results["Predicted_log10_Conductivity"] = y_pred

results.to_excel(
    "GBR_Conductivity_Predictions.xlsx",
    index=False
)

importance.to_excel(
    "GBR_Feature_Importance.xlsx",
    index=False
)
# ==========================================================
# SAVE LOOCV RESULTS
# ==========================================================

loo_results = pd.DataFrame({

    "Actual_log10_Conductivity": y,

    "LOOCV_Predicted_log10_Conductivity": loo_pred

})

loo_results.to_excel(
    "LOOCV_Results.xlsx",
    index=False
)

print("\nFiles Saved Successfully")

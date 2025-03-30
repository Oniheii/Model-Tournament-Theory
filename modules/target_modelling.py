# target_modelling.py
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from linearmodels.panel import compare
from linearmodels import PooledOLS, PanelOLS, RandomEffects
from sklearn.neighbors import NearestNeighbors


def survival_analysis(df):
    """
    Performs survival analysis on tennis players' tournament progress,
    comparing the maximum round reached by players in tournaments with and without stars.
    The analysis is performed for all players and for each quartile group based on cumulative points.
    Survival curves are plotted, and log-rank tests are performed to assess statistical significance.
    """
    # Map 'max_round' to numerical values
    round_mapping = {
        'R128': 1,
        'R64': 2,
        'R32': 3,
        'R16': 4,
        'QF': 5,
        'SF': 6,
        'F': 7,
        'W': 8
    }
    df['round_num'] = df['max_round'].map(round_mapping)

    # Create 'event' column: 1 if player was eliminated (did not win the tournament), 0 if they won
    df['event'] = np.where(df['round_num'] < 8, 1, 0)

    # Create a flag for each tournament indicating whether there is a star present
    tournament_star = df.groupby(['tourney_date', 'tourney_name'])['is_star'].any().reset_index()
    tournament_star = tournament_star.rename(columns={'is_star': 'star_in_tournament'})
    df = df.merge(tournament_star, on=['tourney_date', 'tourney_name'], how='left')

    # Prepare the reverse mapping of round numbers to names for plotting
    round_mapping_rev = {v: k for k, v in round_mapping.items()}
    round_numbers_sorted = sorted(round_mapping_rev.keys())
    round_labels = [round_mapping_rev[num] for num in round_numbers_sorted]

    # Define quartile groups
    quartile_groups = ['All Players', '0-25%', '25-50%', '50-75%', '75-100%']

    # Convert 'quartile_points' to string and handle missing values
    df['quartile'] = df['quartile_points'].astype(str)
    df['quartile'] = df['quartile_points'].dropna()

    # Add 'All Players' group
    df_all = df.copy()
    df_all['quartile'] = 'All Players'
    df = pd.concat([df, df_all], ignore_index=True)

    # Set up subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # For each quartile group
    for idx, quartile in enumerate(quartile_groups):
        ax = axes[idx]
        df_quartile = df[df['quartile'] == quartile]
        if df_quartile.empty:
            continue
        # Group by 'star_in_tournament' (True or False)
        groups = df_quartile.groupby('star_in_tournament')

        kmf = KaplanMeierFitter()
        survival_functions = {}
        for name, group in groups:
            if group.empty:
                continue
            label = 'Star Present' if name else 'No Star'
            kmf.fit(durations=group['round_num'], event_observed=group['event'], label=label)
            kmf.plot_survival_function(ax=ax, ci_show=False)
            survival_functions[label] = (group['round_num'], group['event'])

        # Perform log-rank test between groups
        if len(survival_functions) == 2:
            durations_A, events_A = survival_functions['No Star']
            durations_B, events_B = survival_functions['Star Present']
            results = logrank_test(
                durations_A=durations_A,
                durations_B=durations_B,
                event_observed_A=events_A,
                event_observed_B=events_B
            )
            # Display test results on the plot
            ax.text(0.5, 0.1, f"Log-rank p-value: {results.p_value:.4e}",
                    transform=ax.transAxes, horizontalalignment='center')
            if results.p_value < 0.05:
                interpretation = "Significant diff"
            else:
                interpretation = "No significant diff"
            ax.text(0.5, 0.05, interpretation, transform=ax.transAxes, horizontalalignment='center')
            # Add hypotheses
            ax.text(0.5, 0.01, "H0: No difference", transform=ax.transAxes, horizontalalignment='center', fontsize=8)
        else:
            ax.text(0.5, 0.1, "Not enough data", transform=ax.transAxes, horizontalalignment='center')

        ax.set_title(f"Survival Curve - {quartile}")
        ax.set_xlabel("Round Reached")
        ax.set_ylabel("Survival Probability")
        ax.set_xticks(round_numbers_sorted)
        ax.set_xticklabels(round_labels)
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for idx in range(len(quartile_groups), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

    # Overall interpretations can be printed here
    print("Hypotheses tested:")
    print(
        "- Null hypothesis (H0): There is no difference in survival between players in tournaments with and without "
        "stars.")
    print(
        "- Alternative hypothesis (H1): There is a difference in survival between players in tournaments with and "
        "without stars.")
    print("Interpretation:")
    print("- If p-value < 0.05, we reject H0 and accept H1, indicating a significant difference.")
    print("- If p-value >= 0.05, we fail to reject H0, indicating no significant difference.")


def hausman_test(fe_res, re_res):
    """
    Effectue le test de Hausman pour comparer le modèle à effets fixes et le modèle à effets aléatoires.

    Parameters:
    fe_res: Résultats du modèle à effets fixes (PanelOLS)
    re_res: Résultats du modèle à effets aléatoires (RandomEffects)

    Returns:
    dict: Résultats du test de Hausman (statistique, p-value, conclusion)
    """
    # Coefficients et covariance des modèles FE et RE
    beta_fe = fe_res.params
    beta_re = re_res.params

    cov_fe = fe_res.cov
    cov_re = re_res.cov

    # Vérifier que les coefficients sont comparables (même variables)
    common_vars = beta_fe.index.intersection(beta_re.index)
    beta_diff = beta_fe[common_vars] - beta_re[common_vars]

    # Calcul de la matrice de covariance des différences
    cov_diff = cov_fe.loc[common_vars, common_vars] - cov_re.loc[common_vars, common_vars]

    # Calcul de la statistique de test de Hausman
    try:
        stat = np.dot(beta_diff.T, np.linalg.inv(cov_diff)).dot(beta_diff)
        p_value = 1 - stats.chi2.cdf(stat, len(common_vars))
    except np.linalg.LinAlgError:
        print("Problème de singularité dans la matrice de covariance. Les résultats peuvent ne pas être fiables.")
        stat, p_value = np.nan, np.nan

    conclusion = "Fixed Effects model is preferred" if p_value < 0.05 else "Random Effects model is appropriate"

    return {
        'hausman_statistic': stat,
        'p_value': p_value,
        'conclusion': conclusion
    }


def breusch_pagan_test(re_res, pooled_res):
    """
    Effectue le test de Breusch-Pagan pour comparer le modèle à effets aléatoires et le modèle Pooled OLS.

    Parameters:
    re_res: Résultats du modèle à effets aléatoires (RandomEffects)
    pooled_res: Résultats du modèle Pooled OLS (PooledOLS)

    Returns:
    dict: Résultats du test de Breusch-Pagan (statistique, p-value, conclusion)
    """
    n = re_res.nobs
    k = re_res.df_model

    # Calcul de la statistique de test de Breusch-Pagan
    sigma_re = re_res.s2
    sigma_pooled = pooled_res.s2
    lm_stat = (n / 2) * ((sigma_re - sigma_pooled) ** 2) / sigma_pooled ** 2
    p_value = 1 - stats.chi2.cdf(lm_stat, k)

    conclusion = "Random Effects model is preferred over Pooled OLS" if p_value < 0.05 else "Pooled OLS is sufficient"

    return {
        'breusch_pagan_statistic': lm_stat,
        'p_value': p_value,
        'conclusion': conclusion
    }


# Manual calculation of the AIC for the Pooled OLS model
def calculate_aic(model_results):
    """
    Calculates the AIC manually for a Pooled OLS model.
    Parameters: - model_results : PanelResults (Pooled OLS model results)
    Returns: - aic (float) : AIC value for the model
    """
    log_likelihood = model_results.loglik  # log-likelihood of the model
    num_params = model_results.params.shape[0]  # number of parameters (constant + coefficients)
    aic = 2 * num_params - 2 * log_likelihood
    return aic


def estimate_and_compare_models(df, X_df, target_col, count_model=False, logit_model=False):
    # 1. S'assurer que df a un MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        if 'player_id' in df.columns and 'tourney_date' in df.columns:
            df = df.set_index(['player_id', 'tourney_date'])
        else:
            raise ValueError("Le DataFrame doit contenir les colonnes 'player_id' et 'tourney_date'.")

    # 2. Variable dépendante y (Series) avec bon index
    y = df[target_col].astype(float)
    y.index = df.index

    # 3. Alignement de X_df
    X_df.index = df.index

    # 4. get_dummies sur X_df avec index conservé
    X_dummies = pd.get_dummies(X_df, drop_first=True).astype(float)

    # 5. Ajouter constante en respectant l’index
    X_dummies = sm.add_constant(X_dummies, has_constant='add')
    X_dummies.index = df.index  # Important

    # 6. Vérification stricte des dimensions
    if y.shape[0] != X_dummies.shape[0]:
        print(f"⚠️ Problème de dimensions : y={y.shape[0]} vs X={X_dummies.shape[0]}")
        raise ValueError("Mismatch in observations between y and X_dummies")

    # 7. Estimation des modèles panel
    pooled_model = PooledOLS(y, X_dummies)
    pooled_res = pooled_model.fit()

    fe_model = PanelOLS(y, X_dummies, entity_effects=True, drop_absorbed=True)
    fe_res = fe_model.fit()

    re_model = RandomEffects(y, X_dummies)
    re_res = re_model.fit()

    fe_time_model = PanelOLS(y, X_dummies, entity_effects=True, time_effects=True, drop_absorbed=True)
    fe_time_res = fe_time_model.fit()

    # 8. Compilation des résultats
    results = {
        'Pooled OLS': pooled_res,
        'Fixed Effects': fe_res,
        'Random Effects': re_res,
        'Year Fixed Effects': fe_time_res
    }

    comparison = compare(results)

    # 9. Comparaison AIC
    display_model_comparisons(fe_res, re_res, pooled_res, fe_time_res, y, X_dummies, count_model, logit_model)

    # 10. Tests statistiques
    display_statistical_tests_results(fe_res, re_res, pooled_res)

    return comparison


def display_model_comparisons(fe_res, re_res, pooled_res, fe_time_res, y, X_df, count_model, logit_model):
    # Calcul des AIC pour chaque modèle
    pooled_aic = round(calculate_aic(pooled_res), 2)
    fe_aic = round(calculate_aic(fe_res), 2)
    re_aic = round(calculate_aic(re_res), 2)
    fe_time_aic = round(calculate_aic(fe_time_res), 2)

    print("Comparaison des AIC :")
    print(f"  Pooled OLS AIC: {pooled_aic}")
    print(f"  Fixed Effect AIC: {fe_aic}")
    print(f"  Random Effect AIC: {re_aic}")
    print(f"  Fixed Time Effect AIC: {fe_time_aic}\n")

    # Comparaison avec un modèle de comptage si `count_model=True`
    if count_model:
        X_df_b = pd.get_dummies(X_df, drop_first=True).astype(float)
        poisson_model = sm.GLM(y, X_df_b, family=sm.families.Poisson()).fit()
        poisson_aic = poisson_model.aic

        # Test de surdispersion
        observed_var = y.var()
        predicted_mean = poisson_model.mu
        dispersion_ratio = observed_var / predicted_mean.mean()

        print("Résultats du modèle de Poisson :")
        print(f"Poisson AIC: {round(poisson_aic, 2)}")
        print(f"Dispersion Ratio (Variance observée / Moyenne prédite): {dispersion_ratio:.4f}")

        # Vérification de surdispersion
        if dispersion_ratio > 1:
            print("Surdispersion détectée : Utilisation d'un modèle binomial négatif.\n")

            # Modèle binomial négatif en cas de surdispersion
            negbin_model = sm.GLM(y, X_df_b, family=sm.families.NegativeBinomial()).fit()
            negbin_aic = negbin_model.aic
            # print("Résultats du modèle binomial négatif :")
            print(f"Negative Binomial AIC: {round(negbin_aic, 2)}")

            # Comparaison de l'AIC entre les modèles
            if negbin_aic < poisson_aic:
                print(
                    "Conclusion : Le modèle binomial négatif offre un meilleur "
                    "ajustement que le modèle de Poisson (AIC).\n")
            else:
                print(
                    "Conclusion : Le modèle de Poisson offre un ajustement similaire "
                    "ou meilleur que le binomial négatif (AIC).\n")

        else:
            print("Pas de surdispersion significative détectée : Le modèle de Poisson est approprié.\n")

            # Conclusion sur la comparaison AIC entre OLS et Poisson
            if poisson_aic < pooled_aic:
                print("Conclusion : Le modèle de Poisson offre un meilleur ajustement que l'OLS (AIC).\n")
            else:
                print(
                    "Conclusion : L'OLS peut offrir un ajustement similaire ou meilleur "
                    "que le modèle de Poisson (AIC).\n")

    # Comparaison avec un modèle logistique/probit si `logit_model=True`
    if logit_model:

        X_df_log = pd.get_dummies(X_df, drop_first=True).astype(float)

        if y.nunique() == 2:
            # Estimation du modèle logit ou probit binaire
            logit_probit_model = sm.Logit(y, X_df_log).fit()
            logit_aic = logit_probit_model.aic
            print("Modèle Logit/Probit (binaire) :")
            print(f"Logit AIC: {logit_aic}\n")
        else:
            # Estimation du modèle logit/probit multinomial
            logit_probit_model = sm.MNLogit(y, X_df_log).fit()
            logit_aic = logit_probit_model.aic
            print("Modèle Logit/Probit Multinomial :")
            print(f"Multinomial Logit AIC: {logit_aic}\n")

        # Conclusion sur la comparaison AIC entre OLS et Logit/Probit
        if logit_aic < pooled_aic:
            print("Conclusion : Le modèle Logit/Probit offre un meilleur ajustement que l'OLS (AIC).\n")
        else:
            print(
                "Conclusion : L'OLS peut offrir un ajustement similaire ou meilleur "
                "que le modèle Logit/Probit (AIC).\n")


def display_statistical_tests_results(fe_res, re_res, pooled_res):
    # Test de Hausman
    hausman_results = hausman_test(fe_res, re_res)
    print("Résultats du Test de Hausman :")
    print(f"Chi-square statistic: {hausman_results['hausman_statistic']:.4f}")
    print(f"P-value: {hausman_results['p_value']:.4e}")
    print(f"Conclusion: {hausman_results['conclusion']}")

    # Test de Breusch-Pagan
    bp_results = breusch_pagan_test(re_res, pooled_res)
    print("\nRésultats du Test de Breusch-Pagan :")
    print(f"LM statistic: {bp_results['breusch_pagan_statistic']:.4f}")
    print(f"P-value: {bp_results['p_value']:.4e}")
    print(f"Conclusion: {bp_results['conclusion']}")

    # Vérifications de robustesse
    print("\nVérification de robustesse :")
    print("- Assurez-vous qu'il n'y a pas de multicolinéarité parfaite ou de matrice de covariance singulière.")
    print("- Les tests sont sensibles aux hypothèses d'homoscedasticité et d'absence d'autocorrélation.\n")


def perform_matching_and_estimation(df, treatment, outcome, X_cols, is_player_level=False):
    """
    Performs matching based on the treatment variable and estimates its effect on the specified outcome.

    Parameters:
    - df : DataFrame containing the data.
    - treatment : Treatment variable (e.g., 'num_stars_in_tournament').
    - outcome : Outcome variable (e.g., 'round_num', 'winner_lower_rank').
    - X_cols : List of covariate columns to compute the propensity score.
    - is_player_level : Boolean indicating whether the data is at player level or match level.
    """
    # Prepare data
    df = df.copy()
    df = df.dropna(subset=[treatment, outcome] + X_cols)
    df = df.copy(deep=True)

    # Encode categorical variables
    # Prepare the data for panel regression
    X = df[X_cols]
    X = ensure_type_conversion(X, cat_codes=is_player_level)
    X = X.copy()
    X = pd.get_dummies(X, drop_first=True).astype(float)

    # Treatment variable
    T = df[treatment]

    # Outcome variable
    Y = df[outcome]

    # Check if treatment is binary or multi-valued
    if T.nunique() == 2:
        # Binary treatment
        # Estimate propensity scores using logistic regression
        logit_model = sm.Logit(T, sm.add_constant(X)).fit(maxiter=1000, disp=False)
        df['propensity_score'] = logit_model.predict(sm.add_constant(X))

        # Perform nearest neighbor matching
        treated = df[T == 1]
        control = df[T == 0]

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control[['propensity_score']])
        distances, indices = nn.kneighbors(treated[['propensity_score']])
        matched_control = control.iloc[indices.flatten()].reset_index(drop=True)
        matched_treated = treated.reset_index(drop=True)

        matched_df = pd.concat([matched_treated, matched_control], ignore_index=True)

        # Estimate treatment effect using matched data
        X_matched = matched_df[X_cols]
        X_matched = pd.get_dummies(X_matched, drop_first=True).astype(float)
        X_matched = sm.add_constant(X_matched)
        Y_matched = matched_df[outcome]

        # Choose appropriate model based on outcome
        if outcome == 'round_num' and is_player_level:
            model = sm.OLS(Y_matched, X_matched).fit()
        elif outcome in ['winner_lower_rank', 'victory_dominant']:
            model = sm.Logit(Y_matched, X_matched).fit(maxiter=1000, disp=False)
        else:
            model = sm.OLS(Y_matched, X_matched).fit()

        # Display results
        print(f"\nResults for {outcome} (Treatment: {treatment}):")
        print(model.summary())

        # Calculate and display AIC
        print(f"\nModel AIC: {model.aic if hasattr(model, 'aic') else 'N/A'}\n")

        # Interpret the treatment effect
        if treatment in model.params:
            treatment_coef = model.params[treatment]
            p_value = model.pvalues[treatment]
            print(f"Estimated Treatment Effect (Coefficient of {treatment}): {treatment_coef}")
            print(f"P-value: {p_value}")
            if p_value < 0.05:
                print("The treatment effect is statistically significant.")
            else:
                print("The treatment effect is not statistically significant.")
        else:
            print("Treatment variable not found in the model parameters.")

        # Plotting the effect
        plt.figure(figsize=(8,6))
        plt.bar(['Treatment Effect'], [treatment_coef], yerr=[model.bse[treatment]])
        plt.ylabel('Coefficient')
        plt.title(f'Effect of {treatment} on {outcome}')
        plt.show()

    else:
        # Multi-valued treatment
        # For simplicity, perform OLS regression including the treatment variable
        # Encode treatment if necessary
        if df[treatment].dtype == 'O' or df[treatment].dtype.name == 'category':
            T_encoded = pd.get_dummies(T, drop_first=True)
            X_full = pd.concat([X, T_encoded], axis=1)
        else:
            X_full = X.copy()
            X_full[treatment] = T
        X_full = sm.add_constant(X_full)

        # Choose appropriate model based on outcome
        if outcome == 'round_num' and is_player_level:
            model = sm.OLS(Y, X_full).fit()
        elif outcome in ['winner_lower_rank', 'victory_dominant']:
            model = sm.Logit(Y, X_full).fit(maxiter=1000, disp=False)
        else:
            model = sm.OLS(Y, X_full).fit()

        # Display results
        print(f"\nResults for {outcome} (Treatment: {treatment}):")
        print(model.summary())

        # Calculate and display AIC
        print(f"\nModel AIC: {model.aic if hasattr(model, 'aic') else 'N/A'}\n")

        # Interpret the treatment effect
        # If treatment is multi-valued, it may have multiple coefficients
        treatment_vars = [var for var in model.params.index if treatment in var]
        print("Estimated Treatment Effects:")
        for var in treatment_vars:
            coef = model.params[var]
            pval = model.pvalues[var]
            print(f"  {var}: Coefficient = {coef}, P-value = {pval}")
            if pval < 0.05:
                print("  The treatment effect is statistically significant.")
            else:
                print("  The treatment effect is not statistically significant.")

        # Plotting the treatment effects
        treatment_coefs = model.params.filter(like=treatment)
        treatment_errors = model.bse.filter(like=treatment)
        treatment_coefs.plot(kind='bar', yerr=treatment_errors, figsize=(8,6))
        plt.ylabel('Coefficient')
        plt.title(f'Effect of {treatment} on {outcome}')
        plt.show()


def ensure_type_conversion(df, cat_codes=False):
    # Parcourir chaque colonne du DataFrame
    for col in df.columns:
        # Si la colonne a un type 'object' ou est de type 'category', ou si elle a un petit nombre de
        # valeurs uniques, elle est considérée comme catégorielle
        if df[col].dtype == 'object' or df[col].dtype.name == 'category' or df[col].nunique() <= 10:
            # Convertir en 'category' pour optimiser la mémoire et simplifier les transformations ultérieures
            if cat_codes:
                df = df.copy()
                df[col] = df[col].astype('category')
            else:
                df = df.copy()
                df[col] = df[col].astype('category').cat.codes


        # Si la colonne a exactement 2 valeurs uniques, elle est considérée comme binaire
        elif df[col].nunique() == 2:
                # Utiliser les codes pour encoder en 0/1 et convertir en entier
                df[col] = df[col].astype(int)
    return df
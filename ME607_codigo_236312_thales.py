# -*- coding: utf-8 -*-
"""
pacotes
"""

#instalar as dependências:
#pip install orbit-ml matplotlib pandas numpy scipy

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter

"""pasta das imagens

"""

os.makedirs("./figuras", exist_ok=True)

"""Carregamento dos dados"""

from orbit.utils.dataset import load_iclaims
df = load_iclaims()

print(f"  Shape do dataset: {df.shape}")
print(f"  Colunas: {list(df.columns)}")
print(f"  Período: {df['week'].min()} a {df['week'].max()}\n")

"""
ANÁLISE EXPLORATÓRIA"""

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Análise Exploratória — Dataset iclaims\n"
             "Pedidos Iniciais de Seguro-Desemprego (EUA, 2010–2018)",
             fontsize=13, fontweight='bold', y=1.01)

cols    = ["claims", "trend.unemploy", "trend.filling", "trend.job"]
labels  = ["log(claims)\n(variável resposta)",
           "Google Trends:\ndesemprego",
           "Google Trends:\nsolicitação",
           "Google Trends:\nemprego"]
colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for ax, col, label, color in zip(axes, cols, labels, colors):
    ax.plot(df["week"], df[col], color=color, linewidth=0.9, alpha=0.85)
    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))

axes[-1].set_xlabel("Semana", fontsize=10)
plt.tight_layout()
plt.savefig("./figuras/01_exploratoria.png", dpi=150, bbox_inches='tight')
plt.close()

"""AJUSTE DO MODELO KTR (BTVC)"""

n_test  = 52   # último ano como teste (52 semanas)
df_train = df.iloc[:-n_test].copy()
df_test  = df.iloc[-n_test:].copy()

print(f"Treino: {len(df_train)} obs | Teste: {len(df_test)} obs\n")

from orbit.models import KTR

 # Modelo KTR com regressores de Google Trends
ktr_model = KTR(
        response_col    = "claims",
        date_col        = "week",
        regressor_col   = ["trend.unemploy", "trend.filling", "trend.job"],
        regressor_sign  = ["=", "=", "="],          # sinais livres para dados log
        seasonality     = [52],                      # sazonalidade anual (semanal)
        seed            = 2024,
        estimator       = "pyro-svi",
        n_bootstrap_draws = 500,
        num_steps       = 400,
        message         = 200,
    )
if not hasattr(np, "in1d"):
    np.in1d = np.isin
ktr_model.fit(df=df_train)

# Previsão
pred_df = ktr_model.predict(df=df_test, decompose=True)
pred_train_df = ktr_model.predict(df=df_train, decompose=True)
model_fitted = True

"""VISUALIZAÇÃO DOS RESULTADOS"""

if model_fitted:
    # ── 4a. Ajuste e previsão ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_train["week"], df_train["claims"],
            color="#333333", lw=1.0, label="Observado (treino)")
    ax.plot(df_test["week"], df_test["claims"],
            color="#555555", lw=1.0, ls="--", label="Observado (teste)")
    ax.plot(pred_df["week"], pred_df["prediction"],
            color="#e74c3c", lw=1.5, label="Predição KTR")
    ax.fill_between(pred_df["week"],
                    pred_df["prediction_5"],
                    pred_df["prediction_95"],
                    color="#e74c3c", alpha=0.2, label="IC 90%")
    ax.axvline(df_test["week"].iloc[0], color="gray", ls=":", lw=1.2)
    ax.set_title("Ajuste e Previsão — Modelo KTR (BTVC)\n"
                 "Pedidos de Seguro-Desemprego (EUA)", fontsize=12)
    ax.set_xlabel("Semana"); ax.set_ylabel("log(claims)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./figuras/02_ajuste_previsao.png", dpi=150, bbox_inches='tight')
    plt.close()


    # ── 4b. Decomposição ────────────────────────────────────────────────────
    components = [c for c in ["trend", "seasonality_52", "regression"]
                  if c in pred_train_df.columns]
    if components:
        fig, axes = plt.subplots(len(components) + 1, 1, figsize=(13, 3 * (len(components) + 1)),
                                 sharex=True)
        axes[0].plot(df_train["week"], df_train["claims"], color="#333", lw=0.9)
        axes[0].set_ylabel("Observado", fontsize=9)
        comp_labels = {"trend": "Tendência",
                       "seasonality_52": "Sazonalidade\n(52 semanas)",
                       "regression": "Componente\nde Regressão"}
        comp_colors = ["#2980b9", "#27ae60", "#e67e22"]
        for ax, comp, color in zip(axes[1:], components, comp_colors):
            ax.plot(df_train["week"], pred_train_df[comp], color=color, lw=1.0)
            ax.set_ylabel(comp_labels.get(comp, comp), fontsize=9)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Semana")
        fig.suptitle("Decomposição do Sinal — Modelo KTR\n"
                     "Tendência + Sazonalidade + Regressão",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("./figuras/03_decomposicao.png", dpi=150, bbox_inches='tight')
        plt.close()


    # ── 4c. Coeficientes variáveis no tempo ─────────────────────────────────
    try:
        coef_df = ktr_model.get_regression_coefs(df=df_train)
        regressores = [c for c in coef_df.columns
                       if c not in ["week", "date"]]
        if regressores:
            fig, axes = plt.subplots(len(regressores), 1,
                                     figsize=(13, 3.5 * len(regressores)), sharex=True)
            if len(regressores) == 1:
                axes = [axes]
            nomes_br = {"trend.unemploy": "Google Trends: desemprego",
                        "trend.filling":  "Google Trends: solicitação",
                        "trend.job":      "Google Trends: emprego"}
            for ax, reg in zip(axes, regressores):
                date_col_c = "week" if "week" in coef_df.columns else coef_df.columns[0]
                ax.plot(coef_df[date_col_c], coef_df[reg],
                        color="#8e44ad", lw=1.2)
                ax.axhline(0, color="gray", ls="--", lw=0.8)
                ax.set_ylabel(f"β(t)\n{nomes_br.get(reg, reg)}", fontsize=9)
                ax.grid(alpha=0.3)
            axes[-1].set_xlabel("Semana")
            fig.suptitle("Coeficientes de Regressão Variáveis no Tempo — KTR\n"
                         "(Estimativas Posteriores via SVI)",
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig("./figuras/04_coeficientes.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Figura 4 salva: 04_coeficientes.png")
    except Exception as e:
        print(f"  (coeficientes não disponíveis nesta versão: {e})")

"""MÉTRICAS DE DESEMPENHO"""

def smape(actual, predicted):
    """Symmetric Mean Absolute Percentage Error."""
    return np.mean(2 * np.abs(actual - predicted) /
                   (np.abs(actual) + np.abs(predicted))) * 100

def mae(actual, predicted):
    """Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))

def rmse(actual, predicted):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))

if model_fitted:
    y_true = df_test["claims"].values
    y_pred = pred_df["prediction"].values

    print("=" * 45)
    print("   MÉTRICAS DE DESEMPENHO — CONJUNTO TESTE")
    print("=" * 45)
    print(f"  SMAPE : {smape(y_true, y_pred):.3f}%")
    print(f"  MAE   : {mae(y_true, y_pred):.4f}")
    print(f"  RMSE  : {rmse(y_true, y_pred):.4f}")
    print("=" * 45)

"""diagnostico do modelo"""

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Fazer a predição usando o modelo KTR já ajustado
pred_df = ktr_model.predict(df)

# 2. Calcular os resíduos (Real - Previsto)
# Assumindo que a coluna dos dados originais se chama "claims"
residuos = df['claims'].values - pred_df['prediction'].values

# 3. Criar a figura para os gráficos de diagnóstico
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plotar o ACF dos resíduos
plot_acf(residuos, ax=axes[0], lags=55, alpha=0.05, color='#1f77b4')
axes[0].set_title("ACF dos Resíduos (Modelo BTVC)", fontsize=14)
axes[0].set_xlabel("Lags (Semanas)")
axes[0].set_ylabel("Autocorrelação")
axes[0].grid(True, alpha=0.3)

# Plotar o PACF dos resíduos
plot_pacf(residuos, ax=axes[1], lags=55, alpha=0.05, color='#ff7f0e')
axes[1].set_title("PACF dos Resíduos (Modelo BTVC)", fontsize=14)
axes[1].set_xlabel("Lags (Semanas)")
axes[1].set_ylabel("Autocorrelação Parcial")
axes[1].grid(True, alpha=0.3)

# Mostrar e salvar para o relatório
plt.tight_layout()
plt.savefig("acf_pacf_residuos.png", dpi=300, bbox_inches='tight')
plt.show()

"""VAlidacao"""

# Série principal
yt   = df["claims"].values
datas = df["week"].values

#2. PRIMEIRAS DIFERENÇAS
dy1 = np.diff(yt, n=1)
dy2 = np.diff(yt, n=2)

datas_dy1 = datas[1:]
datas_dy2 = datas[2:]

#  3. ESTATÍSTICAS DESCRITIVAS
print("── Estatísticas Descritivas (dados REAIS) ──────────────────")
for nome, ser in [("Yt", yt), ("ΔYt", dy1), ("Δ²Yt", dy2)]:
    print(f"  {nome:8s}: n={len(ser):3d}  média={ser.mean():.4f}"
          f"  var={ser.var():.5f}  min={ser.min():.4f}  max={ser.max():.4f}")
print()

# 4. ACF E PACF IMPLEMENTADOS MANUALMENTE
def acf_manual(x, nlags=48):
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x) / n
    vals = np.array([
        np.dot(x[:n-k], x[k:]) / (n * c0) if k > 0 else 1.0
        for k in range(nlags + 1)
    ])
    ci = 1.96 / np.sqrt(n)
    return vals, ci

def pacf_manual(x, nlags=48):
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x) / n
    r = np.array([np.dot(x[:n-k], x[k:]) / n for k in range(nlags + 1)])
    r_norm = r / c0

    pacf_vals = [1.0]
    phi = np.array([r_norm[1]])
    pacf_vals.append(float(phi[0]))

    for k in range(2, nlags + 1):
        num = r_norm[k] - np.dot(phi, r_norm[1:k][::-1])
        den = 1.0  - np.dot(phi, r_norm[1:k])
        phi_k = num / den if abs(den) > 1e-12 else 0.0
        phi_new = phi - phi_k * phi[::-1]
        phi = np.append(phi_new, phi_k)
        pacf_vals.append(float(phi_k))

    ci = 1.96 / np.sqrt(n)
    return np.array(pacf_vals), ci

NLAGS = 60   # ~1 ano e meio de lags semanais
lags  = np.arange(NLAGS + 1)

acf_y,   ci_y   = acf_manual(yt,  NLAGS)
pacf_y,  _      = pacf_manual(yt,  NLAGS)
acf_d1,  ci_d1  = acf_manual(dy1, NLAGS)
pacf_d1, _      = pacf_manual(dy1, NLAGS)
acf_d2,  ci_d2  = acf_manual(dy2, NLAGS)
pacf_d2, _      = pacf_manual(dy2, NLAGS)

# 5. FIGURA A: SÉRIE + DIFERENÇAS
fig_a, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=False)
fig_a.suptitle(
    r"Série $Y_t$ e suas Diferenças — Dataset iclaims (dados reais, 2010–2018)",
    fontsize=12, fontweight='bold', y=1.01
)

axes[0].plot(datas, yt, color="#1a3a5c", lw=0.9)
axes[0].set_ylabel(r"$Y_t$ — log(claims)", fontsize=10)
axes[0].set_title(
    r"$Y_t$: Série original (tendência decrescente + sazonalidade anual)",
    fontsize=9, loc='left'
)
axes[0].grid(alpha=0.3)

axes[1].plot(datas_dy1, dy1, color="#2980b9", lw=0.85)
axes[1].axhline(0, color='gray', lw=0.7, ls='--')
axes[1].set_ylabel(r"$\Delta Y_t$", fontsize=10)
axes[1].set_title(
    r"$\Delta Y_t = Y_t - Y_{t-1}$: 1ª diferença (tendência removida)",
    fontsize=9, loc='left'
)
axes[1].grid(alpha=0.3)

axes[2].plot(datas_dy2, dy2, color="#27ae60", lw=0.85)
axes[2].axhline(0, color='gray', lw=0.7, ls='--')
axes[2].set_ylabel(r"$\Delta^2 Y_t$", fontsize=10)
axes[2].set_title(
    r"$\Delta^2 Y_t$: 2ª diferença (sobre-diferenciada — somente para fins comparativos)",
    fontsize=9, loc='left'
)
axes[2].grid(alpha=0.3)
axes[2].set_xlabel("Semana", fontsize=10)

for ax in axes:
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))

plt.tight_layout()
fig_a.savefig("./figuras/06_serie_diferencas.png", dpi=160, bbox_inches='tight')
plt.close(fig_a)


#  6. FIGURA B: PAINEL 3×3 (série / FAC / FACP)
fig_b, axes_b = plt.subplots(3, 3, figsize=(15, 9))
fig_b.suptitle(
    r"Painel Resumo: $Y_t$, $\Delta Y_t$, $\Delta^2 Y_t$ — Série, FAC e FACP"
    "\n(Dataset iclaims — dados reais)",
    fontsize=12, fontweight='bold', y=1.02
)

series_list = [(datas,     yt,  r"$Y_t$"),
               (datas_dy1, dy1, r"$\Delta Y_t$"),
               (datas_dy2, dy2, r"$\Delta^2 Y_t$")]

acfs  = [(acf_y,  ci_y),  (acf_d1, ci_d1), (acf_d2, ci_d2)]
pacfs = [(pacf_y, ci_y),  (pacf_d1,ci_d1), (pacf_d2,ci_d2)]

c_ser  = ["#1a3a5c", "#2980b9", "#27ae60"]
c_acf  = ["#1a3a5c", "#2980b9", "#27ae60"]
c_pacf = ["#8e44ad", "#e67e22", "#c0392b"]

for i, ((dt, ser, lbl), ca, cp, cs) in enumerate(
        zip(series_list, c_acf, c_pacf, c_ser)):

    #  Coluna 0: série
    axes_b[i, 0].plot(dt, ser, color=cs, lw=0.85)
    axes_b[i, 0].axhline(np.mean(ser), color='gray', ls='--', lw=0.7, alpha=0.6)
    axes_b[i, 0].set_ylabel(lbl, fontsize=11)
    axes_b[i, 0].grid(alpha=0.3)
    axes_b[i, 0].xaxis.set_major_formatter(DateFormatter('%Y'))
    if i == 0:
        axes_b[i, 0].set_title("Série temporal", fontsize=10)

    #  Coluna 1: FAC
    a_v, a_ci = acfs[i]
    axes_b[i, 1].bar(lags, a_v, color=ca, alpha=0.65, width=0.7)
    axes_b[i, 1].axhline(0, color='black', lw=0.5)
    axes_b[i, 1].axhline( a_ci, color='red', lw=1.0, ls='--', label='IC 95%')
    axes_b[i, 1].axhline(-a_ci, color='red', lw=1.0, ls='--')
    axes_b[i, 1].fill_between(lags, -a_ci, a_ci, alpha=0.07, color='red')
    axes_b[i, 1].axvline(52, color='orange', lw=1.0, ls=':', alpha=0.9,
                         label='Lag 52 (1 ano)')
    axes_b[i, 1].set_xlim(-1, NLAGS + 1)
    axes_b[i, 1].set_ylim(-1.05, 1.05)
    axes_b[i, 1].grid(alpha=0.25)
    axes_b[i, 1].legend(fontsize=7, loc='upper right')
    if i == 0:
        axes_b[i, 1].set_title("FAC", fontsize=10)

    #  Coluna 2: FACP
    p_v, p_ci = pacfs[i]
    axes_b[i, 2].bar(lags, p_v, color=cp, alpha=0.65, width=0.7)
    axes_b[i, 2].axhline(0, color='black', lw=0.5)
    axes_b[i, 2].axhline( p_ci, color='red', lw=1.0, ls='--', label='IC 95%')
    axes_b[i, 2].axhline(-p_ci, color='red', lw=1.0, ls='--')
    axes_b[i, 2].fill_between(lags, -p_ci, p_ci, alpha=0.07, color='red')
    axes_b[i, 2].axvline(52, color='orange', lw=1.0, ls=':', alpha=0.9,
                         label='Lag 52 (1 ano)')
    axes_b[i, 2].set_xlim(-1, NLAGS + 1)
    axes_b[i, 2].set_ylim(-1.05, 1.05)
    axes_b[i, 2].grid(alpha=0.25)
    axes_b[i, 2].legend(fontsize=7, loc='upper right')
    if i == 0:
        axes_b[i, 2].set_title("FACP", fontsize=10)

    if i == 2:
        axes_b[i, 0].set_xlabel("Semana", fontsize=9)
        axes_b[i, 1].set_xlabel("Lag (semanas)", fontsize=9)
        axes_b[i, 2].set_xlabel("Lag (semanas)", fontsize=9)

plt.tight_layout()
fig_b.savefig("./figuras/08_painel_resumo.png", dpi=160, bbox_inches='tight')
plt.close(fig_b)


# 7. LAGS SIGNIFICATIVOS
print("\n── Lags significativos na FACP de Yt (dados reais) ─────────")
sig = [k for k in range(1, NLAGS+1) if abs(pacf_y[k]) > ci_y]
print(f"  {sig[:15]}")
print("\n── Lags significativos na FAC de Yt ────────────────────────")
sig2 = [k for k in range(1, NLAGS+1) if abs(acf_y[k]) > ci_y]
print(f"  {sig2[:20]}")
import sys
import os
import numpy as np
import pandas as pd
import pickle
import pyEDM
from scipy.signal import argrelmin

BaseFolder = sys.argv[1]
prefix = sys.argv[2]
cores = int(sys.argv[3])
max_mi_shift = int(sys.argv[4])
embedding_dim = int(sys.argv[5])
lag = int(sys.argv[6])
eccm_window_size = int(sys.argv[7])

if not BaseFolder.endswith("/"):
    BaseFolder += "/"

with open(BaseFolder + prefix + 'amplified_dfs.pickle', 'rb') as handle:
    amplified_dfs = pickle.load(handle)
with open(BaseFolder + prefix + 'DictCols.pickle', 'rb') as handle:
    DictCols = pickle.load(handle)
with open(BaseFolder + prefix + 'x1_x2_columns.pickle', 'rb') as handle:
    cols_x1, cols_x2 = pickle.load(handle)

sf = not prefix.startswith("surr_")

def calcCCM(df, x1, x2, prefix, d):
    for col in [x1, x2]:
        if col not in df.columns:
            df[col] = np.nan
    try:
        libstart = 1
        libend = int(len(df) * 0.8)
        predstart = libend + 1
        predend = len(df)
        try:
            df_EmbedDim = pyEDM.EmbedDimension(
                dataFrame=df[[x1, x2]].reset_index(),
                columns=str(x1),
                maxE=embedding_dim,
                target=str(x2),
                lib=f"{libstart} {libend}",
                pred=f"{predstart} {predend}",
                showPlot=False,
                numThreads=1
            )
            optimalrho = df_EmbedDim["rho"].max()
            embed = df_EmbedDim[df_EmbedDim["rho"] == optimalrho]["E"].values[0]
            if embed < 3:
                embed = 3
        except Exception as e:
            print("EmbedDimension error:", e)
            embed = 5

        try:
            df_PNL = pyEDM.PredictNonlinear(
                dataFrame=df[[x1, x2]].reset_index(),
                E=int(embed),
                columns=str(x1),
                lib=f"{libstart} {libend}",
                pred=f"{predstart} {predend}",
                showPlot=False
            )
            NonLinearity = (df_PNL["rho"].max() != df_PNL["rho"].values.tolist()[0])
            if not NonLinearity:
                return [0, 0, 0, 0, False, 0, 0]
        except Exception as e:
            print("PredictNonlinear error:", e)
            NonLinearity = False

        arr = pd.Series(df[x1]).ewm(span=3).mean().values
        try:
            lagX1 = int(argrelmin(arr)[0][0])
        except Exception:
            lagX1 = 2
        if lagX1 < 2:
            lagX1 = 2
        if lagX1 > lag:
            lagX1 = lag
        lagX2 = lagX1
        lagX1 = int(lagX1)
        lagX2 = int(lagX2)
        embed = int(embed)
        df = df.fillna(0)

        E_used = int(embed)
        nrows = len(df)
        max_lib = nrows - 2
        min_lib = E_used
        
        libSizes = [min_lib, max_lib,1]
        print("xxxx1111")
        print(x1)
        print("xxxxx222")
        print(x2)
        try:
            x1 = str(x1)
            x2 = str(x2)
            ccm_result_x1 = pyEDM.CCM(
                dataFrame=df[[x1, x2]].reset_index().copy(),
                E=E_used,
                Tp=0,
                columns=str(x1),
                target=str(x2),
                libSizes=libSizes,
                sample=1,
                showPlot=False
            )
            ccm_result_x2 = pyEDM.CCM(
                dataFrame=df[[x1, x2]].reset_index().copy(),
                E=E_used,
                Tp=0,
                columns=str(x2),
                target=str(x1),
                libSizes=libSizes,
                sample=1,
                showPlot=False
            )
            df_Scores = pd.DataFrame()
            key_x1x2 = f"{x1}:{x2}"
            key_x2x1 = f"{x2}:{x1}"
            if key_x1x2 in ccm_result_x1 and key_x2x1 in ccm_result_x2:
                df_Scores["Library length"] = ccm_result_x1["LibSize"]
                df_Scores["x1"] = ccm_result_x1[key_x1x2]
                df_Scores["x2"] = ccm_result_x2[key_x2x1]
            elif "rho" in ccm_result_x1 and "rho" in ccm_result_x2:
                df_Scores["Library length"] = ccm_result_x1["LibSize"]
                df_Scores["x1"] = ccm_result_x1["rho"]
                df_Scores["x2"] = ccm_result_x2["rho"]
            else:
                raise ValueError("Missing rho in CCM result")
            df_Scores = df_Scores.set_index("Library length").fillna(0)
            Score_X1 = (
                df_Scores["x1"].values[-5:].mean() if len(df_Scores["x1"].values) >= 5 else df_Scores["x1"].mean()
            )
        except Exception as e:
            print(f"pyEDM CCM calculation error for {x1}->{x2}: {e}")
            print("df columns:", list(df.columns))
            df_Scores, Score_X1 = 0, 0

    except Exception as e:
        print("CCM overall error:", e)
        lagX1 = 2
        embed = 5
        df_Scores, Score_X1, x1, x2, NonLinearity = 0, 0, 0, 0, False

    if (x1 in list(d.keys())) and (x2 in list(d.keys())):
        return [df_Scores, Score_X1, d[x1], d[x2], NonLinearity, lagX1, embed]
    else:
        return [0, 0, 0, 0, False, 0, 0]

def fullCCM(dfsList, col, targetCol, dic, prefix_, showFig=False):
    tmp_results = []
    for j in dfsList:
        j = j.fillna(0)
        tmp_results.append(calcCCM(j,
                                   x1=dic[col],
                                   x2=dic[targetCol],
                                   prefix=prefix_,
                                   d=dic.copy()))
    Final_results = []
    calculated_dfs = []
    for k, valk in enumerate(tmp_results):
        if valk[-3] == True and hasattr(valk[0], 'reset_index'):
            calculated_dfs.append(valk[0].reset_index())
            Final_results.append(valk)
    if len(calculated_dfs) > 1:
        c = pd.concat(calculated_dfs, axis=0, ignore_index=False)
        c_means = pd.DataFrame()
        c_means["x1_mean"] = c.groupby("Library length")["x1"].agg("mean")
        c_means["x2_mean"] = c.groupby("Library length")["x2"].agg("mean")
        c_means = c_means.reset_index()

        if showFig == True:
            import matplotlib.pyplot as plt
            from scipy.stats import gaussian_kde
            import numpy as np
            try:
                xy = np.vstack([c["Library length"].values, c["x1"].values])
                kde = gaussian_kde(xy)
                x_grid = np.linspace(c["Library length"].min(), c["Library length"].max(), 100)
                y_grid = np.linspace(c["x1"].min(), c["x1"].max(), 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kde(positions).T, X.shape)

                fig, ax = plt.subplots()
                cax = ax.imshow(Z, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
                                origin='lower', cmap='coolwarm', aspect='auto')
                ax.contour(X, Y, Z, colors='blue')
                ax.scatter(c_means["Library length"].values, c_means["x1_mean"].values, color="black", s=7)
                cbar = fig.colorbar(cax, ax=ax)
                cbar.set_label('Density')
                ax.set_title('Density Plot of ' + str(col) + ' affects ' + str(targetCol))
                ax.set_xlabel('Library size (l)')
                ax.set_ylabel('rho (p)')
                plt.savefig(BaseFolder + f"ccm_density_{col}_{targetCol}.png")
                plt.close()
            except Exception:
                fig, ax = plt.subplots()
                ax.scatter(c_means["Library length"].values, c_means["x1_mean"].values, color="red", s=3)
                ax.set_xlabel('Library size (l)')
                ax.set_ylabel('rho (p)')
                plt.savefig(BaseFolder + f"ccm_{col}_{targetCol}.png")
                plt.close()
        return c, c_means, Final_results
    else:
        return 0, 0, Final_results


def manipulate(v):
    All_causal_CCM_dfs = []
    for valj in cols_x2:
        All_causal_CCM_dfs.append(fullCCM(
            dfsList=amplified_dfs,
            col=v,
            targetCol=valj,
            dic=DictCols,
            prefix_=prefix,
            showFig=sf
        ))
    return All_causal_CCM_dfs

if __name__ == "__main__":
    results_list_final = [manipulate(x) for x in cols_x1]
    results_list_fixed = []
    for i in results_list_final:
        results_list_fixed.extend(i)
    with open(BaseFolder + 'All_' + prefix + 'results.pickle', 'wb') as handle:
        pickle.dump(results_list_fixed, handle, protocol=pickle.HIGHEST_PROTOCOL)

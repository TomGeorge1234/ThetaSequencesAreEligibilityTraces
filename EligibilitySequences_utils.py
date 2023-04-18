import numpy as np
import scipy
from scipy import stats
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tomplotlib import tomplotlib as tpl
import pickle
import time

import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells
from ratinabox.contribs.ValueNeuron import ValueNeuron
from ratinabox.contribs.ThetaSequenceAgent import ThetaSequenceAgent


def r_squared(prediction, true):
    # coefficient of determination
    residuals = np.sum((prediction - true) ** 2)
    mean = np.mean(true)
    var = np.sum((true - mean) ** 2)
    return 1 - residuals / var


def corr_coef(prediction, true):
    # pearson correlation coefficient
    try:
        return scipy.stats.pearsonr(prediction, true).statistic
    except ValueError:
        return np.nan


def get_agent(agent_type="normal", Env=None, v_sweep=5, **constants):
    if agent_type == "normal":
        Ag = Agent(
            Env,
            params={
                "dt": constants["dt"],
                "speed_std": 0.0,
                "speed_mean": constants["speed"],
            },
        )
        Ag.compression_factor = 1
    elif agent_type == "theta":
        Ag = ThetaSweepAgent(
            Env,
            params={
                "dt": constants["dt"],
                "speed_std": 0.0,
                "speed_mean": constants["speed"],
                "v_sweep": v_sweep,
                "theta_frac": constants["theta_frac"],
                "theta_freq": constants["theta_freq"],
            },
        )
        Ag.compression_factor = (Ag.v_sweep + Ag.speed_mean) / (Ag.speed_mean)
    return Ag


def get_true_value_function(Env, Reward, **constants):
    x_range = Env.discrete_coords.reshape(-1)
    rew = Reward.get_state(evaluate_at="all").reshape(-1)
    tau_x = constants["speed"] * constants["tau"]
    kernel = (
        (1 / constants["speed"])
        * np.exp(-x_range / (constants["tau"] * constants["speed"]))
        * Env.dx
    )
    rew = np.concatenate((rew, rew, rew))
    true_value = np.convolve(rew, kernel[::-1])[len(kernel) :][
        len(x_range) : 2 * len(x_range)
    ]
    return x_range, true_value


def init_simulation(
    agent_type="normal",
    tau_e=0.01,
    v_sweep=5,
    tau_=4,
    **constants,
):
    """Returns an Environment, an Agent as well as Feature, Reward and Value neurons classes
    returns (tuple): (Env, Ag, Features, Reward, Value)
    """
    Env = Environment(
        params={
            "dx": constants["dx"],
            "dimensionality": "1D",
            "scale": constants["size"],
            "boundary_conditions": "periodic",
        }
    )

    # create agent
    Ag = get_agent(agent_type=agent_type, Env=Env, v_sweep=v_sweep, **constants)

    # create basis features and rewards
    Features = PlaceCells(
        Ag,
        params={
            "n": constants["N_cells"],
            "widths": constants["radius_cells"],
            "description": "gaussian",
            "place_cell_centres": "uniform",
            "name": "Features",
            "save_history": False,
        },
    )
    Reward = PlaceCells(
        Ag,
        params={
            "n": 1,
            "place_cell_centres": np.array([Env.scale - 0.05]),
            "widths": 0.02,
            "description": "gaussian",
            "color": "C2",
            "save_history": False,
        },
    )

    # create value neuron
    Value = ValueNeuron(
        Ag,
        params={
            "input_layer": Features,
            "tau": tau_,
            "tau_e": tau_e,
            "eta": constants["eta"],
            "L2": constants["L2"],
            "color": "C4",
            "save_history": False,
        },
    )

    Value.inputs["Features"]["w"] = np.random.normal(
        scale=1e-4, size=Value.inputs["Features"]["w"].shape
    )
    Value.history["TDerror"] = []
    Value.history["pos"] = []
    if agent_type == "theta":
        Value.history["theta_pos"] = []
    Value.history["score"] = {}
    Value.history["score"]["t"] = []
    Value.history["score"]["r2"] = []
    Value.history["score"]["cc"] = []
    Value.history["score"]["ratemap"] = []

    return Env, Ag, Features, Reward, Value


def run_simulation(
    N_repeats=1, agent_type="normal", tau_e=0.01, v_sweep=5, max_laps=50, **constants
):
    if agent_type == "normal":
        constants["dt"] = min(tau_e / 4, 0.05)
        compression_factor = 1
        tau = constants["tau"]
        t_start_learning = 0
        theta_frac = 1
    elif agent_type == "theta":
        compression_factor = (v_sweep + constants["speed"]) / constants["speed"]
        tau = constants["tau"] / compression_factor
        theta_frac = constants["theta_frac"]
        t_start_learning = (
            v_sweep
            * constants["theta_frac"]
            / (constants["theta_freq"] * 2 * constants["speed"])
        )
    results = []
    for N in range(N_repeats):
        Env, Ag, Features, Reward, Value = init_simulation(
            agent_type=agent_type, tau_e=tau_e, v_sweep=v_sweep, tau_=tau, **constants
        )

        x_range, true_value = get_true_value_function(Env, Reward, **constants)
        t_max = max_laps * constants["size"] / constants["speed"]
        pbar2 = tqdm(range(int(t_max / Ag.dt)))
        if agent_type == "theta":
            while Ag.t < t_start_learning:  # spin up (for theta agent)
                Ag.update()
            Ag.t, Ag.TrueAgent.t = 0, 0
        for i in pbar2:
            Ag.update()
            Features.update()
            Reward.update()
            if np.isnan(Ag.pos[0]):
                Value.eta = 0
            else:
                Value.eta = constants["eta"]
            Value.update()
            Value.update_weights(
                reward=compression_factor * Reward.firingrate / theta_frac
            )
            # Value.history['TDerror'].append(Value.td_error)
            # Value.history['pos'].append(Value.Agent.pos)
            # if agent_type == 'theta':
            #     Value.history['theta_pos'].append(Value.Agent.TrueAgent.pos)

            # periodically save some data
            if i % (int(1 / Ag.dt)) == 0:
                predicted_value = Value.get_state(evaluate_at="all").reshape(-1)
                if max(predicted_value > 1e-6):
                    predicted_value *= max(true_value) / max(predicted_value)
                Value.history["score"]["t"].append(Ag.t)
                Value.history["score"]["r2"].append(
                    r_squared(predicted_value, true_value)
                )
                Value.history["score"]["cc"].append(
                    corr_coef(predicted_value, true_value)
                )
                Value.history["score"]["ratemap"].append(
                    Value.get_state(evaluate_at="all")
                )
                lap_count = Ag.t / (constants["size"] / constants["speed"])
                pbar2.set_description(
                    f"Lap = {lap_count:.3f}, R2 = {Value.history['score']['r2'][-1]:.3f}, CC = {Value.history['score']['cc'][-1]:.3f}"
                )

            # if the last 20 (1 whole lap) r2 scores are >= 0.99 terminate
            r2 = np.array(Value.history["score"]["r2"])
            if len(r2) > 20:
                if np.prod((r2[-20:] >= 0.99)) == 1:
                    break

        results.append(Value.history)

    return results, (Env, Ag, Features, Reward, Value)


def plot_r2(
    results, fig=None, ax=None, color="C1", label="", func_to_plot="r2", **constants
):
    # results is a history dataframe

    if fig is None and ax is None:
        fig, ax = plt.subplots()
        ax.set_ylim(bottom=-1)

    t = np.arange(641)
    loops = t / (constants["size"] / constants["speed"])
    r2 = np.ones((len(results), 641))
    for i, result in enumerate(results):
        if func_to_plot == "r2":
            r2_ = np.array(result["score"]["r2"])
        elif func_to_plot == "cc":
            r2_ = np.array(result["score"]["cc"])
        r2[i, : len(r2_)] = r2_
    one = np.argmin(np.abs(loops[loops < 1] - 1))
    # one=0
    mean = r2.mean(axis=0)[one:]
    std = r2.std(axis=0)[one:]
    loops = loops[one:]
    ax.plot(loops, mean, color=color, label=label)
    ax.fill_between(loops, mean + std, mean - std, alpha=0.2, facecolor=color)
    ax.set_xscale("log")
    ax.set_xlim(left=1, right=32)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_ylim(bottom=-0.5, top=1)

    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    # ax.spines['left'].set_position(1/2)
    # ax.spines['bottom'].set_position('zero')
    ax.minorticks_off()
    return fig, ax


def plot_ratemap_evolution(results, fig=None, ax=None, **constants):
    cmap = matplotlib.cm.get_cmap("plasma_r")
    loops = np.arange(33)
    times = loops * constants["size"] / constants["speed"]
    full_times = np.arange(641)
    ratemaps = np.zeros(
        (len(results), len(times), results[0]["score"]["ratemap"][0][0].shape[0])
    )
    x_range = np.linspace(0, constants["size"], ratemaps.shape[-1])
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    for i, time in enumerate(times):
        color = cmap(i / (len(times) - 1))
        id = np.argmin(np.abs(full_times - time))
        for j, result in enumerate(results):
            recorded_times = np.array(results[j]["score"]["t"])
            if id >= len(recorded_times):
                id_ = len(recorded_times) - 1
            else:
                id_ = id
            ratemaps[j, i, :] = np.array(result["score"]["ratemap"][id_][0])
        mean = ratemaps.mean(axis=0)[i]
        ax.plot(
            x_range - 0 * i / len(times),
            mean + 0 * i / len(times),
            color=color,
            alpha=0.8,
        )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=cax)

    return fig, ax


def pickle_and_save(object, name="", directory="./results/"):
    with open(directory + name + ".pickle", "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_and_depickle(name, directory="./results/"):
    with open(directory + name + ".pickle", "rb") as handle:
        object = pickle.load(handle)
    return object

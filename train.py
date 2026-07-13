import time

import torch

from loss import MultiScaleLoss
from models.cdpss import MultiScaleCDPSS
from models.model import SGDDyG
from utils import DataLoader, util
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args


def build_edge_index_from_edges(edges):
    return edges[1:3]


def build_input_multiscale_views(A_seq, r_mid, r_long):
    views = {"short": A_seq}
    for name, window in (("mid", r_mid), ("long", r_long)):
        aggregated = []
        window_sum = None
        for t in range(len(A_seq)):
            window_sum = A_seq[t] if window_sum is None else window_sum + A_seq[t]
            remove_idx = t - window
            if remove_idx >= 0:
                window_sum = window_sum - A_seq[remove_idx]
            aggregated.append(window_sum.coalesce())
        views[name] = aggregated
    return views


def make_encoder(T, N, args):
    return SGDDyG(
        T,
        N,
        hidden_features=args.hidden_features,
        num_feature=args.num_feature,
        bandwidth=args.bandwidth,
        tgc_dropout=args.tgc_dropout,
        fft_dropout=args.fft_dropout,
    )


def encode_multiscale(model, A_views, edge_nodes, M, cl):
    node_emb_by_scale = {}
    feature_reg_loss = None
    for scale_id, scale in enumerate(("short", "mid", "long")):
        out = model(A_views[scale][:-1], edge_nodes, M, cl, return_embeddings=True, scale_id=scale_id)
        node_emb_by_scale[scale] = out["node_emb_seq"]
        if feature_reg_loss is None:
            feature_reg_loss = out["feature_reg_loss"]
    return {
        "node_emb_seq": node_emb_by_scale,
        "feature_reg_loss": feature_reg_loss,
    }


def encode_short_for_cl(model, A_views, edge_nodes, M):
    out = model(A_views["short"][:-1], edge_nodes, M, True, return_embeddings=True, scale_id=0)
    return out["node_emb_seq"]


def run_cdpss(cdpss, H, edge_index, edge_nodes, graph_stats):
    return cdpss(H, edge_index, edge_nodes[0], edge_nodes[1], graph_stats)


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    (
        TS,
        labels,
        A_train,
        A_val,
        A_test,
        N,
        graph_stats_train,
        graph_stats_val,
        graph_stats_test,
    ) = DataLoader.load_data(args, device)
    T = len(A_train) - 1
    M = util.func_createM(T, args.bandwidth, args.m_choice, device)

    edges_train, target_train, edges_val, target_val, K_val, edges_test, target_test, K_test = util.split_data(labels, TS)
    train_edge_nodes, val_edge_nodes, test_edge_nodes = util.get_all_edges_nodes(edges_train, edges_val, edges_test, N)
    edge_index_train = build_edge_index_from_edges(edges_train)
    edge_index_val = build_edge_index_from_edges(edges_val)
    edge_index_test = build_edge_index_from_edges(edges_test)

    A_train_views = build_input_multiscale_views(A_train, args.r_mid, args.r_long)
    A_val_views = build_input_multiscale_views(A_val, args.r_mid, args.r_long)
    A_test_views = build_input_multiscale_views(A_test, args.r_mid, args.r_long)

    run_results = []
    for run in range(args.num_runs):
        util.set_seed(args.seed)
        model = make_encoder(T, N, args).to(device=device)
        cdpss = MultiScaleCDPSS(
            emb_dim=args.hidden_features[-1],
            num_nodes=N,
            attention_temperature=args.attention_temperature,
            lora_rank=args.lora_rank,
            hidden_dim=args.cdpss_hidden_dim,
            dropout=args.cdpss_dropout,
        ).to(device)

        _, save_model_folder = util.get_save_parameter(args.lr, args.lam, args.num_feature, run, args.tau, args)
        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=save_model_folder,
            model_name=args.save_model_name,
            extra_model=cdpss,
        )

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(cdpss.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = MultiScaleLoss(
            lam=args.lam,
            tau=args.tau,
            cl_powerlaw_eta=args.cl_powerlaw_eta,
            cl_weight_min=args.cl_weight_min,
            cl_weight_max=args.cl_weight_max,
            cl_degree_bins=args.cl_degree_bins,
            cl_disagreement_alpha=args.cl_disagreement_alpha,
            cl_disagreement_min=args.cl_disagreement_min,
            cl_disagreement_max=args.cl_disagreement_max,
            cl_disagreement_normalize=args.cl_disagreement_normalize,
        )

        epoch_logs = []
        epochs_trained = 0
        synchronize_if_needed(device)
        train_start = time.perf_counter()
        for ep in range(1, args.epochs + 1):
            epochs_trained = ep
            model.train()
            cdpss.train()
            optimizer.zero_grad()

            out_train = encode_multiscale(model, A_train_views, train_edge_nodes, M, False)
            h2_train = encode_short_for_cl(model, A_train_views, train_edge_nodes, M)
            H_train = out_train["node_emb_seq"]
            cdpss_train = run_cdpss(cdpss, H_train, edge_index_train, train_edge_nodes, graph_stats_train)
            output_train = cdpss_train["final_logits"]
            h1_train = H_train["short"]

            loss_dict = criterion(
                final_logits=output_train,
                target=target_train,
                feature_reg_loss=out_train["feature_reg_loss"],
                cl_h1=h1_train,
                cl_h2=h2_train,
                scale_logits=cdpss_train["scale_logits"],
                graph_stats=graph_stats_train,
                cl_edge_time_ids=edges_train[0],
            )
            loss_train = loss_dict["total"]
            train_metrics = util.compute_metrics(output_train, target_train, edges_train)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                cdpss.eval()
                out_val = encode_multiscale(model, A_val_views, val_edge_nodes, M, False)
                h2_val = encode_short_for_cl(model, A_val_views, val_edge_nodes, M)
                H_val = out_val["node_emb_seq"]
                cdpss_val = run_cdpss(cdpss, H_val, edge_index_val, val_edge_nodes, graph_stats_val)
                output_val = cdpss_val["final_logits"]
                loss_val_dict = criterion(
                    final_logits=output_val[-K_val:],
                    target=target_val[-K_val:],
                    feature_reg_loss=out_val["feature_reg_loss"],
                    cl_h1=H_val["short"],
                    cl_h2=h2_val,
                    scale_logits=cdpss_val["scale_logits"][-K_val:],
                    graph_stats=graph_stats_val,
                    cl_edge_time_ids=edges_val[0, -K_val:],
                )
                loss_val = loss_val_dict["total"]
                val_metrics = util.compute_metrics(output_val[-K_val:], target_val[-K_val:], edges_val[:, -K_val:])

            log = util.log_metric("Train", ep, train_metrics, loss_train)
            log += util.log_metric("Val", ep, val_metrics, loss_val)
            alpha_val_mean = cdpss_val["alpha"].mean(dim=0)
            log += f"alpha_s: {alpha_val_mean[0]:.4f} alpha_m: {alpha_val_mean[1]:.4f} alpha_l: {alpha_val_mean[2]:.4f}  "
            lora_gate_val = cdpss_val.get("lora_gate")
            if lora_gate_val is not None:
                gate_mean = lora_gate_val.mean(dim=0)
                gate_std = lora_gate_val.std(dim=0)
                log += (
                    f"gamma_g: {cdpss.gamma_g.item():.6f} "
                    f"gate_s: {gate_mean[0]:.4f}/{gate_std[0]:.4f} "
                    f"gate_m: {gate_mean[1]:.4f}/{gate_std[1]:.4f} "
                    f"gate_l: {gate_mean[2]:.4f}/{gate_std[2]:.4f}  "
                )
            log += f"L_edge: {loss_dict['edge'].item():.6f} "
            log += f"L_c: {loss_dict['contrastive'].item():.6f} "
            log += f"L_x: {loss_dict['x'].item():.6f} "
            log += f"D_c: {loss_dict['cl_disagreement'].item():.6f} "

            print(log)
            epoch_logs.append(log)

            val_metric_indicator = [(name, value, True) for name, value in val_metrics.items()]
            if early_stopping.step(val_metric_indicator, model):
                break

        synchronize_if_needed(device)
        training_seconds = time.perf_counter() - train_start
        per_iteration_seconds = training_seconds / max(epochs_trained, 1)
        time_log = (
            f"Run {run + 1}. Training time: {training_seconds:.6f} seconds "
            f"Epochs: {epochs_trained} "
            f"Per-Iteration Time Cost (Seconds): {per_iteration_seconds:.6f}"
        )
        print(time_log)
        epoch_logs.append(time_log)

        test_results = {}
        for metric_name in ("AP", "ROC_AUC"):
            early_stopping.load_checkpoint(model, metric_name)
            early_stopping.load_checkpoint(cdpss, metric_name, suffix="_cdpss")
            model.eval()
            cdpss.eval()
            with torch.no_grad():
                out_test = encode_multiscale(model, A_test_views, test_edge_nodes, M, False)
                cdpss_test = run_cdpss(cdpss, out_test["node_emb_seq"], edge_index_test, test_edge_nodes, graph_stats_test)
                output_test = cdpss_test["final_logits"]
            test_metric = util.compute_metrics(output_test[-K_test:], target_test[-K_test:], edges_test[:, -K_test:])
            log = (
                f"Test {metric_name}: {test_metric.get(metric_name)} in Val "
                f"{metric_name}: {early_stopping.best_metrics.get(metric_name)}"
            )
            print(log)
            epoch_logs.append(log)
            test_results[metric_name] = {
                "test": float(test_metric.get(metric_name)),
                "validation": float(early_stopping.best_metrics.get(metric_name)),
            }

        print(f"Finished run {run}.")
        run_results.append(
            {
                "run": run,
                "training_seconds": float(training_seconds),
                "epochs_trained": int(epochs_trained),
                "per_iteration_seconds": float(per_iteration_seconds),
                "best_validation": {k: float(v) for k, v in early_stopping.best_metrics.items()},
                "test": test_results,
                "checkpoint_dir": save_model_folder,
            }
        )

    return run_results


def main():
    args = get_link_prediction_args()
    train(args)


if __name__ == "__main__":
    main()

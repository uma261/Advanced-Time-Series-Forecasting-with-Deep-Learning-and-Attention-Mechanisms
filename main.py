"""
End-to-end script:
- Generates synthetic multivariate data
- Trains baseline LSTM and LSTM+Attention
- Evaluates RMSE/MAE on test set
"""
import argparse, os, torch, time
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import seed_everything, generate_synthetic_multivariate, SeqDataset, rmse, mae
from model import LSTMAttention, LSTMBaseline
from tqdm import trange

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        preds = model(xb)
        # model may return (out, attn) or out
        if isinstance(preds, tuple):
            out = preds[0]
        else:
            out = preds
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device):
    model.eval()
    ys = []
    yps = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if isinstance(model(xb), tuple):
                out = model(xb)[0]
            else:
                out = model(xb)
            ys.append(yb.numpy())
            yps.append(out.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yps, axis=0)
    return y_true, y_pred

def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    data = generate_synthetic_multivariate(n_series=args.n_series, length=args.length, trend=args.trend, noise_std=args.noise, seed=args.seed)
    # split
    train_frac = 0.7; val_frac = 0.15
    n = data.shape[0]
    nt = int(n*train_frac)
    nv = int(n*(train_frac+val_frac))
    train = data[:nt]
    val = data[nt:nv]
    test = data[nv:]
    # datasets
    ds_train = SeqDataset(train, input_len=args.input_len, horizon=args.horizon)
    ds_val = SeqDataset(val, input_len=args.input_len, horizon=args.horizon)
    ds_test = SeqDataset(test, input_len=args.input_len, horizon=args.horizon)
    trn_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
    input_size = data.shape[1]
    output_size = data.shape[1]
    # models
    attn = LSTMAttention(input_size, args.hidden, output_size, num_layers=args.layers, dropout=args.dropout).to(device)
    base = LSTMBaseline(input_size, args.hidden, output_size, num_layers=args.layers, dropout=args.dropout).to(device)
    # optimizers
    opt_attn = optim.Adam(attn.parameters(), lr=args.lr)
    opt_base = optim.Adam(base.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best_val_loss_attn = 1e9
    best_val_loss_base = 1e9
    for epoch in trange(args.epochs, desc="Epochs"):
        l_attn = train_one_epoch(attn, trn_loader, opt_attn, loss_fn, device)
        l_base = train_one_epoch(base, trn_loader, opt_base, loss_fn, device)
        # val
        _, pred_val_attn = eval_model(attn, val_loader, device)
        _, pred_val_base = eval_model(base, val_loader, device)
        # compute val mse
        yv = np.concatenate([yb.numpy() for xb,yb in val_loader], axis=0)
        val_mse_attn = ((yv - pred_val_attn)**2).mean()
        val_mse_base = ((yv - pred_val_base)**2).mean()
        if val_mse_attn < best_val_loss_attn:
            best_val_loss_attn = val_mse_attn
            torch.save(attn.state_dict(), os.path.join(args.out_dir, "best_attention.pt"))
        if val_mse_base < best_val_loss_base:
            best_val_loss_base = val_mse_base
            torch.save(base.state_dict(), os.path.join(args.out_dir, "best_lstm.pt"))
        if (epoch+1) % max(1, args.epochs//5) == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | TrnLoss(attn):{l_attn:.5f} TrnLoss(base):{l_base:.5f} ValMSE(attn):{val_mse_attn:.6f} ValMSE(base):{val_mse_base:.6f}")
    # load best and evaluate on test
    attn.load_state_dict(torch.load(os.path.join(args.out_dir, "best_attention.pt"), map_location=device))
    base.load_state_dict(torch.load(os.path.join(args.out_dir, "best_lstm.pt"), map_location=device))
    y_test, p_attn = eval_model(attn, test_loader, device)
    _, p_base = eval_model(base, test_loader, device)
    # compute metrics (multivariate: average across features)
    from utils import rmse, mae
    attn_rmse = rmse(y_test, p_attn)
    attn_mae = mae(y_test, p_attn)
    base_rmse = rmse(y_test, p_base)
    base_mae = mae(y_test, p_base)
    print("Test metrics (Attention): RMSE: {:.6f}, MAE: {:.6f}".format(attn_rmse, attn_mae))
    print("Test metrics (Baseline LSTM): RMSE: {:.6f}, MAE: {:.6f}".format(base_rmse, base_mae))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length", type=int, default=2000)
    parser.add_argument("--n_series", type=int, default=3)
    parser.add_argument("--input_len", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--trend", type=float, default=0.001)
    parser.add_argument("--noise", type=float, default=0.1)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)

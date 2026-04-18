# uconn_io.py
import os, json, datetime
import torch
from safetensors.torch import save_file, load_file

def _now_iso():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

def save_u_conn(
    path: str,
    u_conn: torch.Tensor,
    conn_ids: torch.Tensor,
    meta: dict | None = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensors = {
        "u_conn": u_conn.detach().to("cpu").to(torch.float32),
        "conn_ids": conn_ids.detach().to("cpu").to(torch.int64),
    }
    metadata = {
        "format": "u_conn_v1",
        "dim": str(list(u_conn.shape)),
        "dtype": "float32",
        "conn_ids_dtype": "int64",
        "created_at": _now_iso(),
    }
    if meta:
        metadata["meta_json"] = json.dumps(meta, ensure_ascii=False)
    save_file(tensors, path, metadata=metadata)

def load_u_conn(
    path: str,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    data = load_file(path, device="cpu")
    print(data)
    if "u_hneg" in data:
        u = data["u_hneg"]
    else:
        u = data["u_conn"]

    if device is not None:
        u = u.to(device)

    if dtype is not None:
        u = u.to(dtype=dtype)

    return u

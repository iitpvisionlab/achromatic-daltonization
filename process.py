import pickle

from tqdm import tqdm

import numpy as np
import torch
from torch import Tensor

from utils import linearRGB_from_sRGB, get_dw, get_sign_guide
from utils import M2, M1, M1_diff
from typing import TypedDict

class ProcessedData(TypedDict):
    file_name: str
    original: Tensor
    original_sim: Tensor
    loss_graph: Tensor
    watermark: Tensor
    daltonized: Tensor


def run_ma(
    dataloader,
    device,
    out_path,
    opt_name,
    eps,
    num_epochs,
    sign_g,
    avg_v_fill,
    sim_func,
    sim_func_name,
    px,
    cvdtype,
    n_data,
    h_max,
    w_max,
    method,
    thresh,
    LR,
) -> list[ProcessedData]:
    # dataset_processed = {"file_name":[], "original":[], "original_sim":[],
    #                       "loss_graph": [], "watermark": []};
    print(f"Starting {sign_g}..")
    inputs, _ = next(iter(dataloader))
    result = np.zeros((n_data, h_max, w_max))
    curr_b = 0
    for inputs, file_names in dataloader:
        losses = []
        b, c, h, w = inputs.shape
        # watermark
        x = torch.ones((b, h, w), requires_grad=True, device=device)
        if opt_name == "Adam":
            optimizer = torch.optim.Adam([x], lr=LR)

        inputs = inputs.to(device)

        # N, H, W
        sign_guide = get_sign_guide(
            inputs, device, sign_g, sim_func_name=sim_func_name, CVD_type=cvdtype
        )

        dw_x = get_dw(
            inputs, device, sim_func, sign_guide, px, 2, method, thresh, avg_v_fill
        )
        dw_y = get_dw(
            inputs, device, sim_func, sign_guide, px, 1, method, thresh, avg_v_fill
        )

        loss = M2(x, dw_y, dw_x, px, eps).sum()
        print(f"Initial Loss: {loss.item()}")

        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            loss = M2(x, dw_y, dw_x, px, eps)
            losses.append(loss.cpu().detach().numpy())
            loss = loss.sum()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % (num_epochs // 10) == 0:
                print(f"Mean Loss: {loss.item()}")
        losses = np.array(losses)

        x = x.detach().cpu().numpy()
        originals = inputs
        originals_sim = sim_func(originals.permute(0, 2, 3, 1))
        dataset_processed = []
        for i in range(len(inputs)):
            original = originals[i].cpu().numpy()
            original_sim = originals_sim[i].cpu().numpy()

            ret: ProcessedData = {
                "file_name": file_names[i],
                "original": original,
                "original_sim": original_sim,
                "loss_graph": losses[:, i],
                "watermark": x[i] 
            }
            dataset_processed.append(
                ret
            )
        # savings
        result[curr_b : curr_b + b] = x
        curr_b += b
        # with open(f"{out_path}", "wb") as handle:
        #     pickle.dump(dataset_processed, handle)
    return dataset_processed


def append_to_dic_ma(dictionary, file_name, original, original_sim, loss_graph, wmark):
    dictionary["file_name"].append(file_name)
    dictionary["original"].append(original)
    dictionary["original_sim"].append(original_sim)
    dictionary["loss_graph"].append(loss_graph)
    dictionary["watermark"].append(wmark)


def run_o_ma(
    dataloader,
    device,
    ma_path,
    out_path,
    opt_name,
    eps,
    num_epochs,
    sim_func,
    sim_func_name,
    px,
    n_data,
    h_max,
    w_max,
    LR,
) -> list[ProcessedData]:
    dataset_processed = {
        "file_name": [],
        "original": [],
        "original_sim": [],
        "loss_graph": [],
        "watermark": [],
    }
    if not ma_path is None:
        with open(ma_path, "rb") as handle:
            init_watermark = np.array(pickle.load(handle)["watermark"])

    result = np.zeros((n_data, h_max, w_max))
    curr_b = 0
    for inputs, file_names in dataloader:
        losses = []
        b, c, h, w = inputs.shape
        # watermark
        if not ma_path is None:
            x = torch.tensor(
                init_watermark[curr_b : curr_b + b], requires_grad=True, device=device
            )
        else:
            x = torch.ones((b, h, w), requires_grad=True, device=device)
        if opt_name == "Adam":
            optimizer = torch.optim.Adam([x], lr=LR)

        inputs = inputs.to(device)

        # N, C, H, W
        grad_x_in = M1_diff(inputs, px, 3)
        grad_y_in = M1_diff(inputs, px, 2)
        # N, C, H, W
        inNorm2_x = torch.sum(grad_x_in**2, dim=1)
        inNorm2_y = torch.sum(grad_y_in**2, dim=1)

        loss = M1(x, px, inputs, inNorm2_x, inNorm2_y, sim_func, eps)
        loss = loss.sum()
        print(f"Initial Loss: {loss.mean()}")

        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            loss = M1(x, px, inputs, inNorm2_x, inNorm2_y, sim_func, eps)
            losses.append(loss.cpu().detach().numpy())
            loss = loss.sum()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % (num_epochs // 10) == 0:
                print(f"Mean Loss: {loss.item()}")

        losses = np.array(losses)
        print(losses.shape)
        x = x.detach().cpu().numpy()
        originals = inputs
        originals_sim = sim_func(originals.permute(0, 2, 3, 1))
        dataset_processed = list[ProcessedData]
        for i, img in enumerate(inputs):
            original = originals[i].cpu().numpy()
            original_sim = originals_sim[i].cpu().numpy()
            dataset_processed.append(
                ProcessedData(file_names[i], original, original_sim, losses[:, i], x[i])
            )
            # append_to_dic_ma(
            #     dataset_processed,
            #     file_names[i],
            #     original,
            #     original_sim,
            #     losses[:, i],
            #     x[i],
            # )
        # savings
        result[curr_b : curr_b + b] = x
        curr_b += b
        # with open(f"{out_path}", "wb") as handle:
        #     pickle.dump(dataset_processed, handle)
    return dataset_processed

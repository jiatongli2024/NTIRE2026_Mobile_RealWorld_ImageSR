import os.path
import logging
import torch
import argparse
import json
import glob
import importlib

from pprint import pprint
from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util


def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        # DAT baseline, ICCV 2023
        from models.team00_DAT import main as DAT
        name = f"{model_id:02}_DAT_baseline"
        model_path = os.path.join('model_zoo', 'team00_dat.pth')
        model_func = DAT
    if model_id == 1:
        from models.team01_VEPG import main as VEPG
        name = f"{model_id:02}_VEPG"
        model_path = os.path.join('model_zoo', 'team01_VEPG')
        model_func = VEPG
    elif model_id == 5:
        from models.team05_NoReject import main as DRRE
        name = f"{model_id:02}_NoReject"
        model_path = os.path.join('model_zoo', 'team05_NoReject', 'DRRE.pkl')
        model_func = DRRE
    elif model_id == 6:
        from models.team06_Antman import main as ESRGAN
        name = f"{model_id:02}_Antman"
        model_path = os.path.join('model_zoo', 'team06_Antman')
        model_func = ESRGAN
    # elif model_id == 10:
    #     from models.team10_SFVision.inference_super_sr import main as SFVision
    #     name = f"{model_id:02}_SFVision"
    #     model_path = os.path.join('model_zoo', 'team10_SFVision.pth')
    #     model_func = SFVision
    elif model_id == 8:
        from models.team08_Super03 import main as Super03
        name = f"{model_id:02}_Super03"
        model_path = os.path.join('model_zoo', 'team08_Super03', 'team08_Super03.pkl')
        model_func = Super03
    elif model_id == 9:
        from models.team09_SNOWVision import main as SNOWVision
        name = f"{model_id:02}_SNOWVision"
        model_path = os.path.join('model_zoo', 'team09_SNOWVision', 'best.pth')
        model_func = SNOWVision
    elif model_id == 10:
        from models.team10_ACM_HCC import main as ACM_HCC
        name = f"{model_id:02}_ACM_HCC"
        model_path = os.path.join('model_zoo', 'team10_ACM_HCC', 'model_final.pkl')
        model_func = ACM_HCC
    elif model_id == 11:
        from models.team11_SamsungAKCamera import main as SamsungAKCamera
        name = f"{model_id:02}_SamsungAICamera"
        model_path = os.path.join('model_zoo', 'team11_SamsungAKCamera', 'team11_mobile_samsung')
        model_func = SamsungAKCamera
    elif model_id == 12:
        from models.team12_BVISR.TADSR.test_tadsr import main as BVISR
        name = f"{model_id:02}_BVISR"
        model_path = os.path.join('model_zoo', 'team12_BVISR', 'team12_BVISR.pkl')
        model_func = BVISR
    elif model_id == 13:
        team13_io = importlib.import_module("models.team13_MDAP.io")
        name = f"{model_id:02}_MDAP"
        model_path = os.path.join('model_zoo', 'team13_MDAP')
        model_func = team13_io.main
    elif model_id == 15:
        # Team 20 YuFans: DiffBIR-RealESRGAN Blend
        from models.team15_YuFans import main as BlendSR
        name = f"{model_id:02}_YuFans"
        model_path = os.path.join('model_zoo', 'team15_YuFans')
        model_func = BlendSR
    elif model_id == 16:
        from models.team16_EIC_ECNU import omgsr_inference as EIC_ECNU
        name = f"{model_id:02}_EIC_ECNU"
        model_path = os.path.join('model_zoo', 'team16_EIC_ECNU')
        model_func = EIC_ECNU
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    return model_func, model_path, name


def run(model_func, model_name, model_path, device, args, mode="test"):
    # --------------------------------
    # dataset path
    # --------------------------------
    if mode == "valid":
        data_path = args.valid_dir
    elif mode == "test":
        data_path = args.test_dir
    assert data_path is not None, "Please specify the dataset path for validation or test."
    
    flat_output_ids = {13, 14, 17, 18, 21}
    if args.model_id in flat_output_ids:
        save_path = os.path.join(args.save_dir, mode)
    else:
        save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    model_func(model_dir=model_path, input_path=data_path, output_path=save_path, device=device)
    end.record()
    torch.cuda.synchronize()
    print(f"Model {model_name} runtime (Including I/O): {start.elapsed_time(end)} ms")


def main(args):
    utils_logger.logger_info("NTIRE2026-MobileSR", log_path="NTIRE2026-MobileSR.log")
    logger = logging.getLogger("NTIRE2026-MobileSR")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model_func, model_path, model_name = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if args.valid_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="valid")
        
    if args.test_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2026-MobileSR")
    parser.add_argument("--valid_dir", default=None, type=str, help="Path to the validation set")
    parser.add_argument("--test_dir", default=None, type=str, help="Path to the test set")
    parser.add_argument("--save_dir", default="NTIRE2026-MobileSR/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)

    args = parser.parse_args()
    pprint(args)

    main(args)

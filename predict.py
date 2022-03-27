import tempfile
import time
from imageio import imwrite
import torch
import numpy as np
from cog import BasePredictor, Path, Input

# Internal Project Imports
from pretrained.vgg import Vgg16Pretrained
from utils import misc as misc
from utils.misc import load_path_for_pytorch
from utils.stylize import produce_stylization


class Predictor(BasePredictor):
    def setup(self):
        # Define feature extractor
        cnn = misc.to_device(Vgg16Pretrained())
        self.phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

    def predict(
        self,
        content: Path = Input(description="Content image."),
        style: Path = Input(description="Style image."),
        colorize: bool = Input(
            default=True, description="Whether use color correction in the output."
        ),
        high_res: bool = Input(
            default=False,
            description="Whether output high resolution image (1024 instead if 512).",
        ),
        alpha: float = Input(
            default=0.75,
            ge=0.0,
            le=1.0,
            description="alpha=1.0 corresponds to maximum content preservation, alpha=0.0 is maximum stylization.",
        ),
        content_loss: bool = Input(
            default=False, description="Whether use experimental content loss."
        ),
    ) -> Path:

        max_scls = 4
        sz = 512
        if high_res:
            max_scls = 5
            sz = 1024

        flip_aug = True
        misc.USE_GPU = True
        content_weight = 1.0 - alpha

        # Error checking for arguments
        # error checking for paths deferred to imageio
        assert (0.0 <= content_weight) and (
            content_weight <= 1.0
        ), "alpha must be between 0 and 1"
        assert torch.cuda.is_available() or (
            not misc.USE_GPU
        ), "attempted to use gpu when unavailable"

        # Load images
        content_im_orig = misc.to_device(
            load_path_for_pytorch(str(content), target_size=sz)
        ).unsqueeze(0)
        style_im_orig = misc.to_device(
            load_path_for_pytorch(str(style), target_size=sz)
        ).unsqueeze(0)

        # Run Style Transfer
        torch.cuda.synchronize()
        start_time = time.time()
        output = produce_stylization(
            content_im_orig,
            style_im_orig,
            self.phi,
            max_iter=200,
            lr=2e-3,
            content_weight=content_weight,
            max_scls=max_scls,
            flip_aug=flip_aug,
            content_loss=content_loss,
            dont_colorize=not colorize,
        )
        torch.cuda.synchronize()
        print("Done! total time: {}".format(time.time() - start_time))

        # Convert from pyTorch to numpy, clip to valid range
        new_im_out = np.clip(
            output[0].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0
        )

        # Save stylized output
        save_im = (new_im_out * 255).astype(np.uint8)
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        imwrite("ooo.png", save_im)
        imwrite(str(out_path), save_im)

        # Free gpu memory in case something else needs it later
        if misc.USE_GPU:
            torch.cuda.empty_cache()

        return out_path

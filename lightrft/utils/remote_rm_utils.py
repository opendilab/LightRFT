import base64
import io
import time
import requests
import torch

from PIL import Image
from typing import Any, Dict, List, Optional, Union

from .logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url: str,
                        data: Dict[str, Any],
                        score_key: str = "rewards",
                        try_max_times: int = 5) -> Union[float, List[float]]:
    """
    Synchronous request API wrapper for reward model scoring.

    This function makes HTTP POST requests to a reward model API endpoint and
    handles retries with exponential backoff for failed requests.

    :param url: The API endpoint URL to send requests to
    :type url: str
    :param data: The request payload data as a dictionary
    :type data: Dict[str, Any]
    :param score_key: The key in the response JSON that contains the reward scores
    :type score_key: str
    :param try_max_times: Maximum number of retry attempts for failed requests
    :type try_max_times: int
    :return: Reward scores extracted from the API response, either as a single float
             or a list of floats depending on the API response structure
    :rtype: Union[float, List[float]]
    :raises Exception: When all retry attempts fail after the maximum number of tries

    Example::

        score = request_api_wrapper(
            url="http://localhost:8000/score",
            data={"text": "Hello world"},
            score_key="rewards",
            try_max_times=5
        )
    """
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=3000)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
            logger.info(f"Request error data: {data}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(
    api_url: str,
    queries: List[str],
    prompts: List[str],
    labels: Optional[List[Any]] = None,
    references: Optional[List[str]] = None,
    raw_images: Optional[List[Optional[Union[Image.Image, List[Image.Image]]]]] = None,
    score_key: str = "rewards"
) -> torch.Tensor:
    """
    Remote reward model API function for scoring text and image inputs.

    This function prepares data and sends requests to a remote reward model API,
    supporting both text-only and multimodal (text + image) scoring scenarios.

    :param api_url: Reward model API endpoint URL
    :type api_url: str
    :param queries: List of query strings with response templates
    :type queries: List[str]
    :param prompts: List of prompt strings for context
    :type prompts: List[str]
    :param labels: Optional list of labels for supervised scoring (currently unused)
    :type labels: Optional[List[Any]]
    :param references: Optional list of reference responses for comparison scoring
    :type references: Optional[List[str]]
    :param raw_images: Optional list of PIL Image objects or lists of PIL Image objects
                      for multimodal scoring. Each element can be None, a single image,
                      or a list of images.
    :type raw_images: Optional[List[Optional[Union[Image.Image, List[Image.Image]]]]]
    :param score_key: Key in the API response that contains the reward scores
    :type score_key: str
    :return: Tensor of reward scores for all input samples
    :rtype: torch.Tensor
    :raises Exception: When API requests fail after maximum retry attempts

    Example::

        # Text-only scoring
        scores = remote_rm_fn(
            api_url="http://localhost:8000/score",
            queries=["What is 2+2?"],
            prompts=["Calculate the following:"],
            score_key="rewards"
        )

        # Multimodal scoring with images
        scores = remote_rm_fn(
            api_url="http://localhost:8000/score",
            queries=["Describe this image"],
            prompts=["Please analyze the image:"],
            raw_images=[Image.open("image.jpg")],
            score_key="rewards"
        )
    """
    data = {"queries": queries, "prompts": prompts}
    if references is not None:
        data["references"] = references
    if raw_images is not None:
        # print(f"=================raw_images: {raw_images}")
        # Convert PIL images to bytes then to base64 strings
        base64_images = []
        for imgs in raw_images:
            base64_imgs = []
            if imgs is None:
                base64_images.append(None)
                continue
            with io.BytesIO() as buf:
                if isinstance(imgs, list):
                    for img in imgs:
                        if img.mode == "RGBA":
                            img = img.convert("RGB")  # to RGB
                        img.save(buf, format='JPEG')
                        base64_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
                    base64_images.append(base64_imgs)
                else:
                    if imgs.mode == "RGBA":
                        imgs = imgs.convert("RGB")  # to RGB
                    imgs.save(buf, format='JPEG')
                    base64_imgs.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
                    base64_images.append(base64_imgs)
        data["images"] = base64_images
    scores = request_api_wrapper(url=api_url, data=data, score_key=score_key)
    return torch.tensor(scores)

import os
import argparse
import cv2
import copy
import glob
import random
import numpy as np
from enum import Enum
import albumentations as A
import fsspec
import webdataset as wds
import logging
from multiprocessing import Process
import time
from torch.utils.data import IterableDataset

logging.basicConfig(level=logging.INFO)


_INTER_STR_TO_CV2 = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "bilinear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
    "lanczos4": cv2.INTER_LANCZOS4,
}

def inter_str_to_cv2(inter_str):
    inter_str = inter_str.lower()
    if inter_str not in _INTER_STR_TO_CV2:
        raise ValueError(f"Invalid option for interpolation: {inter_str}")
    return _INTER_STR_TO_CV2[inter_str]

def split_number_to_index_list(total_size, chunk_size):
    chunks = [[] for _ in range((total_size + chunk_size - 1) // chunk_size)]
    for i in range(total_size):
        chunks[i // chunk_size].append(i)
    return chunks


class ResizeMode(Enum):
    no = 0  # pylint: disable=invalid-name
    keep_ratio = 1  # pylint: disable=invalid-name
    center_crop = 2  # pylint: disable=invalid-name
    border = 3  # pylint: disable=invalid-name
    keep_ratio_largest = 4  # pylint: disable=invalid-name


class Resizer:
    def __init__(
        self,
        image_size,
        resize_mode,
        resize_only_if_bigger,
        upscale_interpolation="lanczos",
        downscale_interpolation="area",
        encode_quality=95,
        encode_format="jpg",
        skip_reencode=False,
        min_image_size=0,
        max_image_area=float("inf"),
        max_aspect_ratio=float("inf"),
    ):
        self.image_size = image_size

        if isinstance(resize_mode, str):
            if resize_mode not in ResizeMode.__members__:  # pylint: disable=unsupported-membership-test
                raise ValueError(f"Invalid option for resize_mode: {resize_mode}")
            resize_mode = ResizeMode[resize_mode]

        self.resize_mode = resize_mode
        self.min_image_size = min_image_size
        self.max_image_area = max_image_area
        self.max_aspect_ratio = max_aspect_ratio
        self.resize_only_if_bigger = resize_only_if_bigger
        self.encode_format = encode_format
        cv2_img_quality = int(cv2.IMWRITE_JPEG_QUALITY)
        self.encode_params = [cv2_img_quality, encode_quality]
        self.what_ext = "jpeg"
        self.skip_reencode = skip_reencode

        self.upscale_interpolation = inter_str_to_cv2(upscale_interpolation)
        self.downscale_interpolation = inter_str_to_cv2(downscale_interpolation)


    def __call__(self, img):
        cv2.setNumThreads(1)
        if img is None:
            raise ValueError("Image decoding error")
        if len(img.shape) == 3 and img.shape[-1] == 4:
            # alpha matting with white background
            alpha = img[:, :, 3, np.newaxis]
            img = alpha / 255 * img[..., :3] + 255 - alpha
            img = np.rint(img.clip(min=0, max=255)).astype(np.uint8)
            encode_needed = True
        original_height, original_width = img.shape[:2]
        # check if image is too small
        if min(original_height, original_width) < self.min_image_size:
            return None, None, None, None, None, "image too small"
        if original_height * original_width > self.max_image_area:
            return None, None, None, None, None, "image area too large"
        # check if wrong aspect ratio
        if max(original_height, original_width) / min(original_height, original_width) > self.max_aspect_ratio:
            return None, None, None, None, None, "aspect ratio too large"

        # resizing in following conditions
        if self.resize_mode in (ResizeMode.keep_ratio, ResizeMode.center_crop):
            downscale = min(original_width, original_height) > self.image_size
            if not self.resize_only_if_bigger or downscale:
                interpolation = self.downscale_interpolation if downscale else self.upscale_interpolation
                img = A.smallest_max_size(img, self.image_size, interpolation=interpolation)
                if self.resize_mode == ResizeMode.center_crop:
                    img = A.center_crop(img, self.image_size, self.image_size)
                encode_needed = True
        elif self.resize_mode in (ResizeMode.border, ResizeMode.keep_ratio_largest):
            downscale = max(original_width, original_height) > self.image_size
            if not self.resize_only_if_bigger or downscale:
                interpolation = self.downscale_interpolation if downscale else self.upscale_interpolation
                img = A.longest_max_size(img, self.image_size, interpolation=interpolation)
                if self.resize_mode == ResizeMode.border:
                    img = A.pad(
                        img,
                        self.image_size,
                        self.image_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=[255, 255, 255],
                    )
                encode_needed = True

        height, width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if encode_needed:
            img_str = cv2.imencode(f".{self.encode_format}", img, params=self.encode_params)[1].tobytes()
        else:
            img_str = img.tobytes()
        return img_str, width, height, original_width, original_height, None


class WebDatasetSampleWriter:
    """WebDatasetSampleWriter is a image+caption writer to webdataset"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        encode_format,
    ):
        self.oom_shard_count = oom_shard_count
        shard_name = f"{shard_id:0{oom_shard_count}d}".format(shard_id=shard_id, oom_shard_count=oom_shard_count)
        self.shard_id = shard_id
        fs, output_path = fsspec.core.url_to_fs(output_folder)
        self.tar_fd = fs.open(f"{output_path}/{shard_name}.tar", "wb")
        self.tarwriter = wds.TarWriter(self.tar_fd)
        self.save_caption = save_caption
        self.encode_format = encode_format

    def write(self, key, img_str, caption, parent_info):
        """write sample to tars"""
        number_of_parents = len(parent_info)
        if img_str is not None:
            sample = {
                "__key__": key, 
                "child."+self.encode_format: img_str,
                "child.txt" : str(caption) if caption is not None else "",
                "numparents.txt": str(number_of_parents),
                }
                
            for i in range(number_of_parents):
                parent_img_str, parent_caption = parent_info[i]
                sample[f"parent{i:03d}."+self.encode_format] = parent_img_str
                sample[f"parent{i:03d}.txt"] = str(parent_caption) if parent_caption is not None else ""
            self.tarwriter.write(sample)

    def close(self):
        self.tarwriter.close()
        self.tar_fd.close()


class ImageTextWebDataset(IterableDataset):
    """
    Iterable dataset that serves instances from a lot of TAR file shards.
    This class uses `WebDataset <https://github.com/webdataset/webdataset>`_
    internally, and expects TAR files to be arranged in a compatible format.
    """

    def __init__(
        self,
        tarfiles: list,
        # mapper: Callable,
        buffer_size: int = 5000,
        infinite_stream: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            tarfiles: Path(s) or glob-patterns for TAR files in WebDataset format.
            mapper: A callable to transform a single dataset dict (image and
                annotations). May implement data augmentation and tokenization.
            buffer_size: Size of the internal buffer of instances. Data is read
                sequentially from TAR files into this buffer and served randomly.
                Shuffling will be disabled if this is set to zero.
            infinite_stream: Yield an infinite stream of instances if this is
                True. In such cases, the user must terminate this iterator manually
                (e.g. run a fixed sized for-loop in training code).
            seed: Random seed for buffer shuffling. If provided, this dataloader
                will load batches deterministically across different runs (only if
                batch size and number of GPUs/CPUs are same). This seed can either
                be same or different per GPU process for multi-GPU training.
        """
        super().__init__()
        self.buffer_size = buffer_size
        self.infinite_stream = infinite_stream
        self.seed = seed

        # Convert a single path (glob) to a list.
        if isinstance(tarfiles, str):
            tarfiles = [tarfiles]

        # Expand all glob patterns to list a full list of individual TAR files.
        self.tarfiles = []
        for _path in tarfiles:
            for _single_glob in _path.split():
                self.tarfiles.extend(glob.glob(_single_glob))

        # Sort paths; webdataset performs a deterministic shuffle (internally).
        self.tarfiles = sorted(self.tarfiles)
        logger.info(f"{self.__class__.__name__} found {len(self.tarfiles)} TARs.")

    def __iter__(self):
        rng = random.Random(self.seed)
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
        )

        if self.buffer_size > 1:
            pipeline.append(
                wds.shuffle(self.buffer_size, initial=self.buffer_size, rng=rng),
            )

        # Decode images using PIL and apply custom mapper.
        pipeline.append(wds.decode("pil", handler=wds.warn_and_continue))

        if self.infinite_stream:
            # Sample an infinite stream of dataset dicts.
            while True:
                pipeline_copy = copy.deepcopy(pipeline)
                yield from pipeline_copy
        else:
            # Run for one epoch and stop:
            yield from pipeline


def shard_process(shard_id, tar_files, args, resizer, save_caption, oom_shard_count, encode_format):
    logger.info(f"Creating shard {shard_id}")
    start_time = time.time()
    sample_writer = WebDatasetSampleWriter(
        shard_id=shard_id,
        output_folder=args.output_tar_directory,
        save_caption=save_caption,
        oom_shard_count=oom_shard_count,
        encode_format=encode_format,
    )
    crop_dim_cutoff = 32*32
    tarfile = tar_files[shard_id]
    dataset = ImageTextWebDataset(tarfiles=[tarfile], infinite_stream=False)

    for sample in dataset:
        sample_key = sample["__key__"]
        caption = sample['json']['caption']
        pil_image = sample['jpg']
        width, height = pil_image.size
        parent_terms = []
        parent_info = []

        for noun_chunk in sample['json']['noun_chunks']:
            noun_term = caption[int(noun_chunk[0]):int(noun_chunk[1])]
            if noun_term not in parent_terms:
                parent_terms.append(noun_term)
                box = noun_chunk[2:-1]
                box[0] = int(box[0] * width)
                box[1] = int(box[1] * height)
                box[2] = int(box[2] * width)
                box[3] = int(box[3] * height)

                if (box[2] - box[0]) * (box[3] - box[1]) > crop_dim_cutoff:
                    cropped_image = pil_image.crop(box)
                    entity_img_str, _, _, _, _, error = resizer(np.array(cropped_image))
                    parent_info.append((entity_img_str, noun_term))
        
        for ref_exp in sample['json']['ref_exps']:
            ref_exp_term = caption[int(ref_exp[0]):int(ref_exp[1])]
            if ref_exp_term not in parent_terms:
                parent_terms.append(ref_exp_term)
                box = ref_exp[2:-1]
                box[0] = int(box[0] * width)
                box[1] = int(box[1] * height)
                box[2] = int(box[2] * width)
                box[3] = int(box[3] * height)

                if (box[2] - box[0]) * (box[3] - box[1]) > crop_dim_cutoff:
                    cropped_image = pil_image.crop(box)
                    entity_img_str, _, _, _, _, error = resizer(np.array(cropped_image))
                    parent_info.append((entity_img_str, ref_exp_term))
            
        img_str, _, _, _, _, error = resizer(np.array(pil_image))

        if len(parent_info) > 0:
            sample_writer.write(sample_key, img_str, caption, parent_info)

    sample_writer.close()
    logger.info(f"Shard {shard_id} took {time.time() - start_time} seconds") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_webdataset_path", type=str, help="grit raw download directory")
    parser.add_argument("--processed_webdataset_path", type=str, help="output folder for processed webdataset")
    parser.add_argument("--max_num_processes", type=int, default=16, help="max number of processes")

    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    logger.info("Creating webdataset")

    save_caption = True
    encode_format = "jpg"
    oom_shard_count = 5

    resizer = Resizer(
        image_size=224,
        resize_mode="border",
        resize_only_if_bigger=False,
        upscale_interpolation="lanczos",
        downscale_interpolation="area",
        encode_quality=95,
        encode_format=encode_format,
        skip_reencode=False,
        min_image_size=0,
        max_image_area=float("inf"),
        max_aspect_ratio=float("inf"),
    )

    grit_tarfiles = [os.path.join(args.original_wd_path, "*.tar")]
    all_tarfiles = []
    for _path in grit_tarfiles:
        for _single_glob in _path.split():
            all_tarfiles.extend(glob.glob(_single_glob))
    
    total_shards = len(all_tarfiles)
    logger.info(f"Total shards: {total_shards}")

    if not os.path.exists(args.output_tar_directory):
        logger.info(f"Creating output directory: {args.output_tar_directory}")
        os.makedirs(args.output_tar_directory)
    
    shard_list_split = split_number_to_index_list(total_shards, args.max_num_processes)
    logger.info(f"Total splits {len(shard_list_split)}")

    for idx, shard_list in enumerate(shard_list_split):
        logger.info(f"{idx} : {shard_list}")

    for shard_list in shard_list_split:
        processes = [Process(target=shard_process, args=(shard_id, all_tarfiles, 
                                                        args, resizer, 
                                                        save_caption, oom_shard_count, 
                                                        encode_format)) for shard_id in shard_list]

        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
    
    logger.info("Finished creating webdataset")